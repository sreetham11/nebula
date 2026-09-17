"""
End-to-end pipeline runner: for each subsystem (door, bogie) --
  1. load raw sensor data + fault log
  2. compute baseline rolling z-score flags
  3. split healthy units BY UNIT into train / validation (no unit's
     windows appear in both), train the GRU autoencoder on training-unit
     windows only, with early stopping on validation-unit loss
  4. score every unit (healthy + faulty) with the trained autoencoder
  5. calibrate anomaly thresholds from the held-out validation units'
     reconstruction error distribution only (never train-unit error,
     which the model has already fit and would be optimistic)
  6. fuse baseline + autoencoder into a risk level per reading
  7. persist: scored CSV, model + scaler + thresholds, fleet_status.csv
  8. run validation (see validate.py) producing per-fault lead-time plots

Run: python pipeline/run_pipeline.py
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import torch

import data_loader
from baseline import compute_baseline_flags
from autoencoder import (
    fit_scaler, build_windows, train_autoencoder, score_windows, get_device,
    SequenceAutoencoder,
)
from fusion import compute_recon_error_thresholds, fuse_risk
import schema_config as sc
from model_config import AE_WINDOW_LENGTH, AE_WINDOW_STRIDE, AE_SCORING_STRIDE, AE_VAL_UNIT_FRACTION, RANDOM_SEED


def process_subsystem(subsystem_type, fault_log, rng):
    print(f"\n=== {subsystem_type.upper()} ===")
    cfg = sc.SUBSYSTEMS[subsystem_type]
    df = data_loader.load_subsystem_data(subsystem_type)

    healthy_ids = data_loader.healthy_unit_ids(df, subsystem_type, fault_log)
    all_ids = data_loader.all_unit_ids(df, subsystem_type)
    train_units, val_units = data_loader.split_healthy_units(healthy_ids, AE_VAL_UNIT_FRACTION, rng)
    print(f"units: {len(all_ids)} total, {len(healthy_ids)} healthy "
          f"-> {len(train_units)} train / {len(val_units)} held-out val (unit-level split)")
    print(f"  train units: {train_units}")
    print(f"  val units:   {val_units}")

    # 1. Baseline
    df = compute_baseline_flags(df, subsystem_type)
    print(f"baseline flags raised on {df['baseline_flag'].sum():,} / {len(df):,} readings")

    # 2. Autoencoder: fit scaler + train on TRAIN units only, so the held-out
    # val units never leak into either scaling stats or gradient updates.
    scaler = fit_scaler(df, subsystem_type, train_units)
    X_train, _, _ = build_windows(df, subsystem_type, train_units, scaler,
                                   AE_WINDOW_LENGTH, AE_WINDOW_STRIDE)
    # Val windows use the dense scoring stride, both for a more stable
    # early-stopping signal and because the same array doubles as the
    # threshold-calibration sample below.
    X_val, _, _ = build_windows(df, subsystem_type, val_units, scaler,
                                 AE_WINDOW_LENGTH, AE_SCORING_STRIDE)
    print(f"training windows: {X_train.shape}, validation windows: {X_val.shape}")
    model, device, best_train_loss, best_val_loss = train_autoencoder(
        X_train, X_val, n_features=len(cfg["signal_cols"])
    )
    print(f"trained on device={device}: train_loss={best_train_loss:.5f}, val_loss={best_val_loss:.5f}")

    # 3. Score every unit (dense stride) for the full timeline
    X_all, end_index, _unit_of_win = build_windows(df, subsystem_type, all_ids, scaler,
                                                     AE_WINDOW_LENGTH, AE_SCORING_STRIDE)
    recon_errors = score_windows(model, device, X_all)
    df["ae_recon_error"] = np.nan
    df.loc[end_index, "ae_recon_error"] = recon_errors

    # Thresholds are calibrated ONLY from the held-out validation units'
    # reconstruction error (re-scored here fresh, same X_val used in
    # training) -- these units' windows never touched a gradient update,
    # so this reflects genuine generalization error on unseen-but-healthy
    # data, not in-sample error from units the model was fit on.
    val_recon_errors = score_windows(model, device, X_val)
    thresholds = compute_recon_error_thresholds(val_recon_errors)
    print(f"recon error thresholds (from {len(val_units)} held-out healthy unit(s), "
          f"{len(val_recon_errors):,} windows): {thresholds}")

    # 4. Fusion
    df = fuse_risk(df, "ae_recon_error", "baseline_flag", thresholds)

    # 5. Persist scored data
    df.to_csv(cfg["scored_path"], index=False)

    # 6. Persist model artifacts
    os.makedirs(sc.MODELS_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(sc.MODELS_DIR, f"{subsystem_type}_autoencoder.pt"))
    joblib.dump(scaler, os.path.join(sc.MODELS_DIR, f"{subsystem_type}_scaler.joblib"))
    # Raw held-out healthy reconstruction-error array, so a threshold at any
    # arbitrary percentile can be recomputed later (e.g. the dashboard's
    # sensitivity slider) without rerunning the model.
    np.save(os.path.join(sc.MODELS_DIR, f"{subsystem_type}_val_recon_errors.npy"), val_recon_errors)
    with open(os.path.join(sc.MODELS_DIR, f"{subsystem_type}_meta.json"), "w") as f:
        json.dump({
            "thresholds": thresholds,
            "signal_cols": cfg["signal_cols"],
            "window_length": AE_WINDOW_LENGTH,
            "healthy_ids": healthy_ids,
            "train_units": train_units,
            "val_units": val_units,
            "train_loss": best_train_loss,
            "val_loss": best_val_loss,
        }, f, indent=2)

    return df


def build_fleet_status(door_df, bogie_df, fault_log):
    rows = []
    for subsystem_type, df in [("door", door_df), ("bogie", bogie_df)]:
        cfg = sc.SUBSYSTEMS[subsystem_type]
        id_col, ts_col = cfg["id_col"], cfg["timestamp_col"]
        for uid, g in df.groupby(id_col):
            g = g.sort_values(ts_col)
            latest = g.iloc[-1]
            has_fault = uid in data_loader.faulty_unit_ids(fault_log, subsystem_type)
            rows.append({
                "unit_id": uid,
                "subsystem_type": subsystem_type,
                "train_id": latest[cfg["train_col"]],
                "latest_timestamp": latest[ts_col],
                "risk_level": latest["risk_level"] if pd.notna(latest["risk_level"]) else "Normal",
                "ae_recon_error": latest["ae_recon_error"],
                "baseline_max_abs_z": latest["baseline_max_abs_z"],
                "has_ground_truth_fault": has_fault,
            })
    fleet = pd.DataFrame(rows).sort_values(["subsystem_type", "unit_id"]).reset_index(drop=True)
    fleet.to_csv(sc.FLEET_STATUS_PATH, index=False)
    return fleet


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    fault_log = data_loader.load_fault_log()
    door_df = process_subsystem("door", fault_log, rng)
    bogie_df = process_subsystem("bogie", fault_log, rng)
    fleet = build_fleet_status(door_df, bogie_df, fault_log)

    print("\n=== FLEET STATUS SUMMARY ===")
    print(fleet["risk_level"].value_counts())
    print(f"\nSaved: {sc.FLEET_STATUS_PATH}")

    print("\n=== Running validation ===")
    import validate
    validate.run_validation(door_df, bogie_df, fault_log)


if __name__ == "__main__":
    main()
