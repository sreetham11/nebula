"""
Step 3 (classification) + Step 4 (validation): adapts the existing
healthy-only autoencoder + percentile-threshold approach (pipeline/
autoencoder.py, pipeline/fusion.py's calibration method) to Door segments.

Split is BY SEGMENT, not by row -- the same leakage class already caught
once this session (train/val must never share the same underlying entity
across the split; here that entity is a segment, not a unit). No segment's
rows appear in both the AE's training set and the held-out validation set
used for threshold calibration / evaluation.
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Appended, not inserted at position 0 -- pipeline/ has its own
# data_loader.py, and this project's local data_loader.py (imported below)
# must win that name collision, which requires this project's own
# directory (already sys.path[0] for the running script) to be searched
# first.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline"))
from autoencoder import SequenceAutoencoder, train_autoencoder, score_windows, get_device  # noqa: E402

import config
import data_loader
import segmentation
import features

POSITIVE_LABEL = "Abnormal resistance"


def split_segments(labeled_segments, val_fraction=config.VAL_FRACTION, seed=config.RANDOM_SEED):
    """
    Normal segments are split train/val; ALL Abnormal segments go to val
    only (the autoencoder is healthy-only trained by design, so it never
    sees an Abnormal segment during training regardless of split fraction).
    """
    normal = labeled_segments[labeled_segments["status"] == "Normal"]
    abnormal = labeled_segments[labeled_segments["status"] == POSITIVE_LABEL]

    normal_train, normal_val = train_test_split(
        normal, test_size=val_fraction, random_state=seed, shuffle=True,
    )
    val = pd.concat([normal_val, abnormal]).sort_index()
    return normal_train, val


def fit_scaler(X_train_normal):
    scaler = StandardScaler()
    n, t, f = X_train_normal.shape
    scaler.fit(X_train_normal.reshape(n * t, f))
    return scaler


def scale_windows(X, scaler):
    n, t, f = X.shape
    return scaler.transform(X.reshape(n * t, f)).reshape(n, t, f).astype(np.float32)


def run(save_artifacts=True, verbose=True):
    raw_train = data_loader.load_raw(config.TRAIN_PATH)
    answer = data_loader.load_segments_answer()

    detected = segmentation.detect_segments(raw_train)
    seg_eval = segmentation.evaluate_segmentation(detected, answer)

    labeled = features.attach_true_labels(detected, answer)
    normal_train_segs, val_segs = split_segments(labeled)

    if verbose:
        print(f"segment split: {len(normal_train_segs)} Normal train / "
              f"{len(val_segs)} held-out val "
              f"({(val_segs['status'] == 'Normal').sum()} Normal + "
              f"{(val_segs['status'] == POSITIVE_LABEL).sum()} Abnormal)")

    X_train_raw, _ = features.extract_segment_windows(raw_train, normal_train_segs)
    X_val_raw, val_seg_ids = features.extract_segment_windows(raw_train, val_segs)

    scaler = fit_scaler(X_train_raw)
    X_train = scale_windows(X_train_raw, scaler)
    X_val = scale_windows(X_val_raw, scaler)

    val_status = val_segs.set_index("segment_id").loc[val_seg_ids, "status"].values
    X_val_normal = X_val[val_status == "Normal"]

    model, device, train_loss, val_loss = train_autoencoder(
        X_train, X_val_normal if len(X_val_normal) else X_train,
        n_features=len(config.CLASSIFY_SIGNAL_COLS),
        epochs=config.AE_EPOCHS, batch_size=config.AE_BATCH_SIZE,
        lr=config.AE_LEARNING_RATE, patience=config.AE_EARLY_STOP_PATIENCE,
        verbose=verbose,
    )
    if verbose:
        print(f"trained on device={device}: train_loss={train_loss:.5f}, val_loss={val_loss:.5f}")

    val_recon_errors = score_windows(model, device, X_val)
    normal_val_recon_errors = val_recon_errors[val_status == "Normal"]
    threshold = float(np.percentile(normal_val_recon_errors, config.THRESHOLD_PERCENTILE))

    val_pred = np.where(val_recon_errors > threshold, POSITIVE_LABEL, "Normal")
    precision, recall, f1, _ = precision_recall_fscore_support(
        val_status, val_pred, labels=[POSITIVE_LABEL], average="binary", pos_label=POSITIVE_LABEL,
    )
    cm = confusion_matrix(val_status, val_pred, labels=["Normal", POSITIVE_LABEL])

    result = {
        "segmentation_eval": seg_eval,
        "threshold": threshold,
        "threshold_percentile": config.THRESHOLD_PERCENTILE,
        "n_val_normal": int((val_status == "Normal").sum()),
        "n_val_abnormal": int((val_status == POSITIVE_LABEL).sum()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),  # rows/cols: [Normal, Abnormal resistance]
        "train_loss": train_loss,
        "val_loss": val_loss,
    }

    if save_artifacts:
        os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(config.ARTIFACTS_DIR, "door_autoencoder.pt"))
        joblib.dump(scaler, os.path.join(config.ARTIFACTS_DIR, "door_scaler.joblib"))
        joblib.dump(
            {"threshold": threshold, "signal_cols": config.CLASSIFY_SIGNAL_COLS,
             "window_length": config.SEGMENT_WINDOW_LENGTH},
            os.path.join(config.ARTIFACTS_DIR, "door_meta.joblib"),
        )

    return result, model, scaler, threshold


if __name__ == "__main__":
    result, *_ = run()
    print()
    print("=== Segmentation (Step 2) ===")
    print(result["segmentation_eval"])
    print()
    print("=== Classification validation (Step 4) ===")
    print(f"threshold (p{result['threshold_percentile']} of held-out Normal recon error): {result['threshold']:.5f}")
    print(f"held-out val set: {result['n_val_normal']} Normal + {result['n_val_abnormal']} Abnormal resistance")
    print(f"precision={result['precision']:.3f}  recall={result['recall']:.3f}  f1={result['f1']:.3f}")
    print(f"confusion matrix [rows=true, cols=pred], order [Normal, Abnormal resistance]:")
    print(result["confusion_matrix"])
