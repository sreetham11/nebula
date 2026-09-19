"""
Step 5: runs the full segment + classify pipeline on Test.csv (the
held-out, single continuous, unlabeled stream) and writes
door_predictions.csv in the exact submission format: start_time, end_time,
prediction -- no file_id, one row per predicted segment.
"""

import os
import sys

import joblib
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline"))
from autoencoder import SequenceAutoencoder, score_windows, get_device  # noqa: E402

import config
import data_loader
import segmentation
import features
from train_classify import scale_windows, POSITIVE_LABEL


def load_artifacts():
    scaler = joblib.load(os.path.join(config.ARTIFACTS_DIR, "door_scaler.joblib"))
    meta = joblib.load(os.path.join(config.ARTIFACTS_DIR, "door_meta.joblib"))
    device = get_device()
    model = SequenceAutoencoder(n_features=len(meta["signal_cols"]))
    model.load_state_dict(torch.load(
        os.path.join(config.ARTIFACTS_DIR, "door_autoencoder.pt"), map_location=device,
    ))
    model.to(device)
    model.eval()
    return model, device, scaler, meta["threshold"]


def run(test_path=config.TEST_PATH, output_path=config.PREDICTIONS_PATH, verbose=True):
    model, device, scaler, threshold = load_artifacts()

    raw_test = data_loader.load_raw(test_path)
    detected = segmentation.detect_segments(raw_test)
    if verbose:
        print(f"Test.csv: {len(raw_test)} rows -> {len(detected)} detected segments")

    X_raw, seg_ids = features.extract_segment_windows(raw_test, detected)
    X = scale_windows(X_raw, scaler)
    recon_errors = score_windows(model, device, X)
    predictions = np.where(recon_errors > threshold, POSITIVE_LABEL, "Normal")

    out = detected.copy()
    out["prediction"] = predictions
    out["start_time"] = out["start_time"].apply(data_loader.format_datetime)
    out["end_time"] = out["end_time"].apply(data_loader.format_datetime)
    out = out[["start_time", "end_time", "prediction"]]
    out.to_csv(output_path, index=False)

    if verbose:
        print(f"wrote {len(out)} predicted segments -> {output_path}")
        print(out["prediction"].value_counts())

    return out


if __name__ == "__main__":
    run()
