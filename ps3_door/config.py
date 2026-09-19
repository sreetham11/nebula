"""
Paths and constants for the PS3 Door subsystem pipeline (real NEBULA X
competition data, not the synthetic dataset in pipeline/). Kept as its own
project since the real Door schema (Motor current(mA), DCSR/DCSL/DLSR/DLSL,
Door leaf position, ...) shares no column names with schema_config.py's
synthetic assumption.
"""

import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(_THIS_DIR, ".."))

PS3_DOOR_DATA_DIR = os.path.normpath(os.path.join(
    REPO_ROOT, "..", "NebulaX-Hackathon-ProblemStatement", "PS3", "02_Datasets", "Door"
))
TRAIN_PATH = os.path.join(PS3_DOOR_DATA_DIR, "Train.csv")
TEST_PATH = os.path.join(PS3_DOOR_DATA_DIR, "Test.csv")
SEGMENTS_ANSWER_PATH = os.path.join(PS3_DOOR_DATA_DIR, "Train_Segments_Answer.csv")

ARTIFACTS_DIR = os.path.join(_THIS_DIR, "artifacts")
PREDICTIONS_PATH = os.path.join(_THIS_DIR, "door_predictions.csv")

RANDOM_SEED = 42

# --- Segmentation -----------------------------------------------------------
# The controller only logs rows while a door is actively opening or closing
# (verified: every row has exactly one of "Door is opening"/"Door is
# closing" == 1, never both, never neither) at a steady ~20ms cadence within
# a cycle. Consecutive cycles are separated by a genuine time gap in the
# recording (the controller simply doesn't log while idle) -- any gap well
# above the 20ms cadence is unambiguously a cycle boundary. Verified this
# reconstructs all 110/110 Train_Segments_Answer.csv segments exactly
# (matching start_time, end_time, AND n_rows) before relying on it for Test.
GAP_THRESHOLD_SEC = 0.1

# --- Classification -----------------------------------------------------
# Per the task brief: the signals believed most informative for motor
# resistance (current draw + the door-lock/close switches + leaf position,
# which together trace the physical motion profile of the cycle), excluding
# voltage/back-EMF/derived timers to keep the model's input focused.
CLASSIFY_SIGNAL_COLS = [
    "Motor current(mA)", "DCSR", "DCSL", "DLSR", "DLSL", "Door leaf position",
]

# Segments range 137-190 raw rows (verified); resampling every segment to a
# fixed length lets the exact same SequenceAutoencoder architecture used for
# the synthetic subsystems apply here unchanged, treating one resampled
# segment as one "window" (unlike the synthetic pipeline, which slices many
# overlapping windows out of one long per-unit stream -- here the segment
# itself IS the unit of classification, so there is exactly one window per
# segment).
SEGMENT_WINDOW_LENGTH = 64

# Fraction of NORMAL segments held out for validation/threshold calibration;
# ALL Abnormal segments are held out too (the autoencoder is healthy-only
# trained, so it never sees an Abnormal segment during training regardless).
VAL_FRACTION = 0.2

AE_HIDDEN_SIZE = 32
AE_LATENT_SIZE = 12
AE_NUM_LAYERS = 1
AE_DROPOUT = 0.1
AE_BATCH_SIZE = 16
AE_EPOCHS = 150
AE_LEARNING_RATE = 1e-3
AE_EARLY_STOP_PATIENCE = 15

# Single binary threshold (Normal vs Abnormal resistance), calibrated as a
# percentile of the held-out NORMAL validation segments' reconstruction
# error -- same percentile-of-healthy-only-validation method used by
# pipeline/fusion.py's compute_recon_error_thresholds, just one cut point
# instead of three since this task is binary, not four-level risk.
THRESHOLD_PERCENTILE = 95.0

# Boundary-match tolerance for scoring segmentation against known answers
# during development (Step 2) -- a detected boundary within this many
# seconds of a true boundary counts as "found". Purely a reporting
# convenience; the gap rule itself reconstructs boundaries exactly (see
# above), this just re-states that in IoU/tolerance terms for the report.
SEGMENTATION_TOLERANCE_SEC = 0.5
