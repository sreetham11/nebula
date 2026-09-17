"""
Hyperparameters and thresholds for the detection pipeline. Kept separate
from schema_config.py (column names) so tuning the model doesn't risk
touching the data-mapping layer.
"""

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Baseline rolling model
# ---------------------------------------------------------------------------
# Rolling window expressed in number of readings (not fixed time) since
# each unit's sampling interval is jittered. At ~10 min/reading this is
# roughly a 1-day and a 3-day window.
BASELINE_SHORT_WINDOW = 144   # ~1 day
BASELINE_MIN_PERIODS = 30     # readings needed before a window is trusted
BASELINE_Z_THRESHOLD = 3.0    # flag if |z| exceeds this many rolling std devs

# ---------------------------------------------------------------------------
# Autoencoder (LSTM/GRU) -- trained healthy-only
# ---------------------------------------------------------------------------
AE_ARCHITECTURE = "gru"       # "gru" or "lstm"
AE_WINDOW_LENGTH = 48         # readings per window (~8 hours at 10 min cadence)
AE_WINDOW_STRIDE = 4          # stride when slicing training windows
AE_HIDDEN_SIZE = 32
AE_LATENT_SIZE = 12
AE_NUM_LAYERS = 1
AE_DROPOUT = 0.1

AE_BATCH_SIZE = 128
AE_EPOCHS = 25
AE_LEARNING_RATE = 1e-3

# Healthy units are split by UNIT (not by row/window) into train vs.
# validation, so no window from a unit the model trains on can also show
# up in validation -- windows from the same unit overlap heavily in time
# and share that unit's specific noise profile, so a row-level split would
# leak and understate true held-out error. With 8 healthy units per
# subsystem this rounds up to 2 held-out validation units, 6 training
# units. The held-out validation units' reconstruction errors are also
# what fusion's anomaly thresholds are calibrated against (see
# run_pipeline.py) -- never the training units' own error, which would be
# optimistic (in-sample).
AE_VAL_UNIT_FRACTION = 0.25
AE_EARLY_STOP_PATIENCE = 5

# Inference stride: how far apart (in readings) scored windows are during
# scoring/validation. Smaller = smoother score curve, slower to compute.
AE_SCORING_STRIDE = 1

# ---------------------------------------------------------------------------
# Fusion -> risk level
# ---------------------------------------------------------------------------
# Reconstruction error is converted to a percentile against the healthy
# training distribution, then combined with the baseline z-flag. See
# pipeline/fusion.py for the documented combination logic.
RECON_ERROR_CAUTION_PERCENTILE = 90.0
RECON_ERROR_WARNING_PERCENTILE = 97.5
RECON_ERROR_CRITICAL_PERCENTILE = 99.5

RISK_LEVELS = ["Normal", "Caution", "Warning", "Critical"]
