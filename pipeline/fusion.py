"""
Fusion logic: combines the baseline rolling z-score flag with the
autoencoder's reconstruction error into one four-level risk label.

============================== COMBINATION LOGIC ==============================
1. BASE LEVEL from the autoencoder reconstruction error, converted to a
   percentile against the healthy-unit training distribution (so "how
   unusual is this window compared to what normal fleet operation looks
   like" rather than a raw, unit-less MSE number):

       recon_error_percentile < CAUTION_PCT    -> Normal
       CAUTION_PCT  <= percentile < WARNING_PCT  -> Caution
       WARNING_PCT  <= percentile < CRITICAL_PCT -> Warning
       percentile >= CRITICAL_PCT                -> Critical

   The autoencoder is the "slow drift" detector: it sees the whole
   multivariate window, so it's the one that should catch gradual,
   noisy, plateau-and-ramp degradation before any single reading looks
   extreme.

2. ESCALATION from the baseline rolling z-score flag: if the CURRENT
   reading is itself a >3-sigma outlier against that unit's own recent
   rolling mean/std (baseline_flag == True), the combined risk level is
   bumped up by one step (Normal->Caution->Warning->Critical, capped at
   Critical). The baseline flag is the "sudden spike" detector -- fast,
   explainable, good at catching sharp one-off excursions that a
   window-averaged reconstruction error might still be diluting.

Net effect: a unit can reach Warning/Critical two ways --
  (a) sustained drift that slowly pushes reconstruction error up through
      the percentile bands (this is the primary intended failure mode
      this pipeline is built to catch), or
  (b) a sudden single-reading spike that baseline catches immediately,
      escalating whatever the AE currently says.
This mirrors how a demo would explain a specific alert: "the model saw
this window drifting from healthy behavior" and/or "this exact reading
is also way outside this unit's own normal range."
================================================================================
"""

import numpy as np
import pandas as pd

from model_config import (
    RECON_ERROR_CAUTION_PERCENTILE,
    RECON_ERROR_WARNING_PERCENTILE,
    RECON_ERROR_CRITICAL_PERCENTILE,
    RISK_LEVELS,
)

NORMAL, CAUTION, WARNING, CRITICAL = RISK_LEVELS


def compute_recon_error_thresholds(healthy_recon_errors):
    """Percentile cut points computed from healthy-unit reconstruction errors."""
    return {
        CAUTION: float(np.percentile(healthy_recon_errors, RECON_ERROR_CAUTION_PERCENTILE)),
        WARNING: float(np.percentile(healthy_recon_errors, RECON_ERROR_WARNING_PERCENTILE)),
        CRITICAL: float(np.percentile(healthy_recon_errors, RECON_ERROR_CRITICAL_PERCENTILE)),
    }


def _base_level_from_recon_error(recon_error, thresholds):
    if pd.isna(recon_error):
        return None
    if recon_error >= thresholds[CRITICAL]:
        return CRITICAL
    if recon_error >= thresholds[WARNING]:
        return WARNING
    if recon_error >= thresholds[CAUTION]:
        return CAUTION
    return NORMAL


def _escalate(level):
    idx = RISK_LEVELS.index(level)
    return RISK_LEVELS[min(idx + 1, len(RISK_LEVELS) - 1)]


def fuse_risk(df, recon_error_col, baseline_flag_col, thresholds):
    """
    Adds:
      risk_level      : one of RISK_LEVELS, or None where recon_error is
                         unavailable (first AE_WINDOW_LENGTH-1 readings of
                         a unit's history)
      risk_level_idx   : 0..3 ordinal (NaN where risk_level is None)
    """
    df = df.copy()

    base = df[recon_error_col].apply(lambda e: _base_level_from_recon_error(e, thresholds))
    flagged = df[baseline_flag_col].fillna(False)

    def combine(base_level, is_flagged):
        if base_level is None:
            return None
        return _escalate(base_level) if is_flagged else base_level

    df["risk_level"] = [combine(b, f) for b, f in zip(base, flagged)]
    level_to_idx = {lvl: i for i, lvl in enumerate(RISK_LEVELS)}
    df["risk_level_idx"] = df["risk_level"].map(level_to_idx)
    return df
