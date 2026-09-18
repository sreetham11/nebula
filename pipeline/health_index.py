"""
Health Index: a bounded 0-100 restatement of a unit's reconstruction error.

WHY THIS EXISTS
---------------
Reconstruction error is the number the model actually produces, and it stays
the number of record -- but on its own it is unreadable. "62.5" means
nothing without knowing that healthy units on this subsystem sit around
0.6 and the Critical cut point is around 4. Worse, the scale is different
per subsystem, because each autoencoder is trained and calibrated
separately, so a door's 8.0 and a bogie's 8.0 are not comparable and must
never be put in the same table column as if they were.

The Health Index fixes exactly that and nothing more. It is a monotone
re-expression of the same reconstruction error against that subsystem's own
calibrated thresholds, so:

  * 100 means "reconstructs as well as the healthy fleet median"
  * the risk band boundaries land on the SAME index values for every
    subsystem, so 58 on a door and 58 on a car mean the same severity
  * it is bounded, so it can drive a gauge, a bar, or a color ramp
  * it is monotone in reconstruction error, so ranking by Health Index and
    ranking by reconstruction error (within a subsystem) never disagree

It is a presentation layer. It is NOT a second model, it adds no
information, and it never changes a risk level -- risk still comes from
fusion.py. Anywhere the index is shown, the raw reconstruction error and
the threshold it is being measured against should be shown alongside it, so
the number stays traceable back to the model output.

ANCHORS
-------
Piecewise-linear in log(reconstruction error), pinned at:

    error <= healthy median          -> 100   (indistinguishable from healthy)
    error == Caution threshold       ->  75
    error == Warning threshold       ->  60
    error == Critical threshold      ->  40
    error >= Critical x SEVERE_MULT  ->   0   (far past any calibrated point)

Interpolating in log space rather than linear space matters: reconstruction
error is a squared quantity with a long right tail, so a linear
interpolation would leave almost the entire 40-100 range compressed into a
sliver near zero and make every flagged unit look identically dead.
"""

import math

# Health Index value pinned at each calibrated threshold. Fixed constants,
# deliberately not configurable per subsystem -- their whole purpose is to
# be the same across subsystems so the numbers are comparable.
HEALTHY_ANCHOR = 100.0
CAUTION_ANCHOR = 75.0
WARNING_ANCHOR = 60.0
CRITICAL_ANCHOR = 40.0
FLOOR_ANCHOR = 0.0

# How far past the Critical threshold counts as "0 health". Beyond the
# calibrated range there is no principled scale left, so this is an honest
# arbitrary cutoff rather than a derived one: 10x Critical.
SEVERE_MULTIPLIER = 10.0

# Health Index bands, for labelling the number in a UI. These mirror the
# anchors above, so a unit's band here always agrees with its fused risk
# level unless the baseline z-score escalation moved it (fusion.py can bump
# a level on a single-reading spike without the window error moving much --
# that disagreement is real and worth showing, not worth hiding).
BANDS = [
    (CAUTION_ANCHOR, "Healthy"),
    (WARNING_ANCHOR, "Watch"),
    (CRITICAL_ANCHOR, "Degraded"),
    (FLOOR_ANCHOR, "Critical"),
]


def _interp_log(x, x0, x1, y0, y1):
    """Linear interpolation of y against log(x), guarded for non-positive x."""
    if x <= 0 or x0 <= 0 or x1 <= 0 or x1 == x0:
        return y0
    frac = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
    frac = min(max(frac, 0.0), 1.0)
    return y0 + frac * (y1 - y0)


def health_index(recon_error, thresholds, healthy_median):
    """
    Map one reconstruction error to a 0-100 Health Index.

    thresholds:     the subsystem's calibrated {"Caution","Warning","Critical"}
                    cut points, as written to models/<subsystem>_meta.json
    healthy_median: median reconstruction error of the held-out healthy
                    validation units for that subsystem

    Returns None for a missing error (the first window_length-1 readings of
    a unit's history have no score yet), so callers render a dash rather
    than a misleading 100.
    """
    if recon_error is None:
        return None
    try:
        e = float(recon_error)
    except (TypeError, ValueError):
        return None
    if e != e:  # NaN
        return None

    caution = float(thresholds["Caution"])
    warning = float(thresholds["Warning"])
    critical = float(thresholds["Critical"])
    median = float(healthy_median)

    # Guard against a degenerate calibration (thresholds collapsing onto
    # each other on a tiny validation set) by forcing a strictly increasing
    # ladder; without this the interpolation below could divide by zero.
    median = max(median, 1e-12)
    caution = max(caution, median * 1.000001)
    warning = max(warning, caution * 1.000001)
    critical = max(critical, warning * 1.000001)
    severe = critical * SEVERE_MULTIPLIER

    if e <= median:
        return HEALTHY_ANCHOR
    if e <= caution:
        value = _interp_log(e, median, caution, HEALTHY_ANCHOR, CAUTION_ANCHOR)
    elif e <= warning:
        value = _interp_log(e, caution, warning, CAUTION_ANCHOR, WARNING_ANCHOR)
    elif e <= critical:
        value = _interp_log(e, warning, critical, WARNING_ANCHOR, CRITICAL_ANCHOR)
    elif e < severe:
        value = _interp_log(e, critical, severe, CRITICAL_ANCHOR, FLOOR_ANCHOR)
    else:
        value = FLOOR_ANCHOR

    return round(min(max(value, 0.0), 100.0), 1)


def health_band(index_value):
    """Short label for a Health Index value ("Healthy" / "Watch" / ...)."""
    if index_value is None:
        return None
    for cutoff, label in BANDS:
        if index_value >= cutoff:
            return label
    return BANDS[-1][1]


def error_vs_healthy(recon_error, healthy_median):
    """
    Reconstruction error expressed as a multiple of the healthy median --
    the plain-language companion figure to the index ("x95 the healthy
    fleet median"), so the raw number keeps a scale attached to it.
    """
    if recon_error is None:
        return None
    try:
        e = float(recon_error)
    except (TypeError, ValueError):
        return None
    if e != e:
        return None
    median = float(healthy_median)
    if median <= 0:
        return None
    return round(e / median, 1)
