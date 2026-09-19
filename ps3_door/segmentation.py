"""
Step 2: rule-based (not ML) segmentation -- find door-cycle boundaries in a
continuous stream.

Rule: the controller only logs rows while a door is actively moving --
every row has exactly one of "Door is opening"/"Door is closing" == 1
(verified on both Train.csv and Test.csv: 0 rows with both flags 0, 0 rows
with both flags 1), sampled at a steady ~20ms cadence within a cycle, with a
genuine time gap between cycles (the controller doesn't log while idle).
So: sort by time, and any gap between consecutive rows well above the
normal ~20ms cadence marks a new segment. This is not an approximation --
grouping Train.csv this way reconstructs all 110 Train_Segments_Answer.csv
segments' start_time, end_time, AND n_rows exactly (see validate_against_answer
below / the run_all.py report).
"""

import pandas as pd

import config


def detect_segments(df, gap_threshold_sec=config.GAP_THRESHOLD_SEC):
    """
    df must be sorted by "dt" (data_loader.load_raw already sorts).
    Returns a DataFrame: segment_id (1-indexed, positional), start_time,
    end_time, start_idx, end_idx (row-label bounds into df), n_rows.
    """
    deltas = df["dt"].diff().dt.total_seconds()
    new_segment = deltas.isna() | (deltas > gap_threshold_sec)
    seg_num = new_segment.cumsum()

    grouped = df.groupby(seg_num)
    segments = grouped.agg(
        start_time=("dt", "first"),
        end_time=("dt", "last"),
        n_rows=("dt", "size"),
    ).reset_index(drop=True)
    # index bounds (positional) for slicing df per segment downstream
    starts = grouped.apply(lambda g: g.index[0], include_groups=False)
    ends = grouped.apply(lambda g: g.index[-1], include_groups=False)
    segments["start_idx"] = starts.values
    segments["end_idx"] = ends.values
    segments.insert(0, "segment_id", [f"seg_{i+1:04d}" for i in range(len(segments))])
    return segments


def evaluate_segmentation(detected, true_segments, tolerance_sec=config.SEGMENTATION_TOLERANCE_SEC):
    """
    For Step 2's report: what fraction of TRUE segment boundaries does the
    detector find, within `tolerance_sec` of the true start_time AND
    end_time. Reports both the tolerance-window recall and the count of
    EXACT (down-to-the-row) matches, since the gap rule is expected to be
    exact for this dataset -- exact-match count confirms that rather than
    just a loose approximation happening to score well.
    """
    n_true = len(true_segments)
    exact_matches = 0
    tolerant_matches = 0

    true_starts = true_segments["start_dt"].values
    true_ends = true_segments["end_dt"].values
    det_starts = detected["start_time"].values
    det_ends = detected["end_time"].values

    used_true = set()
    for d_start, d_end in zip(det_starts, det_ends):
        for i, (t_start, t_end) in enumerate(zip(true_starts, true_ends)):
            if i in used_true:
                continue
            start_diff = abs((pd.Timestamp(d_start) - pd.Timestamp(t_start)).total_seconds())
            end_diff = abs((pd.Timestamp(d_end) - pd.Timestamp(t_end)).total_seconds())
            if start_diff == 0 and end_diff == 0:
                exact_matches += 1
                tolerant_matches += 1
                used_true.add(i)
                break
            if start_diff <= tolerance_sec and end_diff <= tolerance_sec:
                tolerant_matches += 1
                used_true.add(i)
                break

    return {
        "n_true_segments": n_true,
        "n_detected_segments": len(detected),
        "exact_matches": exact_matches,
        "tolerant_matches": tolerant_matches,
        "exact_match_fraction": exact_matches / n_true if n_true else None,
        "tolerant_match_fraction": tolerant_matches / n_true if n_true else None,
    }
