"""
Step 3 (part 1): turns each detected segment into a fixed-length window so
the existing SequenceAutoencoder architecture (built for fixed-length
windows) can be reused unchanged.
"""

import numpy as np

import config


def extract_segment_windows(raw_df, segments_df, signal_cols=config.CLASSIFY_SIGNAL_COLS,
                             target_length=config.SEGMENT_WINDOW_LENGTH):
    """
    Resamples each segment's signal_cols to a fixed target_length via
    linear interpolation over the segment's own row sequence -- segments
    are 137-190 raw rows long (verified), so this compresses (or mildly
    upsamples) every segment onto one common length while preserving the
    shape of each signal's trajectory across the cycle. One segment = one
    window here (unlike the synthetic pipeline's many overlapping windows
    per long per-unit stream), since the segment itself is the unit of
    classification for this task.

    Returns (X, seg_ids): X is (n_segments, target_length, n_signals) float32.
    """
    X_list, seg_ids = [], []
    for _, seg in segments_df.iterrows():
        sub = raw_df.loc[seg["start_idx"]:seg["end_idx"], signal_cols].to_numpy(dtype=np.float64)
        n = len(sub)
        old_x = np.linspace(0.0, 1.0, n)
        new_x = np.linspace(0.0, 1.0, target_length)
        resampled = np.empty((target_length, sub.shape[1]), dtype=np.float32)
        for c in range(sub.shape[1]):
            resampled[:, c] = np.interp(new_x, old_x, sub[:, c])
        X_list.append(resampled)
        seg_ids.append(seg["segment_id"])

    if not X_list:
        return np.empty((0, target_length, len(signal_cols)), dtype=np.float32), np.array([], dtype=object)
    return np.stack(X_list), np.array(seg_ids)


def attach_true_labels(detected_segments, answer_df):
    """
    Joins Train_Segments_Answer.csv's status onto the gap-detected segments
    by exact start_time match (verified 110/110 exact: same start_time,
    end_time, n_rows as the answer file -- see run_all.py's segmentation
    report) -- an inner merge here would silently drop rows if that ever
    stopped being exact, which is the point: it should raise, not mask, a
    future mismatch.
    """
    merged = detected_segments.merge(
        answer_df[["start_dt", "status", "operation", "segment_id"]].rename(
            columns={"segment_id": "true_segment_id"}
        ),
        left_on="start_time", right_on="start_dt", how="left",
    )
    n_missing = merged["status"].isna().sum()
    if n_missing:
        raise ValueError(
            f"{n_missing}/{len(merged)} detected segments failed to join a label by exact "
            "start_time match -- segmentation no longer reconstructs the answer file exactly, "
            "investigate before trusting downstream results."
        )
    return merged
