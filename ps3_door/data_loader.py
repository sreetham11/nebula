"""
Loading + the custom Datetime parser for PS3 Door's non-standard timestamp
format: Year-Month-Date-Hour-Minute-Second-Millisecond, hyphen-separated,
NOT zero-padded (e.g. "2023-7-5-0-0-3-760") -- pandas' %f directive assumes
a zero-padded fraction, so this can't be parsed with a plain strptime
format string; each field is split and reassembled instead.
"""

import pandas as pd

import config


def parse_datetime_column(series):
    """Vectorized parse of the Y-M-D-H-Mi-S-ms format into pandas Timestamps."""
    parts = series.str.split("-", expand=True).astype(int)
    parts.columns = ["y", "mo", "d", "h", "mi", "s", "ms"]
    base = pd.to_datetime(dict(
        year=parts["y"], month=parts["mo"], day=parts["d"],
        hour=parts["h"], minute=parts["mi"], second=parts["s"],
    ))
    return base + pd.to_timedelta(parts["ms"], unit="ms")


def format_datetime(ts):
    """Inverse of parse_datetime_column for a single timestamp -- renders
    back into the dataset's own Y-M-D-H-Mi-S-ms format (not zero-padded),
    for writing predictions in the same style the raw data uses (the Door
    info kit also accepts a standard ISO timestamp, but matching the
    source format keeps predictions visually diffable against the input)."""
    ms = ts.microsecond // 1000
    return f"{ts.year}-{ts.month}-{ts.day}-{ts.hour}-{ts.minute}-{ts.second}-{ms}"


def load_raw(path):
    """Loads a raw Door stream (Train.csv or Test.csv), parses Datetime,
    and sorts by time (should already be sorted, but this is what
    segmentation's gap-detection assumes)."""
    df = pd.read_csv(path)
    df["dt"] = parse_datetime_column(df["Datetime"])
    df = df.sort_values("dt").reset_index(drop=True)
    return df


def load_segments_answer(path=config.SEGMENTS_ANSWER_PATH):
    ans = pd.read_csv(path)
    ans["start_dt"] = parse_datetime_column(ans["start_time"])
    ans["end_dt"] = parse_datetime_column(ans["end_time"])
    return ans


def join_labels_onto_rows(raw_df, segments_df):
    """
    Attaches segment_id/status/operation back onto each raw row by interval
    membership (a row belongs to segment i if start_dt[i] <= row.dt <=
    end_dt[i]) -- every row belongs to exactly one segment for this dataset
    (see config.GAP_THRESHOLD_SEC's docstring), so this is an exact
    assignment via merge_asof + boundary check, not a fuzzy nearest-match.
    """
    raw_df = raw_df.sort_values("dt")
    segs = segments_df.sort_values("start_dt")
    merged = pd.merge_asof(raw_df, segs[["segment_id", "start_dt", "end_dt", "operation", "status"]],
                            left_on="dt", right_on="start_dt", direction="backward")
    within = merged["dt"] <= merged["end_dt"]
    merged.loc[~within, ["segment_id", "operation", "status"]] = None
    return merged
