"""
Validation: for every unit that has a ground-truth fault in fault_log, plot
its raw signal(s) + autoencoder reconstruction error + fused risk level
over time, and compute lead time -- how long before the logged fault
timestamp the fused risk first crossed into Warning/Critical.

This is treated as the most important deliverable of the pipeline: it's
the artifact that answers "does this actually catch failures early, and by
how much." Plots are saved to validation_outputs/, one PNG per faulty unit,
plus a lead_time_summary.csv/md across all of them.
"""

# On Windows, torch must be imported before pandas/numpy: both ship their own
# Intel OpenMP runtime (libiomp5md.dll), and if pandas' loads first, torch's
# c10.dll fails to initialise with "WinError 1114". Importing torch first is
# the documented ordering workaround; it is a no-op on Linux/macOS.
import torch  # noqa: F401,E402  -- must precede pandas/numpy imports


import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import schema_config as sc
from fusion import RISK_LEVELS, WARNING

RISK_COLORS = {
    "Normal": "#2ecc71",
    "Caution": "#f1c40f",
    "Warning": "#e67e22",
    "Critical": "#e74c3c",
}


def _load_thresholds(subsystem_type):
    path = os.path.join(sc.MODELS_DIR, f"{subsystem_type}_meta.json")
    with open(path) as f:
        meta = json.load(f)
    return meta["thresholds"]


def _first_warning_crossing(unit_df, fault_ts, ts_col):
    """Earliest timestamp at/after which risk_level is Warning or Critical
    AND stays at or above Warning from that point through the fault (to
    avoid crediting a single noisy blip as 'detection'). Falls back to the
    first isolated crossing if no sustained one exists before the fault."""
    warn_idx = RISK_LEVELS.index(WARNING)
    before_fault = unit_df[unit_df[ts_col] <= fault_ts].reset_index(drop=True)
    at_or_above = before_fault["risk_level_idx"] >= warn_idx

    # sustained: once crossed, fraction of remaining pre-fault readings
    # that stay >= Warning must be high
    for i in range(len(before_fault)):
        if at_or_above.iloc[i]:
            remainder = at_or_above.iloc[i:]
            if remainder.mean() >= 0.7:
                return before_fault.loc[i, ts_col]

    if at_or_above.any():
        return before_fault.loc[at_or_above.idxmax(), ts_col]
    return None


def _plot_unit(unit_df, subsystem_type, subsystem_id, fault_row, thresholds, out_path):
    cfg = sc.SUBSYSTEMS[subsystem_type]
    ts_col = cfg["timestamp_col"]
    signal_cols = cfg["signal_cols"]
    fault_ts = fault_row["timestamp"]

    first_warning_ts = _first_warning_crossing(unit_df, fault_ts, ts_col)
    if first_warning_ts is not None:
        lead_time_hours = (fault_ts - first_warning_ts).total_seconds() / 3600.0
    else:
        lead_time_hours = None

    plt.style.use("dark_background")

    # One row per modelled signal, then the reconstruction error, then the
    # risk strip. A row each rather than twinned y-axes: subsystems now
    # carry four to six channels, and the informative pattern is often two
    # of them moving in OPPOSITE directions (current up while rpm falls),
    # which is unreadable when several are squeezed onto shared axes.
    n_sig = len(signal_cols)
    height_ratios = [0.75] * n_sig + [1.0, 0.45]
    fig, axes = plt.subplots(
        n_sig + 2, 1, figsize=(13, 3.2 + 1.5 * n_sig), sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )

    # --- Panels 1..n: raw signals, one per row ---
    signal_palette = ["#3498db", "#e84393", "#f9ca24", "#6ab04c", "#e17055", "#a29bfe"]
    for i, sig in enumerate(signal_cols):
        ax = axes[i]
        color = signal_palette[i % len(signal_palette)]
        ax.plot(unit_df[ts_col], unit_df[sig], color=color, linewidth=0.8, label=sig)
        meta = sc.signal_meta(sig)
        ylabel = f"{meta['label']}\n({meta['unit']})" if meta["unit"] else meta["label"]
        ax.set_ylabel(ylabel, color=color, fontsize=8)
        ax.tick_params(axis="y", colors=color, labelsize=7)

    axes[0].set_title(f"{subsystem_id}  ({subsystem_type})  --  fault: {fault_row['fault_type']}, "
                       f"severity {fault_row['severity']}")

    # --- Reconstruction error + threshold lines ---
    ax2 = axes[n_sig]
    ax2.plot(unit_df[ts_col], unit_df["ae_recon_error"], color="#00cec9", linewidth=0.9,
              label="AE reconstruction error")
    for level, y in thresholds.items():
        ax2.axhline(y, color=RISK_COLORS[level], linestyle="--", linewidth=0.8, alpha=0.7,
                    label=f"{level} threshold")
    ax2.set_ylabel("recon error")
    ax2.legend(loc="upper left", fontsize=7, ncol=2)

    # --- Risk level over time as a colored strip ---
    ax3 = axes[n_sig + 1]
    for lvl in RISK_LEVELS:
        mask = unit_df["risk_level"] == lvl
        ax3.scatter(unit_df.loc[mask, ts_col], [1] * mask.sum(), color=RISK_COLORS[lvl],
                    s=6, marker="s", label=lvl)
    ax3.set_yticks([])
    ax3.legend(loc="upper left", fontsize=7, ncol=4)
    ax3.set_ylim(0.5, 1.5)

    for ax in axes:
        ax.axvline(fault_ts, color="white", linestyle="-", linewidth=1.4, alpha=0.9)
        if first_warning_ts is not None:
            ax.axvline(first_warning_ts, color="#00b894", linestyle=":", linewidth=1.4, alpha=0.9)

    axes[0].annotate("FAULT LOGGED", xy=(fault_ts, axes[0].get_ylim()[1]), xytext=(5, -12),
                      textcoords="offset points", color="white", fontsize=8, rotation=90, va="top")
    if first_warning_ts is not None:
        axes[0].annotate("DETECTED", xy=(first_warning_ts, axes[0].get_ylim()[1]), xytext=(5, -12),
                          textcoords="offset points", color="#00b894", fontsize=8, rotation=90, va="top")

    title_suffix = (f"lead time: {lead_time_hours:.1f} h ({lead_time_hours / 24:.1f} d)"
                     if lead_time_hours is not None else "NOT detected before fault")
    fig.suptitle(f"{sc.SUBSYSTEMS[subsystem_type]['id_col']}={subsystem_id}  |  {title_suffix}",
                 fontsize=11, color="#dfe6e9")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)

    return lead_time_hours, first_warning_ts


def run_validation(scored_by_type, fault_log):
    """
    scored_by_type: {subsystem_type: scored DataFrame}, one entry per
    subsystem in schema_config.SUBSYSTEMS. Taking a dict rather than one
    positional argument per subsystem is what lets a new subsystem (the
    car auxiliary systems, say) be added in schema_config alone.
    """
    os.makedirs(sc.VALIDATION_OUTPUT_DIR, exist_ok=True)
    dfs = scored_by_type
    thresholds_cache = {st: _load_thresholds(st) for st in sc.SUBSYSTEM_TYPES}

    summary_rows = []
    for _, fault_row in fault_log.iterrows():
        subsystem_type = fault_row["subsystem_type"]
        subsystem_id = fault_row["subsystem_id"]
        cfg = sc.SUBSYSTEMS[subsystem_type]
        df = dfs[subsystem_type]
        unit_df = df[df[cfg["id_col"]] == subsystem_id].sort_values(cfg["timestamp_col"]).reset_index(drop=True)

        out_path = os.path.join(sc.VALIDATION_OUTPUT_DIR, f"{subsystem_id}.png")
        lead_time_hours, first_warning_ts = _plot_unit(
            unit_df, subsystem_type, subsystem_id, fault_row, thresholds_cache[subsystem_type], out_path
        )

        summary_rows.append({
            "unit_id": subsystem_id,
            "subsystem_type": subsystem_type,
            "fault_type": fault_row["fault_type"],
            "severity": fault_row["severity"],
            "fault_timestamp": fault_row["timestamp"],
            "first_warning_timestamp": first_warning_ts,
            "lead_time_hours": round(lead_time_hours, 2) if lead_time_hours is not None else None,
            "lead_time_days": round(lead_time_hours / 24, 2) if lead_time_hours is not None else None,
            "detected_before_fault": lead_time_hours is not None,
        })
        status = f"lead time {lead_time_hours:.1f}h" if lead_time_hours is not None else "NOT DETECTED"
        print(f"  {subsystem_id:12s} ({subsystem_type:5s}) -> {status}")

    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(sc.VALIDATION_OUTPUT_DIR, "lead_time_summary.csv")
    summary.to_csv(summary_path, index=False)

    md_path = os.path.join(sc.VALIDATION_OUTPUT_DIR, "lead_time_summary.md")
    with open(md_path, "w") as f:
        f.write("# Validation: detection lead time per fault\n\n")
        f.write("Synthetic data pending real dataset. Lead time = fault_log timestamp minus "
                "the first sustained Warning/Critical crossing.\n\n")
        f.write(summary.to_markdown(index=False))
        f.write("\n\n")
        detected = summary["detected_before_fault"].sum()
        f.write(f"**Detected before fault: {detected}/{len(summary)}**\n")
        if detected:
            f.write(f"\nMean lead time (detected only): "
                    f"{summary.loc[summary['detected_before_fault'], 'lead_time_hours'].mean():.1f} hours\n")

    print(f"\nSaved plots + summary to {sc.VALIDATION_OUTPUT_DIR}")
    return summary


if __name__ == "__main__":
    import data_loader
    fault_log = data_loader.load_fault_log()
    scored_by_type = {
        st: pd.read_csv(sc.SUBSYSTEMS[st]["scored_path"],
                        parse_dates=[sc.SUBSYSTEMS[st]["timestamp_col"]])
        for st in sc.SUBSYSTEM_TYPES
    }
    run_validation(scored_by_type, fault_log)
