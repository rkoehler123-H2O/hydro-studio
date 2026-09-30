"""
Hydro-Studio: Hydrological Visual Analytics & Graphics Engine
Based on: Operational Modular Data Evaluation and Prompt Library (Version v8.5)
© Visual Data Analytics, LLC (2026.1)
"""

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
import streamlit as st

# ---------------------------------------------------------
# Page Configuration & Global Typography
# ---------------------------------------------------------
st.set_page_config(
    page_title="Hydro-Studio",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.family": "sans-serif",
    "figure.dpi": 300,
    "axes.edgecolor": "#000000",
    "axes.linewidth": 1.0,
})

MONTH_LABELS = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]
MONTH_TICKS_366 = [1, 32, 62, 93, 124, 153, 184, 214, 245, 275, 306, 337]


# ---------------------------------------------------------
# Data Utilities & Calculations
# ---------------------------------------------------------
def prepare_pairs_and_dowy(df, date_col, value_col):
    """Sorts, checks continuity, calculates rate of change and standardized 366-day DOWY."""
    df_clean = df.dropna(subset=[date_col, value_col]).copy()
    df_clean[date_col] = pd.to_datetime(df_clean[date_col])
    df_clean = df_clean.sort_values(by=date_col).reset_index(drop=True)
    
    # Physical Flow Bounds: Discharge must be positive
    df_clean = df_clean[df_clean[value_col] > 0].copy()
    
    # Axiomatic 1-day temporal continuity: dQ/dt = Q(t+1) - Q(t)
    dt = df_clean[date_col].diff().dt.total_seconds() / 86400.0
    valid_pair = (dt == 1.0)
    
    df_clean["Q_t"] = df_clean[value_col].shift(0)
    df_clean["Q_next"] = df_clean[value_col].shift(-1)
    df_clean["valid_pair"] = valid_pair.shift(-1).fillna(False)
    
    # Water Year and 366-day calendar alignment (Feb 29 = Day 152)
    month = df_clean[date_col].dt.month
    day = df_clean[date_col].dt.day
    year = df_clean[date_col].dt.year
    df_clean["WaterYear"] = np.where(month >= 10, year + 1, year)
    
    ref_dates = pd.date_range("2019-10-01", "2020-09-30", freq="D")
    dowy_lookup = {(d.month, d.day): idx + 1 for idx, d in enumerate(ref_dates)}
    df_clean["Standard_DOWY"] = [dowy_lookup.get((m, d), np.nan) for m, d in zip(month, day)]
    
    return df_clean


def apply_standard_grid(ax):
    """Applies standard solid black major gridlines and dashed gray minor gridlines."""
    ax.grid(True, which="major", color="#000000", linestyle="-", lw=0.9, zorder=1)
    ax.grid(True, which="minor", color="#808080", linestyle="--", lw=0.5, zorder=1)


# ---------------------------------------------------------
# MOD-01: Enhanced Flow Duration Curve (eFDC)
# ---------------------------------------------------------
def run_mod01(df, date_col, value_col):
    df_pairs = prepare_pairs_and_dowy(df, date_col, value_col)
    pairs = df_pairs[df_pairs["valid_pair"]].copy()
    
    q_t = pairs["Q_t"].values
    q_next = pairs["Q_next"].values
    n = len(q_t)
    
    # Exceedance probability of antecedent state Q_t: P(Q_t) = [m / (n + 1)] * 100
    rank = np.argsort(np.argsort(-q_t)) + 1
    p_qt = (rank / (n + 1.0)) * 100.0
    
    dq = q_next - q_t
    rising = dq > 0
    falling = dq < 0
    steady = dq == 0
    
    # Baseline FDC curve
    sort_idx = np.argsort(-q_t)
    p_baseline = (np.arange(1, n + 1) / (n + 1.0)) * 100.0
    q_baseline = q_t[sort_idx]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    
    # Point cloud (P(Q_t), Q(t+1))
    ax.scatter(p_qt[rising], q_next[rising], color="#0000FF", alpha=0.3, s=12, label="Rising Limb (+dQ/dt > 0)", zorder=2)
    ax.scatter(p_qt[falling], q_next[falling], color="#FF0000", alpha=0.3, s=12, label="Falling Limb (-dQ/dt < 0)", zorder=2)
    ax.scatter(p_qt[steady], q_next[steady], color="#000000", alpha=0.3, s=12, label="Steady State (dQ/dt = 0)", zorder=2)
    
    # Baseline FDC
    ax.plot(p_baseline, q_baseline, color="#000000", lw=1.8, label="Baseline FDC ($Q_t$)", zorder=4)

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
    ax.set_xlabel("Exceedance Percentage (%)", fontsize=10, fontweight="bold")
    ax.set_ylabel(f"Discharge ({value_col})", fontsize=10, fontweight="bold")
    
    # Exterior Title and Legend spacing to prevent overlap
    ax.set_title("MOD-01: Enhanced Flow Duration Curve (eFDC)", fontsize=12, fontweight="bold", pad=40)
    apply_standard_grid(ax)

    ax.legend(bbox_to_anchor=(0.5, 1.02), loc="lower center", ncol=4, frameon=True, facecolor="#ffffff", edgecolor="#cccccc", fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------
# MOD-02: Lag-1 Differential Hydrograph Point Cloud
# ---------------------------------------------------------
def run_mod02(df, date_col, value_col):
    df_pairs = prepare_pairs_and_dowy(df, date_col, value_col)
    pairs = df_pairs[df_pairs["valid_pair"]].copy()
    
    q_t = pairs["Q_t"].values
    q_next = pairs["Q_next"].values
    dq = q_next - q_t
    
    rising = dq > 0
    falling = dq < 0
    steady = dq == 0
    
    min_v = 10 ** np.floor(np.log10(min(q_t.min(), q_next.min())))
    max_v = 10 ** np.ceil(np.log10(max(q_t.max(), q_next.max())))

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.scatter(q_t[rising], q_next[rising], color="#0000FF", alpha=0.3, s=14, label="Rising Limb (+dQ/dt > 0)", zorder=2)
    ax.scatter(q_t[falling], q_next[falling], color="#FF0000", alpha=0.3, s=14, label="Falling Limb (-dQ/dt < 0)", zorder=2)
    ax.scatter(q_t[steady], q_next[steady], color="#000000", alpha=0.3, s=14, label="Steady State (dQ/dt = 0)", zorder=2)

    # Reference Lines
    ref_x = np.array([min_v, max_v])
    ax.plot(ref_x, ref_x, color="#000000", lw=1.2, linestyle="-", label="1:1 Steady State ($y=x$)", zorder=3)
    ax.plot(ref_x, 2.0 * ref_x, color="#808080", lw=1.0, linestyle="--", label="Doubling ($y=2x$)", zorder=3)
    ax.plot(ref_x, 0.5 * ref_x, color="#808080", lw=1.0, linestyle="--", label="Halving ($y=0.5x$)", zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(min_v, max_v)
    ax.set_ylim(min_v, max_v)
    ax.set_aspect("equal", adjustable="box")
    
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.set_xlabel(r"Streamflow at Time $t$, $Q(t)$ (cfs)", fontsize=10, fontweight="bold")
    ax.set_ylabel(r"Streamflow at Time $t+1$, $Q(t+1)$ (cfs)", fontsize=10, fontweight="bold")
    ax.set_title("MOD-02: Lag-1 Differential Scatterplot", fontsize=12, fontweight="bold", pad=40)
    apply_standard_grid(ax)

    ax.legend(bbox_to_anchor=(0.5, 1.02), loc="lower center", ncol=3, frameon=True, facecolor="#ffffff", edgecolor="#cccccc", fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------
# Transition Matrix Helper (MOD-03, MOD-04, MOD-05)
# ---------------------------------------------------------
def calculate_transition_matrix(df, date_col, value_col, delta_log10=0.10):
    df_pairs = prepare_pairs_and_dowy(df, date_col, value_col)
    pairs = df_pairs[df_pairs["valid_pair"]].copy()
    q_t = pairs["Q_t"].values
    q_next = pairs["Q_next"].values

    log_min = np.floor(np.log10(min(q_t.min(), q_next.min())))
    log_max = np.ceil(np.log10(max(q_t.max(), q_next.max())))
    bins = 10 ** np.arange(log_min, log_max + delta_log10, delta_log10)
    
    counts, xedges, yedges = np.histogram2d(q_t, q_next, bins=[bins, bins])
    return counts, xedges, yedges, log_min, log_max


# ---------------------------------------------------------
# MOD-03: Discrete Transition Matrix (Raw Frequency Counts)
# ---------------------------------------------------------
def run_mod03(df, date_col, value_col):
    counts, xedges, yedges, log_min, log_max = calculate_transition_matrix(df, date_col, value_col)
    num_bins = len(xedges) - 1
    
    fig, ax = plt.subplots(figsize=(8, 8))
    for i in range(num_bins):
        for j in range(num_bins):
            cnt = int(counts[i, j])
            if cnt > 0:
                xc = 0.5 * (xedges[i] + xedges[i+1])
                yc = 0.5 * (yedges[j] + yedges[j+1])
                if j > i:
                    col = "#0000FF"  # Rising
                elif j < i:
                    col = "#FF0000"  # Falling
                else:
                    col = "#000000"  # Steady State Diagonal
                bbox_prop = dict(boxstyle="square,pad=0.15", facecolor="#ffffff", edgecolor="none", alpha=0.9) if i == j else None
                ax.text(xc, yc, str(cnt), color=col, fontsize=6.5, fontweight="bold", ha="center", va="center", bbox=bbox_prop)

    min_v, max_v = 10**log_min, 10**log_max
    ax.plot([min_v, max_v], [min_v, max_v], color="#000000", lw=1.2, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(min_v, max_v)
    ax.set_ylim(min_v, max_v)
    ax.set_aspect("equal", adjustable="box")
    
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.set_xlabel(r"$Q(t)$ (cfs)", fontsize=10, fontweight="bold")
    ax.set_ylabel(r"$Q(t+1)$ (cfs)", fontsize=10, fontweight="bold")
    ax.set_title("MOD-03: Discrete State Transition Matrix (Counts)", fontsize=12, fontweight="bold", pad=15)
    apply_standard_grid(ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------
# MOD-04: Conditional Forecast Probability Matrix
# ---------------------------------------------------------
def run_mod04(df, date_col, value_col):
    counts, xedges, yedges, log_min, log_max = calculate_transition_matrix(df, date_col, value_col)
    num_bins = len(xedges) - 1
    
    col_sums = counts.sum(axis=1, keepdims=True)
    prob = np.divide(counts, col_sums, out=np.zeros_like(counts), where=col_sums != 0) * 100.0

    fig, ax = plt.subplots(figsize=(8, 8))
    for i in range(num_bins):
        for j in range(num_bins):
            val = prob[i, j]
            if val > 0:
                xc = 0.5 * (xedges[i] + xedges[i+1])
                yc = 0.5 * (yedges[j] + yedges[j+1])
                col = "#0000FF" if j > i else ("#FF0000" if j < i else "#000000")
                txt = f"{val:.1f}%" if val >= 1.0 else f"{val:.2f}%"
                bbox_prop = dict(boxstyle="square,pad=0.15", facecolor="#ffffff", edgecolor="none", alpha=0.9) if i == j else None
                ax.text(xc, yc, txt, color=col, fontsize=5.5, fontweight="bold", ha="center", va="center", bbox=bbox_prop)

    min_v, max_v = 10**log_min, 10**log_max
    ax.plot([min_v, max_v], [min_v, max_v], color="#000000", lw=1.2, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(min_v, max_v)
    ax.set_ylim(min_v, max_v)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.set_xlabel(r"$Q(t)$ (cfs)", fontsize=10, fontweight="bold")
    ax.set_ylabel(r"$Q(t+1)$ (cfs)", fontsize=10, fontweight="bold")
    ax.set_title("Forecast / Column-Normalized Matrix [P(Q(t+1) | Q(t))]", fontsize=12, fontweight="bold", pad=15)
    apply_standard_grid(ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------
# MOD-05: Conditional Antecedent Probability Matrix
# ---------------------------------------------------------
def run_mod05(df, date_col, value_col):
    counts, xedges, yedges, log_min, log_max = calculate_transition_matrix(df, date_col, value_col)
    num_bins = len(xedges) - 1
    
    row_sums = counts.sum(axis=0, keepdims=True)
    prob = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums != 0) * 100.0

    fig, ax = plt.subplots(figsize=(8, 8))
    for i in range(num_bins):
        for j in range(num_bins):
            val = prob[i, j]
            if val > 0:
                xc = 0.5 * (xedges[i] + xedges[i+1])
                yc = 0.5 * (yedges[j] + yedges[j+1])
                col = "#0000FF" if j > i else ("#FF0000" if j < i else "#000000")
                txt = f"{val:.1f}%" if val >= 1.0 else f"{val:.2f}%"
                bbox_prop = dict(boxstyle="square,pad=0.15", facecolor="#ffffff", edgecolor="none", alpha=0.9) if i == j else None
                ax.text(xc, yc, txt, color=col, fontsize=5.5, fontweight="bold", ha="center", va="center", bbox=bbox_prop)

    min_v, max_v = 10**log_min, 10**log_max
    ax.plot([min_v, max_v], [min_v, max_v], color="#000000", lw=1.2, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(min_v, max_v)
    ax.set_ylim(min_v, max_v)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.set_xlabel(r"$Q(t)$ (cfs)", fontsize=10, fontweight="bold")
    ax.set_ylabel(r"$Q(t+1)$ (cfs)", fontsize=10, fontweight="bold")
    ax.set_title("Antecedent / Row-Normalized Matrix [P(Q(t) | Q(t+1))]", fontsize=12, fontweight="bold", pad=15)
    apply_standard_grid(ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------
# MOD-06: Sequential Flow Duration & Persistence Analysis
# ---------------------------------------------------------
def run_mod06(df, date_col, value_col):
    df_clean = df.dropna(subset=[date_col, value_col]).sort_values(by=date_col).copy()
    q = df_clean[value_col].values
    
    log_q = np.floor(np.log10(q) / 0.20) * 0.20
    runs = []
    curr_bin = log_q[0]
    curr_len = 1
    for b in log_q[1:]:
        if b == curr_bin:
            curr_len += 1
        else:
            runs.append((10**curr_bin, curr_len))
            curr_bin = b
            curr_len = 1
    runs.append((10**curr_bin, curr_len))
    rdf = pd.DataFrame(runs, columns=["Q_bin", "Duration"])
    tally = rdf.groupby(["Q_bin", "Duration"]).size().reset_index(name="Count")

    fig, ax = plt.subplots(figsize=(11, 7))
    for _, row in tally.iterrows():
        ax.text(row["Q_bin"], row["Duration"], str(int(row["Count"])),
                color="#000000", fontweight="bold", fontsize=7.5, ha="center", va="center")

    max_dur = max(28, int(np.ceil(tally["Duration"].max() / 7.0) * 7))
    ax.set_xscale("log")
    ax.set_ylim(0, max_dur)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    
    ax.yaxis.set_major_locator(ticker.MultipleLocator(7))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.grid(True, which="major", color="#000000", linestyle="-", lw=0.9)
    ax.grid(True, which="minor", color="#B0B0B0", linestyle="--", lw=0.45)
    
    ax.set_xlabel(f"Categorized Discharge ({value_col})", fontsize=10, fontweight="bold")
    ax.set_ylabel("Sequence Length in Continuous Days", fontsize=10, fontweight="bold")
    ax.set_title("MOD-06: Sequential Flow Duration & Persistence Analysis", fontsize=12, fontweight="bold", pad=15)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------
# MOD-07: Chronological Raster Hydrograph
# ---------------------------------------------------------
def run_mod07(df, date_col, value_col):
    df_clean = prepare_pairs_and_dowy(df, date_col, value_col)
    pivot = df_clean.pivot(index="WaterYear", columns="Standard_DOWY", values=value_col)
    pivot = pivot.reindex(columns=range(1, 367)).sort_index(ascending=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    log_data = np.log10(np.maximum(0.01, pivot.values))
    cmap = plt.cm.Spectral_r.copy()
    cmap.set_bad(color="#d9d9d9")

    mesh = ax.imshow(log_data, aspect="auto", cmap=cmap, origin="lower",
                     extent=[1, 366, pivot.index.min() - 0.5, pivot.index.max() + 0.5])

    cbar = plt.colorbar(mesh, ax=ax, pad=0.03)
    cbar.set_label(f"Discharge ({value_col}) [Log10 Scale]", fontsize=9, fontweight="bold")

    for t in MONTH_TICKS_366[1:]:
        ax.axvline(t - 0.5, color="#000000", linestyle="--", lw=0.75, alpha=0.7)

    ax.set_xticks(MONTH_TICKS_366)
    ax.set_xticklabels(MONTH_LABELS, fontsize=9, fontweight="bold")
    ax.set_xlim(1, 366)
    
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.grid(True, which="major", axis="y", color="#000000", linestyle="-", lw=0.8)

    ax.set_xlabel("Day of Water Year (Leap-Adjusted)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Water Year (Chronological)", fontsize=10, fontweight="bold")
    ax.set_title("MOD-07: Chronological Raster Hydrograph", fontsize=12, fontweight="bold", pad=15)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------
# MOD-08: Volumetric Raster Hydrograph
# ---------------------------------------------------------
def run_mod08(df, date_col, value_col):
    df_clean = prepare_pairs_and_dowy(df, date_col, value_col)
    
    annual_vol = df_clean.groupby("WaterYear")[value_col].sum().sort_values(ascending=False)
    ranked_wy = annual_vol.index
    
    pivot = df_clean.pivot(index="WaterYear", columns="Standard_DOWY", values=value_col)
    pivot = pivot.reindex(index=ranked_wy, columns=range(1, 367))

    fig, ax = plt.subplots(figsize=(12, 8))
    log_data = np.log10(np.maximum(0.01, pivot.values))
    cmap = plt.cm.Spectral_r.copy()
    cmap.set_bad(color="#d9d9d9")

    mesh = ax.imshow(log_data, aspect="auto", cmap=cmap, origin="upper",
                     extent=[1, 366, len(ranked_wy) + 0.5, 0.5])

    cbar = plt.colorbar(mesh, ax=ax, pad=0.03)
    cbar.set_label(f"Discharge ({value_col}) [Log10 Scale]", fontsize=9, fontweight="bold")

    for t in MONTH_TICKS_366[1:]:
        ax.axvline(t - 0.5, color="#000000", linestyle="--", lw=0.75, alpha=0.7)

    ax.set_xticks(MONTH_TICKS_366)
    ax.set_xticklabels(MONTH_LABELS, fontsize=9, fontweight="bold")
    ax.set_xlim(1, 366)
    
    n_ranks = len(ranked_wy)
    major_ranks = [1] + [r for r in range(10, n_ranks + 1, 10)]
    ax.set_yticks(major_ranks)
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.grid(True, which="major", axis="y", color="#000000", linestyle="-", lw=0.8)

    ax.set_xlabel("Day of Water Year (Leap-Adjusted)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Volumetric Rank (1 = Greatest Annual Volume)", fontsize=10, fontweight="bold")
    ax.set_title("MOD-08: Volumetric Raster Hydrograph (Ranked)", fontsize=12, fontweight="bold", pad=15)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------
# MOD-09: Annual FDC Spaghetti Plot
# ---------------------------------------------------------
def run_mod09(df, date_col, value_col):
    df_clean = prepare_pairs_and_dowy(df, date_col, value_col)
    years = sorted(df_clean["WaterYear"].unique())
    norm = Normalize(vmin=min(years), vmax=max(years))
    cmap = plt.cm.Spectral_r

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for y in years:
        sub = df_clean[df_clean["WaterYear"] == y][value_col].dropna().values
        if len(sub) > 10:
            q_s = np.sort(sub)[::-1]
            p = (np.arange(1, len(q_s) + 1) / (len(q_s) + 1.0)) * 100.0
            ax.plot(p, q_s, color=cmap(norm(y)), alpha=0.35, lw=1.3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, ax=ax, pad=0.03)
    cbar.set_label("Water Year", fontsize=9, fontweight="bold")

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
    ax.set_xlabel("Exceedance Percentage (%)", fontsize=10, fontweight="bold")
    ax.set_ylabel(f"Discharge ({value_col})", fontsize=10, fontweight="bold")
    ax.set_title("MOD-09: Annual Flow Duration Curve Spaghetti Plot", fontsize=12, fontweight="bold", pad=15)
    apply_standard_grid(ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------
# Annual Threshold Calculation Helper (MOD-10 & MOD-11)
# ---------------------------------------------------------
def get_annual_thresholds(df, date_col, value_col):
    df_clean = prepare_pairs_and_dowy(df, date_col, value_col)
    records = []
    for y, grp in df_clean.groupby("WaterYear"):
        q = grp[value_col].dropna().values
        if len(q) > 30:
            records.append({
                "WaterYear": y,
                "Volume": q.sum(),
                "Q10": np.percentile(q, 90),
                "Q50": np.percentile(q, 50),
                "Q90": np.percentile(q, 10),
            })
    tdf = pd.DataFrame(records)
    tdf["VolumeRank"] = tdf["Volume"].rank(ascending=False, method="min").astype(int)
    return tdf.sort_values(by="WaterYear")


# ---------------------------------------------------------
# MOD-10: Annual FDC Threshold Trends (Chronological)
# ---------------------------------------------------------
def run_mod10(df, date_col, value_col):
    tdf = get_annual_thresholds(df, date_col, value_col)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    
    ax.plot(tdf["WaterYear"], tdf["Q10"], "o-", color="#0000FF", lw=1.5, markersize=5, label="Q10 (High Flow / Floods)")
    ax.plot(tdf["WaterYear"], tdf["Q50"], "o-", color="#FFD700", lw=1.5, markersize=5, label="Q50 (Median Flow)")
    ax.plot(tdf["WaterYear"], tdf["Q90"], "o-", color="#FF0000", lw=1.5, markersize=5, label="Q90 (Low Flow / Drought Baseflow)")

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.set_xlabel("Water Year", fontsize=10, fontweight="bold")
    ax.set_ylabel(f"Discharge ({value_col})", fontsize=10, fontweight="bold")
    ax.set_title("MOD-10: Annual FDC Threshold Trends (Chronological)", fontsize=12, fontweight="bold", pad=40)
    apply_standard_grid(ax)

    ax.legend(bbox_to_anchor=(0.5, 1.02), loc="lower center", ncol=3, frameon=True, facecolor="#ffffff", edgecolor="#cccccc", fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------
# MOD-11: Annual FDC Volumetric Thresholds (Rank-Ordered)
# ---------------------------------------------------------
def run_mod11(df, date_col, value_col):
    tdf = get_annual_thresholds(df, date_col, value_col).sort_values(by="VolumeRank")
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    
    ax.plot(tdf["VolumeRank"], tdf["Q10"], "o-", color="#0000FF", lw=1.5, markersize=5, label="Q10 (High Flow / Floods)")
    ax.plot(tdf["VolumeRank"], tdf["Q50"], "o-", color="#FFD700", lw=1.5, markersize=5, label="Q50 (Median Flow)")
    ax.plot(tdf["VolumeRank"], tdf["Q90"], "o-", color="#FF0000", lw=1.5, markersize=5, label="Q90 (Low Flow / Drought Baseflow)")

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.set_xlabel("Annual Volume Rank (1 = Wettest)", fontsize=10, fontweight="bold")
    ax.set_ylabel(f"Discharge ({value_col})", fontsize=10, fontweight="bold")
    ax.set_title("MOD-11: Annual FDC Volumetric Thresholds (Rank-Ordered)", fontsize=12, fontweight="bold", pad=40)
    apply_standard_grid(ax)

    ax.legend(bbox_to_anchor=(0.5, 1.02), loc="lower center", ncol=3, frameon=True, facecolor="#ffffff", edgecolor="#cccccc", fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------
# MOD-12: Composite Hydroinformatics Dashboard
# ---------------------------------------------------------
def run_mod12(df, date_col, value_col):
    fig = plt.figure(figsize=(16, 12), dpi=300)
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.25)
    
    # Panel A: MOD-02
    ax_a = fig.add_subplot(gs[0, 0])
    df_pairs = prepare_pairs_and_dowy(df, date_col, value_col)
    pairs = df_pairs[df_pairs["valid_pair"]].copy()
    q_t, q_next = pairs["Q_t"].values, pairs["Q_next"].values
    dq = q_next - q_t
    ax_a.scatter(q_t[dq > 0], q_next[dq > 0], color="#0000FF", alpha=0.3, s=8)
    ax_a.scatter(q_t[dq < 0], q_next[dq < 0], color="#FF0000", alpha=0.3, s=8)
    ax_a.scatter(q_t[dq == 0], q_next[dq == 0], color="#000000", alpha=0.3, s=8)
    min_v, max_v = 10**np.floor(np.log10(min(q_t.min(), q_next.min()))), 10**np.ceil(np.log10(max(q_t.max(), q_next.max())))
    ax_a.plot([min_v, max_v], [min_v, max_v], color="#000000", lw=1.0)
    ax_a.set_xscale("log"); ax_a.set_yscale("log")
    ax_a.set_xlim(min_v, max_v); ax_a.set_ylim(min_v, max_v)
    ax_a.set_aspect("equal", adjustable="box")
    ax_a.set_title("A: Lag-1 Differential Scatter", fontsize=11, fontweight="bold", loc="left", pad=10)
    apply_standard_grid(ax_a)

    # Panel B: MOD-03
    ax_b = fig.add_subplot(gs[0, 1])
    counts, xedges, yedges, lmin, lmax = calculate_transition_matrix(df, date_col, value_col, delta_log10=0.20)
    for i in range(len(xedges)-1):
        for j in range(len(yedges)-1):
            cnt = int(counts[i, j])
            if cnt > 0:
                col = "#0000FF" if j > i else ("#FF0000" if j < i else "#000000")
                ax_b.text(0.5*(xedges[i]+xedges[i+1]), 0.5*(yedges[j]+yedges[j+1]), str(cnt),
                          color=col, fontsize=5.5, fontweight="bold", ha="center", va="center")
    ax_b.plot([10**lmin, 10**lmax], [10**lmin, 10**lmax], color="#000000", lw=1.0)
    ax_b.set_xscale("log"); ax_b.set_yscale("log")
    ax_b.set_xlim(10**lmin, 10**lmax); ax_b.set_ylim(10**lmin, 10**lmax)
    ax_b.set_aspect("equal", adjustable="box")
    ax_b.set_title("B: Discrete Transition Matrix", fontsize=11, fontweight="bold", loc="left", pad=10)
    apply_standard_grid(ax_b)

    # Panel C: MOD-07
    ax_c = fig.add_subplot(gs[1, 0])
    pivot = df_pairs.pivot(index="WaterYear", columns="Standard_DOWY", values=value_col).reindex(columns=range(1, 367)).sort_index()
    cmap = plt.cm.Spectral_r.copy()
    cmap.set_bad(color="#d9d9d9")
    mesh_c = ax_c.imshow(np.log10(np.maximum(0.01, pivot.values)), aspect="auto", cmap=cmap, origin="lower",
                         extent=[1, 366, pivot.index.min() - 0.5, pivot.index.max() + 0.5])
    plt.colorbar(mesh_c, ax=ax_c, pad=0.03, label="Log10 Discharge")
    ax_c.set_xticks(MONTH_TICKS_366); ax_c.set_xticklabels(MONTH_LABELS, fontsize=8)
    ax_c.set_title("C: Chronological Raster", fontsize=11, fontweight="bold", loc="left", pad=10)

    # Panel D: MOD-10
    ax_d = fig.add_subplot(gs[1, 1])
    tdf = get_annual_thresholds(df, date_col, value_col)
    ax_d.plot(tdf["WaterYear"], tdf["Q10"], "o-", color="#0000FF", lw=1.2, markersize=4, label="Q10")
    ax_d.plot(tdf["WaterYear"], tdf["Q50"], "o-", color="#FFD700", lw=1.2, markersize=4, label="Q50")
    ax_d.plot(tdf["WaterYear"], tdf["Q90"], "o-", color="#FF0000", lw=1.2, markersize=4, label="Q90")
    ax_d.set_yscale("log")
    ax_d.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax_d.set_title("D: Annual FDC Threshold Trends", fontsize=11, fontweight="bold", loc="left", pad=10)
    ax_d.legend(loc="lower left", fontsize=7.5)
    apply_standard_grid(ax_d)

    return fig


# ---------------------------------------------------------
# Sidebar Controls & File Ingestion
# ---------------------------------------------------------
st.sidebar.title("🌊 Hydro-Studio")
st.sidebar.markdown("**Operational Modular Visual Analytics (v8.3)**")
st.sidebar.caption("Visual Data Analytics, LLC (© VDA, 2026)")
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader("Upload Daily CSV", type=["csv"])

modules = {
    "MOD-01: Enhanced Flow Duration Curve (eFDC)": run_mod01,
    "MOD-02: Lag-1 Differential Scatterplot": run_mod02,
    "MOD-03: Discrete Transition Matrix (Counts)": run_mod03,
    "MOD-04: Forecast Probability Matrix [P(Q_t+1 | Q_t)]": run_mod04,
    "MOD-05: Antecedent Probability Matrix [P(Q_t | Q_t+1)]": run_mod05,
    "MOD-06: Sequential Flow Duration & Persistence": run_mod06,
    "MOD-07: Chronological Raster Hydrograph": run_mod07,
    "MOD-08: Volumetric Raster Hydrograph (Ranked)": run_mod08,
    "MOD-09: Annual FDC Spaghetti Plot": run_mod09,
    "MOD-10: Annual FDC Threshold Trends (Chronological)": run_mod10,
    "MOD-11: Annual FDC Volumetric Thresholds (Ranked)": run_mod11,
    "MOD-12: Composite Hydroinformatics Dashboard": run_mod12,
}

if uploaded_file is not None:
    try:
        df_current = pd.read_csv(uploaded_file)
        st.sidebar.success("CSV Ingested Successfully")
        
        col_names = list(df_current.columns)
        date_candidates = [c for c in col_names if "date" in c.lower() or "time" in c.lower()]
        val_candidates = [c for c in col_names if c not in date_candidates]

        date_col = st.sidebar.selectbox("Date Column", col_names, index=col_names.index(date_candidates[0]) if date_candidates else 0)
        val_col = st.sidebar.selectbox("Discharge Column", col_names, index=col_names.index(val_candidates[0]) if val_candidates else (1 if len(col_names) > 1 else 0))

        st.sidebar.markdown("---")
        selected_module = st.sidebar.selectbox("Select Operational Module", list(modules.keys()))

        # Controlled Execution: "Generate Plot" Action Button
        st.sidebar.markdown("---")
        generate_clicked = st.sidebar.button("🚀 Generate Plot", type="primary", use_container_width=True)

        if "active_fig" not in st.session_state:
            st.session_state.active_fig = None
        if "active_module" not in st.session_state:
            st.session_state.active_module = None

        if generate_clicked:
            with st.spinner("Processing hydrology data and rendering graphic..."):
                module_func = modules[selected_module]
                st.session_state.active_fig = module_func(df_current, date_col, val_col)
                st.session_state.active_module = selected_module

    except Exception as e:
        st.sidebar.error(f"Error reading CSV: {e}")
        st.session_state.active_fig = None
else:
    st.session_state.active_fig = None

# ---------------------------------------------------------
# Main Panel Display & Download Execution
# ---------------------------------------------------------
if uploaded_file is None:
    st.info("👈 Please upload a daily streamflow CSV file in the sidebar to begin.")
elif st.session_state.active_fig is not None:
    st.pyplot(st.session_state.active_fig)
    
    # Export high-resolution PNG to buffer for download
    buf = io.BytesIO()
    st.session_state.active_fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    
    file_prefix = st.session_state.active_module.split(":")[0].strip()
    
    st.download_button(
        label="💾 Download High-Resolution Plot (PNG - 300 DPI)",
        data=buf,
        file_name=f"{file_prefix}_publication.png",
        mime="image/png",
        use_container_width=True
    )
    
    with st.expander("📋 View Operational Module Metadata"):
        st.markdown(f"**Routine:** {st.session_state.active_module}")
        st.caption("Rendered under Visual Data Analytics, LLC Operational Library standards (v8.3).")
else:
    st.info("👈 Select your operational module in the sidebar and click **'Generate Plot'** to render the visualization.")
