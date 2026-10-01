"""
Hydro-Studio: Hydrological Visual Analytics & Graphics Engine
Based on: Operational Modular Data Evaluation and Prompt Library (Version v8.5)
© Visual Data Analytics, LLC (2026.1)
"""

import io
import calendar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
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
# USGS Color Progression (Version v8.5 Formal Specification)
# ---------------------------------------------------------
USGS_STOPS = [
    (0.000, "#FFFFFF"),  # White
    (0.125, "#891A10"),  # Darkred
    (0.179, "#D93427"),  # Crimson
    (0.303, "#EF8834"),  # Coral
    (0.357, "#F2AB3C"),  # Sandybrown
    (0.428, "#F8D548"),  # Annabanana
    (0.482, "#FDF451"),  # Khaki
    (0.536, "#D1F34E"),  # Greenyellow
    (0.607, "#81DC42"),  # Yellowgreen
    (0.666, "#5ECB3C"),  # Limegreen
    (0.714, "#4EAC4B"),  # Mediumseagreen
    (0.785, "#39828E"),  # Steelblue
    (0.839, "#2A63C4"),  # Royalblue
    (0.893, "#1942B4"),  # Darkslateblue
    (0.946, "#0921A3"),  # Deeproyalblue
    (1.000, "#000094"),  # Midnight blue
]

# Forward USGS colormap
usgs_cmap = LinearSegmentedColormap.from_list("USGS_Streamflow", USGS_STOPS, N=256)
usgs_cmap.set_bad(color="#d9d9d9")

# Reversed USGS colormap (Red/White = Low, Blue = High)
usgs_cmap_r = usgs_cmap.reversed()
usgs_cmap_r.set_bad(color="#d9d9d9")


# ---------------------------------------------------------
# Data Utilities & Calculations
# ---------------------------------------------------------
def prepare_pairs_and_dowy(df, date_col, value_col):
    """Cleans data, checks 1-day step continuity, and formats standardized 366-day DOWY."""
    df_clean = df.dropna(subset=[date_col, value_col]).copy()
    df_clean[date_col] = pd.to_datetime(df_clean[date_col])
    df_clean = df_clean.sort_values(by=date_col).reset_index(drop=True)

    # Physical streamflow bounds: Q > 0
    df_clean = df_clean[df_clean[value_col] > 0].copy()

    # Axiomatic 1-day temporal continuity: dQ/dt = Q(t+1) - Q(t)
    dt = df_clean[date_col].diff().dt.total_seconds() / 86400.0
    valid_pair = (dt == 1.0)

    df_clean["Q_t"] = df_clean[value_col].shift(0)
    df_clean["Q_next"] = df_clean[value_col].shift(-1)
    df_clean["valid_pair"] = valid_pair.shift(-1).fillna(False)

    # Water Year and 366-day calendar alignment (Feb 29 anchored to Day 152)
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


def draw_leap_adjusted_dividers(ax, years, is_ranked=False):
    """
    Draws dashed black month dividers accounting for leap water years.
    Non-leap: Oct(0.5), Nov(31.5), Dec(61.5), Jan(92.5), Feb(123.5),
              Mar(151.5), Apr(182.5), May(212.5), Jun(243.5), Jul(273.5), Aug(304.5), Sep(335.5)
    Leap (+1 day post-Feb 29): Mar(152.5), Apr(183.5), May(213.5), Jun(244.5), Jul(274.5), Aug(305.5), Sep(336.5)
    """
    # Pre-March dividers (constant across all years)
    pre_mar = [31.5, 61.5, 92.5, 123.5]
    for x in pre_mar:
        ax.axvline(x, color="#000000", linestyle="--", lw=0.75, zorder=3)

    # Post-February base dividers for non-leap years
    post_mar_nonleap = [151.5, 182.5, 212.5, 243.5, 273.5, 304.5, 335.5]

    for idx, wy in enumerate(years):
        y_bottom = idx - 0.5
        y_top = idx + 0.5
        is_leap = calendar.isleap(int(wy))
        offset = 1.0 if is_leap else 0.0

        for base_x in post_mar_nonleap:
            x = base_x + offset
            ax.plot([x, x], [y_bottom, y_top], color="#000000", linestyle="--", lw=0.75, zorder=3)


# ---------------------------------------------------------
# Visualization Modules (MOD-01 through MOD-12)
# ---------------------------------------------------------

def run_mod01(df, date_col, value_col):
    """MOD-01: Enhanced Flow Duration Curve (eFDC)."""
    df_clean = prepare_pairs_and_dowy(df, date_col, value_col)
    n = len(df_clean)
    sorted_q = np.sort(df_clean[value_col].values)[::-1]
    prob = (np.arange(1, n + 1) / (n + 1)) * 100.0

    rank_lookup = pd.Series(prob, index=sorted_q)
    df_pairs = df_clean[df_clean["valid_pair"]].copy()

    # Empirical exceedance probability lookup
    df_pairs["prob_t"] = np.interp(df_pairs["Q_t"], sorted_q[::-1], prob[::-1])
    diff = df_pairs["Q_next"] - df_pairs["Q_t"]

    rising = df_pairs[diff > 0]
    falling = df_pairs[diff < 0]

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(rising["prob_t"], rising["Q_next"], color="#1f77b4", s=12, alpha=0.6, label="Rising Limb (+dQ/dt)", zorder=3)
    ax.scatter(falling["prob_t"], falling["Q_next"], color="#d62728", s=12, alpha=0.6, label="Falling Limb (-dQ/dt)", zorder=3)
    ax.plot(prob, sorted_q, color="#000000", lw=2.0, label="Standard FDC (Q_t)", zorder=4)

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}" if y < 1 else f"{int(y):,}"))

    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 10))
    apply_standard_grid(ax)

    ax.set_xlabel("Exceedance Probability (%)", fontsize=10, fontweight="bold")
    ax.set_ylabel(f"Discharge ({value_col}) [cfs]", fontsize=10, fontweight="bold")
    ax.set_title("MOD-01: Enhanced Flow Duration Curve (eFDC)", fontsize=12, fontweight="bold", pad=28)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=3, frameon=False, fontsize=9)

    fig.tight_layout()
    return fig


def run_mod02(df, date_col, value_col):
    """MOD-02: Lag-1 Streamflow Scatter Plot."""
    df_clean = prepare_pairs_and_dowy(df, date_col, value_col)
    df_pairs = df_clean[df_clean["valid_pair"]].copy()

    min_val = min(df_pairs["Q_t"].min(), df_pairs["Q_next"].min())
    max_val = max(df_pairs["Q_t"].max(), df_pairs["Q_next"].max())
    log_min = 10 ** np.floor(np.log10(min_val))
    log_max = 10 ** np.ceil(np.log10(max_val))

    fig, ax = plt.subplots(figsize=(8, 8))

    diff = df_pairs["Q_next"] - df_pairs["Q_t"]
    rising = df_pairs[diff > 0]
    falling = df_pairs[diff < 0]
    eq = df_pairs[diff == 0]

    ax.scatter(rising["Q_t"], rising["Q_next"], color="#1f77b4", s=14, alpha=0.5, label="Rising Limb (+dQ/dt)", zorder=3)
    ax.scatter(falling["Q_t"], falling["Q_next"], color="#d62728", s=14, alpha=0.5, label="Falling Limb (-dQ/dt)", zorder=3)
    ax.scatter(eq["Q_t"], eq["Q_next"], color="#808080", s=14, alpha=0.5, label="Equilibrium (dQ/dt = 0)", zorder=3)

    # 1:1 Identity Line
    ax.plot([log_min, log_max], [log_min, log_max], color="#000000", lw=1.5, linestyle="-", label="1:1 Identity", zorder=4)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(log_min, log_max)
    ax.set_ylim(log_min, log_max)

    for a in [ax.xaxis, ax.yaxis]:
        a.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
        a.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}" if y < 1 else f"{int(y):,}"))

    apply_standard_grid(ax)
    ax.set_xlabel(f"Q(t) ({value_col}) [cfs]", fontsize=10, fontweight="bold")
    ax.set_ylabel(f"Q(t+1) ({value_col}) [cfs]", fontsize=10, fontweight="bold")
    ax.set_title("MOD-02: Lag-1 Streamflow Scatter Plot", fontsize=12, fontweight="bold", pad=28)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=3, frameon=False, fontsize=8.5)

    fig.tight_layout()
    return fig


def run_mod03(df, date_col, value_col):
    """MOD-03: Discrete Transition Matrix (Raw Frequency Counts)."""
    df_clean = prepare_pairs_and_dowy(df, date_col, value_col)
    df_pairs = df_clean[df_clean["valid_pair"]].copy()

    # Identify dynamic decade boundaries spanning the full data range
    min_val = min(df_pairs["Q_t"].min(), df_pairs["Q_next"].min())
    max_val = max(df_pairs["Q_t"].max(), df_pairs["Q_next"].max())
    dec_min = int(np.floor(np.log10(min_val)))
    dec_max = int(np.ceil(np.log10(max_val)))

    # 4 logarithmic subdivisions per decade (delta log10 = 0.25)
    bins_log = np.arange(dec_min, dec_max + 0.25, 0.25)
    bins = 10 ** bins_log
    n_bins = len(bins) - 1

    # 2D histogram tally: x = Q(t), y = Q(t+1)
    counts, _, _ = np.histogram2d(df_pairs["Q_t"], df_pairs["Q_next"], bins=bins)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect("equal")

    # Set log scaling and bounds
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(10 ** dec_min, 10 ** dec_max)
    ax.set_ylim(10 ** dec_min, 10 ** dec_max)

    # Major decade gridlines (solid black) and minor intermediate lines (dashed gray)
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}" if y < 1 else f"{int(y):,}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}" if y < 1 else f"{int(y):,}"))

    ax.grid(True, which="major", color="#000000", linestyle="-", lw=0.9, zorder=1)
    ax.grid(True, which="minor", color="#808080", linestyle="--", lw=0.5, zorder=1)

    # Bin cell boundary lines
    for b in bins:
        ax.axvline(b, color="#e0e0e0", lw=0.4, zorder=2)
        ax.axhline(b, color="#e0e0e0", lw=0.4, zorder=2)

    # 1:1 Identity Line
    ax.plot([10 ** dec_min, 10 ** dec_max], [10 ** dec_min, 10 ** dec_max], color="#000000", lw=1.2, linestyle="-", zorder=3)

    # Cell integer frequency annotations
    for i in range(n_bins):      # x-axis / Q(t)
        for j in range(n_bins):  # y-axis / Q(t+1)
            cnt = int(counts[i, j])
            if cnt > 0:
                xc = 10 ** ((bins_log[i] + bins_log[i + 1]) / 2.0)
                yc = 10 ** ((bins_log[j] + bins_log[j + 1]) / 2.0)

                if j > i:    # Rising (+dQ/dt)
                    ax.text(xc, yc, f"{cnt}", ha="center", va="center", color="#1f77b4", fontweight="bold", fontsize=6.5, zorder=5)
                elif j < i:  # Falling (-dQ/dt)
                    ax.text(xc, yc, f"{cnt}", ha="center", va="center", color="#d62728", fontweight="bold", fontsize=6.5, zorder=5)
                else:        # Steady-state diagonal (j == i) with white background mask
                    ax.text(xc, yc, f"{cnt}", ha="center", va="center", color="#000000", fontweight="bold", fontsize=6.5, zorder=5,
                            bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none", alpha=0.9))

    ax.set_xlabel(f"Q(t) ({value_col}) [cfs]", fontsize=10, fontweight="bold")
    ax.set_ylabel(f"Q(t+1) ({value_col}) [cfs]", fontsize=10, fontweight="bold")
    ax.set_title("MOD-03: Discrete Transition Matrix (Raw Frequency Counts)", fontsize=11, fontweight="bold", pad=12)

    fig.tight_layout()
    return fig


def run_mod04(df, date_col, value_col):
    """MOD-04: Water Year Overlay Hydrograph."""
    df_clean = prepare_pairs_and_dowy(df, date_col, value_col)
    years = sorted(df_clean["WaterYear"].unique())

    fig, ax = plt.subplots(figsize=(11, 7))
    cmap = plt.cm.get_cmap("Spectral_r", len(years))

    for idx, wy in enumerate(years):
        sub = df_clean[df_clean["WaterYear"] == wy].sort_values("Standard_DOWY")
        ax.plot(sub["Standard_DOWY"], sub[value_col], color=cmap(idx), lw=0.8, alpha=0.7)

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}" if y < 1 else f"{int(y):,}"))

    ax.set_xlim(1, 366)
    ax.set_xticks(range(50, 366, 50))
    apply_standard_grid(ax)

    # Top Month Labels
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    month_centers = [16.0, 46.5, 77.0, 108.0, 138.0, 168.0, 198.5, 229.0, 259.5, 290.0, 321.0, 351.5]
    ax_top.set_xticks(month_centers)
    ax_top.set_xticklabels(MONTH_LABELS, fontsize=9, fontweight="bold")
    ax_top.tick_params(length=0, pad=4)

    ax.set_xlabel("Day of Water Year (Starting Oct 1)", fontsize=10, fontweight="bold")
    ax.set_ylabel(f"Discharge ({value_col}) [cfs]", fontsize=10, fontweight="bold")
    ax.set_title("MOD-04: Water Year Overlay Hydrograph", fontsize=12, fontweight="bold", pad=32)

    fig.tight_layout()
    return fig


def run_mod05(df, date_col, value_col):
    """MOD-05: Discrete State Transition Probability Matrix."""
    df_clean = prepare_pairs_and_dowy(df, date_col, value_col)
    df_pairs = df_clean[df_clean["valid_pair"]].copy()

    # Standard USGS 5-tier percentiles
    quantiles = [0.0, 0.10, 0.25, 0.75, 0.90, 1.0]
    bin_edges = np.quantile(df_clean[value_col], quantiles)
    labels = ["Low (<10%)", "Below Normal (10-25%)", "Normal (25-75%)", "Above Normal (75-90%)", "High (>90%)"]

    df_pairs["state_t"] = pd.cut(df_pairs["Q_t"], bins=bin_edges, labels=labels, include_lowest=True)
    df_pairs["state_next"] = pd.cut(df_pairs["Q_next"], bins=bin_edges, labels=labels, include_lowest=True)

    matrix = pd.crosstab(df_pairs["state_t"], df_pairs["state_next"], normalize="index").reindex(index=labels, columns=labels, fill_value=0.0)

    fig, ax = plt.subplots(figsize=(8, 7))
    cax = ax.matshow(matrix.values, cmap="YlGnBu", vmin=0.0, vmax=1.0)

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = matrix.iloc[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontweight="bold", fontsize=10)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="left", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    cbar = fig.colorbar(cax, ax=ax, pad=0.04)
    cbar.set_label("Transition Probability P(t+1 | t)", fontsize=10, fontweight="bold")

    ax.set_xlabel("State at t+1", fontsize=10, fontweight="bold")
    ax.set_ylabel("State at t", fontsize=10, fontweight="bold")
    ax.set_title("MOD-05: Discrete State Transition Matrix", fontsize=12, fontweight="bold", pad=40)

    fig.tight_layout()
    return fig


def run_mod06(df, date_col, value_col):
    """MOD-06: Absolute Differential Streamflow Hydrograph (|dQ/dt|)."""
    df_clean = prepare_pairs_and_dowy(df, date_col, value_col)
    df_pairs = df_clean[df_clean["valid_pair"]].copy()
    df_pairs["abs_diff"] = np.abs(df_pairs["Q_next"] - df_pairs["Q_t"])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_pairs[date_col], df_pairs["abs_diff"], color="#000000", lw=0.9, zorder=3)

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}" if y < 1 else f"{int(y):,}"))

    apply_standard_grid(ax)
    ax.set_xlabel("Date", fontsize=10, fontweight="bold")
    ax.set_ylabel("|dQ/dt| (cfs/day)", fontsize=10, fontweight="bold")
    ax.set_title("MOD-06: Absolute Differential Rate-of-Change Hydrograph", fontsize=12, fontweight="bold", pad=12)

    fig.tight_layout()
    return fig


def run_mod07(df, date_col, value_col):
    """MOD-07: Chronological Raster Hydrograph."""
    df_clean = prepare_pairs_and_dowy(df, date_col, value_col)
    pivot = df_clean.pivot(index="WaterYear", columns="Standard_DOWY", values=value_col)
    pivot = pivot.reindex(columns=range(1, 367)).sort_index(ascending=True)

    fig, ax = plt.subplots(figsize=(12, 8))

    vmin = 10 ** np.floor(np.log10(np.maximum(0.01, df_clean[value_col].min())))
    vmax = 10 ** np.ceil(np.log10(df_clean[value_col].max()))
    norm = LogNorm(vmin=vmin, vmax=vmax)

    years = pivot.index.values

    # Pass untransformed matrix with LogNorm and reversed USGS colormap
    mesh = ax.imshow(
        np.maximum(0.001, pivot.values),
        aspect="auto",
        cmap=usgs_cmap,
        norm=norm,
        origin="lower",
        extent=[0.5, 366.5, -0.5, len(years) - 0.5]
    )

    # Colorbar with base-10 standard integer/decimal notation
    cbar = fig.colorbar(mesh, ax=ax, pad=0.03)
    cbar.set_label(f"Discharge ({value_col}) [cfs]", fontsize=10, fontweight="bold")
    cbar.ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
    cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}" if y < 1 else f"{int(y):,}"))

    # Dynamic leap-year month dividers
    draw_leap_adjusted_dividers(ax, years, is_ranked=False)

    # Y-axis Water Year ticks (5-year increments)
    y_indices = np.arange(len(years))
    major_mask = [(y % 5 == 0) for y in years]
    ax.set_yticks(y_indices[major_mask])
    ax.set_yticklabels(years[major_mask], fontsize=9)
    ax.set_yticks(y_indices, minor=True)
    ax.grid(True, which="major", axis="y", color="#000000", linestyle="-", lw=0.8, zorder=2)
    ax.grid(False, which="minor", axis="y")

    # X-axis Day of Water Year
    ax.set_xlim(0.5, 366.5)
    ax.set_xticks(range(30, 366, 30))
    ax.set_xticks(range(10, 366, 10), minor=True)
    ax.set_xlabel("Day of Water Year", fontsize=10, fontweight="bold")
    ax.set_ylabel("Water Year (Chronological)", fontsize=10, fontweight="bold")

    # Top Month Labels
    month_centers = [16.0, 46.5, 77.0, 108.0, 138.0, 168.0, 198.5, 229.0, 259.5, 290.0, 321.0, 351.5]
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(month_centers)
    ax_top.set_xticklabels(MONTH_LABELS, fontsize=9, fontweight="bold")
    ax_top.tick_params(length=0, pad=4)

    ax.set_title("MOD-07: Chronological Raster Hydrograph", fontsize=12, fontweight="bold", pad=32)
    fig.tight_layout()
    return fig

def run_mod08(df, date_col, value_col):
    """MOD-08: Annual Flow Volume Ranked Raster Hydrograph."""
    df_clean = prepare_pairs_and_dowy(df, date_col, value_col)

    # 1. Calculate cumulative annual flow volume per water year
    wy_totals = df_clean.groupby("WaterYear")[value_col].sum()

    # Rank: 1 = Wettest (largest volume) descending to Driest (smallest volume)
    # Ranked array from Wettest (Rank 1) to Driest (Rank N)
    ranked_wy_descending = wy_totals.sort_values(ascending=False).index.values
    n_years = len(ranked_wy_descending)

    # Reindex pivot matrix: top row (index 0 in imshow default lower-origin) needs to be Rank 1
    # When origin='upper', row 0 is at the top of the canvas.
    pivot = df_clean.pivot(index="WaterYear", columns="Standard_DOWY", values=value_col)
    pivot = pivot.reindex(index=ranked_wy_descending, columns=range(1, 367))

    fig, ax = plt.subplots(figsize=(12, 8))

    vmin = 10 ** np.floor(np.log10(np.maximum(0.01, df_clean[value_col].min())))
    vmax = 10 ** np.ceil(np.log10(df_clean[value_col].max()))
    norm = LogNorm(vmin=vmin, vmax=vmax)

    # 2. Render raster with origin='upper' so Rank 1 sits at the top (y=1)
    # Extent: X from DOWY 0.5 to 366.5; Y from Rank 0.5 (top) to n_years + 0.5 (bottom)
    mesh = ax.imshow(
        np.maximum(0.001, pivot.values),
        aspect="auto",
        cmap=usgs_cmap,
        norm=norm,
        origin="upper",
        extent=[0.5, 366.5, n_years + 0.5, 0.5]  # Inverts Y so 1 is at top, N at bottom
    )

    # 3. Exterior Colorbar with Base-10 Standard Notation
    cbar = fig.colorbar(mesh, ax=ax, pad=0.03)
    cbar.set_label(f"Discharge ({value_col}) [cfs]", fontsize=10, fontweight="bold")
    cbar.ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
    cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}" if y < 1 else f"{int(y):,}"))

    # 4. Dynamic leap-adjusted dividers mapped row-by-row to ranked order
    # row 0 (top) corresponds to ranked_wy_descending[0]
    for idx, wy in enumerate(ranked_wy_descending):
        rank = idx + 1
        y_top = rank - 0.5
        y_bottom = rank + 0.5
        is_leap = calendar.isleap(int(wy))
        offset = 1.0 if is_leap else 0.0

        # Pre-March dividers (constant across all years)
        for x in [31.5, 61.5, 92.5, 123.5]:
            ax.plot([x, x], [y_top, y_bottom], color="#000000", linestyle="--", lw=0.75, zorder=3)

        # Post-February dividers (+1 day step on leap years)
        for base_x in [151.5, 182.5, 212.5, 243.5, 273.5, 304.5, 335.5]:
            x = base_x + offset
            ax.plot([x, x], [y_top, y_bottom], color="#000000", linestyle="--", lw=0.75, zorder=3)

    # 5. Y-Axis: Integer Ranks (Rank 1 at top, multiples of 10, minor ticks at every integer rank)
    major_ranks = [1] + [r for r in range(10, n_years + 1, 10)]
    ax.set_yticks(major_ranks)
    ax.set_yticklabels(major_ranks, fontsize=9)
    ax.set_yticks(range(1, n_years + 1), minor=True)

    ax.set_ylim(n_years + 0.5, 0.5)  # Enforce Rank 1 at top, N at bottom
    ax.grid(True, which="major", axis="y", color="#000000", linestyle="-", lw=0.8, zorder=2)
    ax.grid(False, which="minor", axis="y")

    # 6. X-Axis: Standardized DOWY and Top Month Labels
    ax.set_xlim(0.5, 366.5)
    ax.set_xticks(range(50, 366, 50))
    ax.set_xticks(range(10, 366, 10), minor=True)
    ax.set_xlabel("Day of Water Year (Starting October 1)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Volumetric Rank (1 = Greatest Annual Volume)", fontsize=10, fontweight="bold")

    month_centers = [16.0, 46.5, 77.0, 108.0, 138.0, 168.0, 198.5, 229.0, 259.5, 290.0, 321.0, 351.5]
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(month_centers)
    ax_top.set_xticklabels(MONTH_LABELS, fontsize=9, fontweight="bold")
    ax_top.tick_params(length=0, pad=4)

    ax.set_title("MOD-08: Annual Flow Volume Ranked Raster Hydrograph", fontsize=12, fontweight="bold", pad=32)
    fig.tight_layout()
    return fig


def run_mod09(df, date_col, value_col):
    """MOD-09: Log10 Rate-of-Change vs Flow Scatter Plot."""
    df_clean = prepare_pairs_and_dowy(df, date_col, value_col)
    df_pairs = df_clean[df_clean["valid_pair"]].copy()

    df_pairs["diff"] = df_pairs["Q_next"] - df_pairs["Q_t"]
    df_pairs["abs_diff"] = np.abs(df_pairs["diff"])
    df_pairs = df_pairs[df_pairs["abs_diff"] > 0]

    fig, ax = plt.subplots(figsize=(9, 7))

    rising = df_pairs[df_pairs["diff"] > 0]
    falling = df_pairs[df_pairs["diff"] < 0]

    ax.scatter(rising["Q_t"], rising["abs_diff"], color="#1f77b4", s=12, alpha=0.5, label="Rising Limb (+dQ/dt)", zorder=3)
    ax.scatter(falling["Q_t"], falling["abs_diff"], color="#d62728", s=12, alpha=0.5, label="Falling Limb (-dQ/dt)", zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")

    for a in [ax.xaxis, ax.yaxis]:
        a.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
        a.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}" if y < 1 else f"{int(y):,}"))

    apply_standard_grid(ax)
    ax.set_xlabel(f"Q(t) ({value_col}) [cfs]", fontsize=10, fontweight="bold")
    ax.set_ylabel("|dQ/dt| (cfs/day)", fontsize=10, fontweight="bold")
    ax.set_title("MOD-09: Rate-of-Change vs Streamflow Scatter", fontsize=12, fontweight="bold", pad=28)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2, frameon=False, fontsize=9)

    fig.tight_layout()
    return fig


def run_mod10(df, date_col, value_col):
    """MOD-10: Exceedance Probability Duration Matrix."""
    df_clean = prepare_pairs_and_dowy(df, date_col, value_col)
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    thresholds = np.percentile(df_clean[value_col], percentiles)

    matrix = np.zeros((len(percentiles), 366))
    for p_idx, thresh in enumerate(thresholds):
        df_clean["exceed"] = (df_clean[value_col] >= thresh).astype(int)
        grouped = df_clean.groupby("Standard_DOWY")["exceed"].mean()
        for d in range(1, 367):
            matrix[p_idx, d - 1] = grouped.get(d, 0.0) * 100.0

    fig, ax = plt.subplots(figsize=(12, 6))
    mesh = ax.imshow(matrix, aspect="auto", cmap="Blues", origin="lower", extent=[0.5, 366.5, -0.5, len(percentiles) - 0.5])

    cbar = fig.colorbar(mesh, ax=ax, pad=0.03)
    cbar.set_label("Empirical Frequency of Exceedance (%)", fontsize=10, fontweight="bold")

    # Pre-March and Post-March vertical dividers
    for x in [31.5, 61.5, 92.5, 123.5, 152.5, 183.5, 213.5, 244.5, 274.5, 305.5, 336.5]:
        ax.axvline(x, color="#000000", linestyle="--", lw=0.75, zorder=3)

    ax.set_yticks(range(len(percentiles)))
    ax.set_yticklabels([f"P{p}" for p in percentiles], fontsize=9)

    ax.set_xlim(0.5, 366.5)
    ax.set_xticks(range(50, 366, 50))
    ax.set_xlabel("Day of Water Year (Starting October 1)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Discharge Quantile Threshold", fontsize=10, fontweight="bold")

    month_centers = [16.0, 46.5, 77.0, 108.0, 138.0, 168.0, 198.5, 229.0, 259.5, 290.0, 321.0, 351.5]
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(month_centers)
    ax_top.set_xticklabels(MONTH_LABELS, fontsize=9, fontweight="bold")
    ax_top.tick_params(length=0, pad=4)

    ax.set_title("MOD-10: Exceedance Probability Duration Matrix", fontsize=12, fontweight="bold", pad=32)
    fig.tight_layout()
    return fig


def run_mod11(df, date_col, value_col):
    """MOD-11: Discrete Transition Persistence Heatmap."""
    df_clean = prepare_pairs_and_dowy(df, date_col, value_col)
    df_pairs = df_clean[df_clean["valid_pair"]].copy()

    # 10 Decile states
    deciles = np.quantile(df_clean[value_col], np.linspace(0, 1, 11))
    labels = [f"D{i+1}" for i in range(10)]

    df_pairs["state_t"] = pd.cut(df_pairs["Q_t"], bins=deciles, labels=labels, include_lowest=True)
    df_pairs["state_next"] = pd.cut(df_pairs["Q_next"], bins=deciles, labels=labels, include_lowest=True)

    matrix = pd.crosstab(df_pairs["state_t"], df_pairs["state_next"], normalize="index").reindex(index=labels, columns=labels, fill_value=0.0)

    fig, ax = plt.subplots(figsize=(8, 7))
    cax = ax.matshow(matrix.values, cmap="YlOrRd", vmin=0.0, vmax=1.0)

    for i in range(10):
        for j in range(10):
            val = matrix.iloc[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8, fontweight="bold")

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    cbar = fig.colorbar(cax, ax=ax, pad=0.04)
    cbar.set_label("Transition Probability P(t+1 | t)", fontsize=10, fontweight="bold")

    ax.set_xlabel("State Decile at t+1", fontsize=10, fontweight="bold")
    ax.set_ylabel("State Decile at t", fontsize=10, fontweight="bold")
    ax.set_title("MOD-11: Decile Transition Persistence Matrix", fontsize=12, fontweight="bold", pad=32)

    fig.tight_layout()
    return fig


def run_mod12(df, date_col, value_col):
    """MOD-12: Composite Quad-Panel Hydrological Dashboard."""
    df_clean = prepare_pairs_and_dowy(df, date_col, value_col)
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.22)

    # Panel A: eFDC
    ax_a = fig.add_subplot(gs[0, 0])
    n = len(df_clean)
    sorted_q = np.sort(df_clean[value_col].values)[::-1]
    prob = (np.arange(1, n + 1) / (n + 1)) * 100.0
    df_pairs = df_clean[df_clean["valid_pair"]].copy()
    df_pairs["prob_t"] = np.interp(df_pairs["Q_t"], sorted_q[::-1], prob[::-1])
    diff = df_pairs["Q_next"] - df_pairs["Q_t"]

    ax_a.scatter(df_pairs[diff > 0]["prob_t"], df_pairs[diff > 0]["Q_next"], color="#1f77b4", s=8, alpha=0.5, label="Rising", zorder=3)
    ax_a.scatter(df_pairs[diff < 0]["prob_t"], df_pairs[diff < 0]["Q_next"], color="#d62728", s=8, alpha=0.5, label="Falling", zorder=3)
    ax_a.plot(prob, sorted_q, color="#000000", lw=1.8, label="FDC", zorder=4)
    ax_a.set_yscale("log")
    ax_a.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=8))
    ax_a.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}" if y < 1 else f"{int(y):,}"))
    apply_standard_grid(ax_a)
    ax_a.set_title("Panel A: Enhanced Flow Duration Curve", fontsize=11, fontweight="bold")
    ax_a.set_xlabel("Exceedance Probability (%)", fontsize=9, fontweight="bold")
    ax_a.set_ylabel("Discharge [cfs]", fontsize=9, fontweight="bold")

    # Panel B: Lag-1
    ax_b = fig.add_subplot(gs[0, 1])
    min_v, max_v = df_pairs["Q_t"].min(), df_pairs["Q_t"].max()
    log_min = 10 ** np.floor(np.log10(min_v))
    log_max = 10 ** np.ceil(np.log10(max_v))
    ax_b.scatter(df_pairs[diff > 0]["Q_t"], df_pairs[diff > 0]["Q_next"], color="#1f77b4", s=8, alpha=0.5, zorder=3)
    ax_b.scatter(df_pairs[diff < 0]["Q_t"], df_pairs[diff < 0]["Q_next"], color="#d62728", s=8, alpha=0.5, zorder=3)
    ax_b.plot([log_min, log_max], [log_min, log_max], color="#000000", lw=1.2, zorder=4)
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlim(log_min, log_max)
    ax_b.set_ylim(log_min, log_max)
    for a in [ax_b.xaxis, ax_b.yaxis]:
        a.set_major_locator(ticker.LogLocator(base=10.0, numticks=8))
        a.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}" if y < 1 else f"{int(y):,}"))
    apply_standard_grid(ax_b)
    ax_b.set_title("Panel B: Lag-1 Phase Space", fontsize=11, fontweight="bold")
    ax_b.set_xlabel("Q(t) [cfs]", fontsize=9, fontweight="bold")
    ax_b.set_ylabel("Q(t+1) [cfs]", fontsize=9, fontweight="bold")

    # Panel C: Raster Hydrograph
    ax_c = fig.add_subplot(gs[1, 0])
    pivot = df_clean.pivot(index="WaterYear", columns="Standard_DOWY", values=value_col)
    pivot = pivot.reindex(columns=range(1, 367)).sort_index(ascending=True)
    norm = LogNorm(vmin=log_min, vmax=log_max)
    years = pivot.index.values

    mesh_c = ax_c.imshow(np.maximum(0.001, pivot.values), aspect="auto", cmap=usgs_cmap_r, norm=norm, origin="lower", extent=[0.5, 366.5, -0.5, len(years) - 0.5])
    draw_leap_adjusted_dividers(ax_c, years, is_ranked=False)
    ax_c.set_xlim(0.5, 366.5)
    ax_c.set_xticks(range(50, 366, 50))
    ax_c.set_title("Panel C: Chronological Raster Hydrograph", fontsize=11, fontweight="bold")
    ax_c.set_xlabel("Day of Water Year", fontsize=9, fontweight="bold")
    ax_c.set_ylabel("Water Year", fontsize=9, fontweight="bold")

    # Panel D: Chronological Series
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.plot(df_clean[date_col], df_clean[value_col], color="#000000", lw=0.8, zorder=3)
    ax_d.set_yscale("log")
    ax_d.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=8))
    ax_d.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}" if y < 1 else f"{int(y):,}"))
    apply_standard_grid(ax_d)
    ax_d.set_title("Panel D: Chronological Hydrograph", fontsize=11, fontweight="bold")
    ax_d.set_xlabel("Date", fontsize=9, fontweight="bold")
    ax_d.set_ylabel("Discharge [cfs]", fontsize=9, fontweight="bold")

    fig.suptitle("MOD-12: Composite Quad-Panel Hydrological Dashboard", fontsize=14, fontweight="bold", y=0.98)
    return fig


MODULE_REGISTRY = {
    "MOD-01: Enhanced Flow Duration Curve (eFDC)": run_mod01,
    "MOD-02: Lag-1 Streamflow Scatter Plot": run_mod02,
    "MOD-03: Discrete Transition Matrix (Raw Frequency Counts)": run_mod03,
    "MOD-04: Conditional Forecast Probability Matrix": run_mod04,
    "MOD-05: Conditional Antecedent Probability Matrix": run_mod05,
    "MOD-06: Sequential Flow Duration & Persistence": run_mod06,
    "MOD-07: Chronological Raster Hydrograph": run_mod07,
    "MOD-08: Volumetric Raster Hydrograph": run_mod08,
    "MOD-09: Annual FDC Spaghetti Plot": run_mod09,
    "MOD-10: Annual FDC Threshold Trends": run_mod10,
    "MOD-11: Annual FDC Volumetric Thresholds": run_mod11,
    "MOD-12: Composite Hydroinformatics Dashboard": run_mod12,
}


# ---------------------------------------------------------
# Streamlit Web User Interface
# ---------------------------------------------------------
def main():
    st.sidebar.title("Hydro-Studio")
    st.sidebar.markdown("**Operational Modular Visual Analytics (v8.5)**")
    st.sidebar.caption("© Visual Data Analytics, LLC (2026.1)")
    st.sidebar.markdown("---")

    uploaded_file = st.sidebar.file_uploader(
        "Upload Streamflow CSV",
        type=["csv"],
        help="Upload standard CSV with chronological streamflow data."
    )

    if uploaded_file is None:
        st.info("👋 Welcome to Hydro-Studio. Please upload a streamflow CSV file in the sidebar to begin analysis.")
        return

    # Ingest CSV file
    try:
        df_raw = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error parsing CSV file: {e}")
        return

    cols = list(df_raw.columns)
    if len(cols) < 2:
        st.error("The CSV file must contain at least two columns: Date and Streamflow.")
        return

    st.sidebar.subheader("Dataset Mapping")
    default_date_idx = 0
    default_val_idx = 1 if len(cols) > 1 else 0

    # Auto-detect date column
    for idx, c in enumerate(cols):
        if "date" in str(c).lower() or "time" in str(c).lower():
            default_date_idx = idx
            break

    # Auto-detect flow column
    for idx, c in enumerate(cols):
        if any(term in str(c).lower() for term in ["flow", "discharge", "q", "00060"]):
            default_val_idx = idx
            break

    date_col = st.sidebar.selectbox("Date / Timestamp Column", options=cols, index=default_date_idx)
    value_col = st.sidebar.selectbox("Discharge Column (cfs)", options=cols, index=default_val_idx)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Module Selection")
    module_choice = st.sidebar.selectbox("Select Visual Module", options=list(MODULE_REGISTRY.keys()), index=6)

    st.sidebar.markdown("---")
    generate_btn = st.sidebar.button("🚀 Generate Plot", type="primary", use_container_width=True)

    st.subheader(module_choice)

    # Plot Execution on button press or persistent session state
    if generate_btn or "current_fig" in st.session_state:
        if generate_btn:
            with st.spinner("Processing time-series matrices and executing rendering..."):
                func = MODULE_REGISTRY[module_choice]
                fig = func(df_raw, date_col, value_col)
                st.session_state["current_fig"] = fig
                st.session_state["current_module"] = module_choice

        fig = st.session_state.get("current_fig")
        if fig is not None:
            st.pyplot(fig, clear_figure=False)

            # Export 300 DPI publication graphic
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
            buf.seek(0)

            mod_id = st.session_state.get("current_module", "Plot").split(":")[0].strip()
            st.download_button(
                label="💾 Download High-Resolution Plot (PNG - 300 DPI)",
                data=buf,
                file_name=f"{mod_id}_USGS_Publication.png",
                mime="image/png",
                use_container_width=True
            )


if __name__ == "__main__":
    main()
