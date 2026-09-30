"""
Hydro-Studio: Hydrological Visual Analytics & Graphics Engine
Standardized Operational Modules (MOD-01 through MOD-12)
Deterministic Baseline Generator & AI Code Exporter
"""

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from scipy import stats
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
    "mathtext.fontset": "dejavusans",
    "figure.dpi": 300,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
})

MONTH_LABELS = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]
MONTH_TICKS_366 = [1, 32, 62, 93, 124, 153, 184, 214, 245, 275, 306, 337]


# ---------------------------------------------------------
# Benchmark Synthetic Data Generator
# ---------------------------------------------------------
@st.cache_data
def get_benchmark_data():
    """Generates synthetic multi-year daily discharge with log-normal storm spikes and seasonality."""
    np.random.seed(42)
    dates = pd.date_range(start="2018-10-01", end="2025-09-30", freq="D")
    doy = dates.dayofyear
    
    seasonal_base = 50 + 40 * np.sin(2 * np.pi * (doy - 60) / 365.25)
    noise = np.random.lognormal(mean=0.0, sigma=0.85, size=len(dates))
    discharge = np.maximum(5.0, seasonal_base * noise)
    
    df = pd.DataFrame({"Date": dates, "Discharge_cfs": discharge})
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def prepare_water_year_data(df, date_col, value_col):
    """Calculates Water Year and fixed 366-day Day of Water Year (Feb 29 = Day 152)."""
    df_clean = df.dropna(subset=[date_col, value_col]).copy()
    df_clean["Date"] = pd.to_datetime(df_clean[date_col])
    
    month = df_clean["Date"].dt.month
    day = df_clean["Date"].dt.day
    year = df_clean["Date"].dt.year
    df_clean["WaterYear"] = np.where(month >= 10, year + 1, year)
    
    ref_dates = pd.date_range("2019-10-01", "2020-09-30", freq="D")
    dowy_lookup = {(d.month, d.day): idx + 1 for idx, d in enumerate(ref_dates)}
    
    df_clean["Standard_DOWY"] = [dowy_lookup.get((m, d), np.nan) for m, d in zip(month, day)]
    df_clean = df_clean.dropna(subset=["Standard_DOWY"])
    df_clean["Standard_DOWY"] = df_clean["Standard_DOWY"].astype(int)
    return df_clean


# ---------------------------------------------------------
# MOD-01: Flow Duration Curve (FDC)
# ---------------------------------------------------------
def mod01_fdc(df, value_col):
    q = df[value_col].dropna().values
    q_sorted = np.sort(q)[::-1]
    n = len(q_sorted)
    ranks = np.arange(1, n + 1)
    exceedance = (ranks / (n + 1.0)) * 100.0

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(exceedance, q_sorted, color="#1f77b4", lw=2, label="Empirical FDC")
    
    for p in [10, 50, 90]:
        val = np.percentile(q, 100 - p)
        ax.axvline(p, color="#888888", linestyle="--", lw=0.8, alpha=0.7)
        ax.plot(p, val, "o", color="#d62728", markersize=4)
        ax.annotate(f"Q{p}: {val:,.1f}", xy=(p, val), xytext=(p + 2, val * 1.15),
                    fontsize=8, ha="left", color="#333333")

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.set_xlim(0, 100)
    ax.set_xlabel("Exceedance Probability (%)", fontsize=10, fontweight="bold")
    ax.set_ylabel(f"Discharge ({value_col})", fontsize=10, fontweight="bold")
    ax.set_title("MOD-01: Flow Duration Curve (FDC)", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, which="major", color="#dddddd", linestyle="-", lw=0.7)
    ax.grid(True, which="minor", color="#f0f0f0", linestyle=":", lw=0.5)
    ax.legend(loc="upper right", frameon=True, facecolor="#ffffff", framealpha=0.9)
    plt.tight_layout()

    code = f'''# MOD-01: Standalone FDC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

q = df["{value_col}"].dropna().values
q_sorted = np.sort(q)[::-1]
n = len(q_sorted)
exceedance = (np.arange(1, n + 1) / (n + 1.0)) * 100.0

fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
ax.plot(exceedance, q_sorted, color="#1f77b4", lw=2, label="Empirical FDC")
for p in [10, 50, 90]:
    val = np.percentile(q, 100 - p)
    ax.axvline(p, color="#888888", linestyle="--", lw=0.8)
    ax.plot(p, val, "o", color="#d62728", markersize=4)
    ax.annotate(f"Q{{p}}: {{val:,.1f}}", xy=(p, val), xytext=(p + 2, val * 1.15), fontsize=8)

ax.set_yscale("log")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{{y:g}}"))
ax.set_xlim(0, 100)
ax.set_xlabel("Exceedance Probability (%)", fontweight="bold")
ax.set_ylabel("Discharge ({value_col})", fontweight="bold")
ax.set_title("MOD-01: Flow Duration Curve", fontweight="bold")
ax.grid(True, which="major", color="#dddddd")
ax.legend(loc="upper right")
plt.tight_layout()
plt.show()'''
    return fig, code


# ---------------------------------------------------------
# MOD-02: Lag-1 Rate-of-Change Scatterplot
# ---------------------------------------------------------
def mod02_lag1_scatter(df, value_col):
    df_clean = df.dropna(subset=[value_col]).copy()
    q_t = df_clean[value_col].iloc[:-1].values
    q_next = df_clean[value_col].iloc[1:].values
    delta_q = q_next - q_t
    is_rising = delta_q > 0

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(q_t[~is_rising], q_next[~is_rising], color="#3182bd", alpha=0.4, 
               s=16, edgecolors="none", label="Falling Limb (-dQ/dt)")
    ax.scatter(q_t[is_rising], q_next[is_rising], color="#de2d26", alpha=0.5, 
               s=18, edgecolors="none", label="Rising Limb (+dQ/dt)")

    min_val = max(0.1, min(np.min(q_t), np.min(q_next)))
    max_val = max(np.max(q_t), np.max(q_next)) * 1.2
    ax.plot([min_val, max_val], [min_val, max_val], color="#444444", lw=1.2, 
            linestyle="--", label="1:1 Equilibrium ($Q_{t+1} = Q_t$)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xlabel(r"Streamflow at Time $t$, $Q_t$ (cfs)", fontsize=10, fontweight="bold")
    ax.set_ylabel(r"Streamflow at Time $t+1$, $Q_{t+1}$ (cfs)", fontsize=10, fontweight="bold")
    ax.set_title("MOD-02: Lag-1 Streamflow Rate-of-Change", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, which="major", color="#dddddd", linestyle="-", lw=0.7)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, facecolor="#ffffff")
    plt.tight_layout()

    code = f'''# MOD-02: Standalone Lag-1 Scatter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

q = df["{value_col}"].dropna().values
q_t = q[:-1]
q_next = q[1:]
is_rising = (q_next - q_t) > 0

fig, ax = plt.subplots(figsize=(7, 7), dpi=300)
ax.scatter(q_t[~is_rising], q_next[~is_rising], color="#3182bd", alpha=0.4, s=16, label="Falling Limb (-dQ/dt)")
ax.scatter(q_t[is_rising], q_next[is_rising], color="#de2d26", alpha=0.5, s=18, label="Rising Limb (+dQ/dt)")

min_val = max(0.1, min(np.min(q_t), np.min(q_next)))
max_val = max(np.max(q_t), np.max(q_next)) * 1.2
ax.plot([min_val, max_val], [min_val, max_val], color="#444444", lw=1.2, linestyle="--", label="1:1 Line")
ax.set_xscale("log")
ax.set_yscale("log")
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{{x:g}}"))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{{y:g}}"))
ax.set_xlim(min_val, max_val)
ax.set_ylim(min_val, max_val)
ax.set_xlabel(r"Streamflow $Q_t$", fontweight="bold")
ax.set_ylabel(r"Streamflow $Q_{{t+1}}$", fontweight="bold")
ax.set_title("MOD-02: Lag-1 Rate-of-Change", fontweight="bold")
ax.grid(True, which="major", color="#dddddd")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()'''
    return fig, code


# ---------------------------------------------------------
# MOD-03: 2D Raster Hydrograph (Standardized 366-Day Alignment)
# ---------------------------------------------------------
def mod03_raster_hydrograph(df, date_col, value_col):
    df_clean = prepare_water_year_data(df, date_col, value_col)
    pivot = df_clean.pivot(index="WaterYear", columns="Standard_DOWY", values=value_col)
    pivot = pivot.reindex(columns=range(1, 367)).sort_index(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    log_data = np.log10(np.maximum(0.1, pivot.values))
    cmap = plt.cm.Spectral_r.copy()
    cmap.set_bad(color="#e0e0e0")

    mesh = ax.imshow(
        log_data, aspect="auto", cmap=cmap, origin="lower",
        extent=[1, 366, pivot.index.min() - 0.5, pivot.index.max() + 0.5]
    )
    cbar = plt.colorbar(mesh, ax=ax, pad=0.03)
    cbar.set_label(f"Discharge ({value_col}) [Log10 Scale]", fontsize=9, fontweight="bold")
    
    ax.set_xticks(MONTH_TICKS_366)
    ax.set_xticklabels(MONTH_LABELS, fontsize=9)
    ax.set_xlim(1, 366)
    ax.set_xlabel("Day of Water Year (Leap-Adjusted)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Water Year", fontsize=10, fontweight="bold")
    ax.set_title("MOD-03: 2D Raster Hydrograph (Standardized 366-Day Matrix)", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()

    code = f'''# MOD-03: Standalone Raster Hydrograph with Leap-Year Alignment
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df["Date"] = pd.to_datetime(df["{date_col}"])
month, day, year = df["Date"].dt.month, df["Date"].dt.day, df["Date"].dt.year
df["WaterYear"] = np.where(month >= 10, year + 1, year)

ref_dates = pd.date_range("2019-10-01", "2020-09-30", freq="D")
dowy_lookup = {{(d.month, d.day): idx + 1 for idx, d in enumerate(ref_dates)}}
df["Standard_DOWY"] = [dowy_lookup.get((m, d)) for m, d in zip(month, day)]

pivot = df.pivot(index="WaterYear", columns="Standard_DOWY", values="{value_col}").reindex(columns=range(1, 367)).sort_index(ascending=True)

fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
cmap = plt.cm.Spectral_r.copy()
cmap.set_bad(color="#e0e0e0")
mesh = ax.imshow(np.log10(np.maximum(0.1, pivot.values)), aspect="auto", cmap=cmap, origin="lower",
                 extent=[1, 366, pivot.index.min() - 0.5, pivot.index.max() + 0.5])
cbar = plt.colorbar(mesh, ax=ax, pad=0.03)
cbar.set_label("Discharge [Log10]", fontweight="bold")
ax.set_xticks({MONTH_TICKS_366})
ax.set_xticklabels({MONTH_LABELS})
ax.set_xlim(1, 366)
ax.set_xlabel("Day of Water Year", fontweight="bold")
ax.set_ylabel("Water Year", fontweight="bold")
ax.set_title("MOD-03: 2D Raster Hydrograph", fontweight="bold")
plt.tight_layout()
plt.show()'''
    return fig, code


# ---------------------------------------------------------
# MOD-04: Enhanced Flow Duration Curve (eFDC)
# ---------------------------------------------------------
def mod04_efdc(df, value_col):
    df_clean = df.dropna(subset=[value_col]).copy()
    q = df_clean[value_col].values
    q_t = q[:-1]
    q_next = q[1:]
    is_rising = (q_next - q_t) > 0
    
    q_rising = np.sort(q_t[is_rising])[::-1]
    q_falling = np.sort(q_t[~is_rising])[::-1]
    q_total = np.sort(q)[::-1]
    
    p_rising = (np.arange(1, len(q_rising) + 1) / (len(q_rising) + 1.0)) * 100.0
    p_falling = (np.arange(1, len(q_falling) + 1) / (len(q_falling) + 1.0)) * 100.0
    p_total = (np.arange(1, len(q_total) + 1) / (len(q_total) + 1.0)) * 100.0

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(p_total, q_total, color="#333333", lw=1.8, linestyle="--", label="Total FDC")
    ax.plot(p_rising, q_rising, color="#de2d26", lw=2, label="Rising Limb FDC (+dQ/dt)")
    ax.plot(p_falling, q_falling, color="#3182bd", lw=2, label="Falling Limb FDC (-dQ/dt)")

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.set_xlim(0, 100)
    ax.set_xlabel("Exceedance Probability (%)", fontsize=10, fontweight="bold")
    ax.set_ylabel(f"Discharge ({value_col})", fontsize=10, fontweight="bold")
    ax.set_title("MOD-04: Enhanced Flow Duration Curve (eFDC Limb Partitioning)", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, which="major", color="#dddddd", linestyle="-", lw=0.7)
    ax.legend(loc="upper right", frameon=True, facecolor="#ffffff")
    plt.tight_layout()

    code = f'''# MOD-04: Standalone eFDC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

q = df["{value_col}"].dropna().values
is_rising = (q[1:] - q[:-1]) > 0
q_rising, q_falling, q_total = np.sort(q[:-1][is_rising])[::-1], np.sort(q[:-1][~is_rising])[::-1], np.sort(q)[::-1]

fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
ax.plot((np.arange(1, len(q_total) + 1)/(len(q_total)+1))*100, q_total, color="#333333", lw=1.8, linestyle="--", label="Total FDC")
ax.plot((np.arange(1, len(q_rising) + 1)/(len(q_rising)+1))*100, q_rising, color="#de2d26", lw=2, label="Rising Limb FDC")
ax.plot((np.arange(1, len(q_falling) + 1)/(len(q_falling)+1))*100, q_falling, color="#3182bd", lw=2, label="Falling Limb FDC")

ax.set_yscale("log")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{{y:g}}"))
ax.set_xlim(0, 100)
ax.set_xlabel("Exceedance Probability (%)", fontweight="bold")
ax.set_ylabel("Discharge ({value_col})", fontweight="bold")
ax.set_title("MOD-04: Enhanced Flow Duration Curve (eFDC)", fontweight="bold")
ax.grid(True, which="major", color="#dddddd")
ax.legend(loc="upper right")
plt.tight_layout()
plt.show()'''
    return fig, code


# ---------------------------------------------------------
# MOD-05: Discrete State Transition Matrix Heatmap
# ---------------------------------------------------------
def mod05_transition_matrix(df, value_col):
    q = df[value_col].dropna().values
    bins = np.percentile(q, [0, 20, 40, 60, 80, 100])
    bin_labels = ["Very Low (Q80-100)", "Low (Q60-80)", "Moderate (Q40-60)", "High (Q20-40)", "Very High (Q0-20)"]
    
    states = pd.cut(q, bins=bins, labels=range(5), include_lowest=True)
    state_t = states[:-1]
    state_next = states[1:]
    
    matrix = np.zeros((5, 5))
    for st_now, st_nxt in zip(state_t, state_next):
        if pd.notna(st_now) and pd.notna(st_nxt):
            matrix[st_now, st_nxt] += 1
            
    row_sums = matrix.sum(axis=1, keepdims=True)
    prob_matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    mesh = ax.imshow(prob_matrix, cmap="Blues", origin="lower", vmin=0, vmax=1)
    cbar = plt.colorbar(mesh, ax=ax, pad=0.03)
    cbar.set_label("Transition Probability $P(S_{t+1} \vert{} S_t)$", fontsize=9, fontweight="bold")

    for i in range(5):
        for j in range(5):
            val = prob_matrix[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9, fontweight="bold")

    ax.set_xticks(range(5))
    ax.set_xticklabels(bin_labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(5))
    ax.set_yticklabels(bin_labels, fontsize=8)
    ax.set_xlabel(r"State at Time $t+1$ ($S_{t+1}$)", fontsize=10, fontweight="bold")
    ax.set_ylabel(r"State at Time $t$ ($S_t$)", fontsize=10, fontweight="bold")
    ax.set_title("MOD-05: Discrete State Transition Matrix", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()

    code = f'''# MOD-05: Standalone Transition Matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

q = df["{value_col}"].dropna().values
bins = np.percentile(q, [0, 20, 40, 60, 80, 100])
labels = ["V.Low", "Low", "Mod", "High", "V.High"]
states = pd.cut(q, bins=bins, labels=range(5), include_lowest=True)

matrix = np.zeros((5, 5))
for s_now, s_nxt in zip(states[:-1], states[1:]):
    if pd.notna(s_now) and pd.notna(s_nxt):
        matrix[s_now, s_nxt] += 1
row_sums = matrix.sum(axis=1, keepdims=True)
probs = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)

fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
mesh = ax.imshow(probs, cmap="Blues", origin="lower", vmin=0, vmax=1)
plt.colorbar(mesh, ax=ax, label="Transition Probability")
ax.set_xticks(range(5)); ax.set_xticklabels(labels)
ax.set_yticks(range(5)); ax.set_yticklabels(labels)
ax.set_xlabel(r"State at $t+1$", fontweight="bold")
ax.set_ylabel(r"State at $t$", fontweight="bold")
ax.set_title("MOD-05: State Transition Matrix", fontweight="bold")
plt.tight_layout()
plt.show()'''
    return fig, code


# ---------------------------------------------------------
# MOD-06: Annual Regime Carpet Plot (Ranked by June Volume)
# ---------------------------------------------------------
def mod06_ranked_carpet(df, date_col, value_col):
    df_clean = prepare_water_year_data(df, date_col, value_col)
    
    june_flows = df_clean[df_clean["Date"].dt.month == 6].groupby("WaterYear")[value_col].sum()
    ranked_wy = june_flows.sort_values(ascending=True).index
    
    pivot = df_clean.pivot(index="WaterYear", columns="Standard_DOWY", values=value_col)
    pivot = pivot.reindex(index=ranked_wy, columns=range(1, 367))

    fig, ax = plt.subplots(figsize=(10, 5))
    log_data = np.log10(np.maximum(0.1, pivot.values))
    cmap = plt.cm.Spectral_r.copy()
    cmap.set_bad(color="#e0e0e0")

    mesh = ax.imshow(
        log_data, aspect="auto", cmap=cmap, origin="lower",
        extent=[1, 366, 0.5, len(ranked_wy) + 0.5]
    )
    cbar = plt.colorbar(mesh, ax=ax, pad=0.03)
    cbar.set_label(f"Discharge ({value_col}) [Log10 Scale]", fontsize=9, fontweight="bold")
    
    ax.set_xticks(MONTH_TICKS_366)
    ax.set_xticklabels(MONTH_LABELS, fontsize=9)
    ax.set_xlim(1, 366)
    ax.set_yticks(range(1, len(ranked_wy) + 1))
    ax.set_yticklabels([f"WY {int(y)}" for y in ranked_wy], fontsize=8)
    ax.set_xlabel("Day of Water Year (Leap-Adjusted)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Water Year (Ranked by June Total Volume)", fontsize=10, fontweight="bold")
    ax.set_title("MOD-06: Annual Regime Carpet (Ranked by June Volume)", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()

    code = f'''# MOD-06: Standalone Ranked Carpet Hydrograph
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df["Date"] = pd.to_datetime(df["{date_col}"])
month, day, year = df["Date"].dt.month, df["Date"].dt.day, df["Date"].dt.year
df["WaterYear"] = np.where(month >= 10, year + 1, year)

ref_dates = pd.date_range("2019-10-01", "2020-09-30", freq="D")
dowy_lookup = {{(d.month, d.day): idx + 1 for idx, d in enumerate(ref_dates)}}
df["Standard_DOWY"] = [dowy_lookup.get((m, d)) for m, d in zip(month, day)]

june_rank = df[df["Date"].dt.month == 6].groupby("WaterYear")["{value_col}"].sum().sort_values(ascending=True).index
pivot = df.pivot(index="WaterYear", columns="Standard_DOWY", values="{value_col}").reindex(index=june_rank, columns=range(1, 367))

fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
cmap = plt.cm.Spectral_r.copy()
cmap.set_bad(color="#e0e0e0")
mesh = ax.imshow(np.log10(np.maximum(0.1, pivot.values)), aspect="auto", cmap=cmap, origin="lower",
                 extent=[1, 366, 0.5, len(june_rank) + 0.5])
plt.colorbar(mesh, ax=ax, label="Discharge [Log10]")
ax.set_xticks({MONTH_TICKS_366})
ax.set_xticklabels({MONTH_LABELS})
ax.set_yticks(range(1, len(june_rank) + 1))
ax.set_yticklabels([f"WY {{int(y)}}" for y in june_rank])
ax.set_xlabel("Day of Water Year", fontweight="bold")
ax.set_ylabel("Water Year (Ranked by June Volume)", fontweight="bold")
ax.set_title("MOD-06: Annual Regime Carpet", fontweight="bold")
plt.tight_layout()
plt.show()'''
    return fig, code


# ---------------------------------------------------------
# MOD-07: Comparative Dual-FDC Overlay (Temporal Split)
# ---------------------------------------------------------
def mod07_comparative_fdc(df, date_col, value_col):
    df_clean = df.dropna(subset=[date_col, value_col]).copy()
    df_clean["Date"] = pd.to_datetime(df_clean[date_col])
    median_date = df_clean["Date"].iloc[len(df_clean) // 2]

    era1 = df_clean[df_clean["Date"] <= median_date][value_col].dropna().values
    era2 = df_clean[df_clean["Date"] > median_date][value_col].dropna().values

    q1_sorted = np.sort(era1)[::-1]
    q2_sorted = np.sort(era2)[::-1]
    p1 = (np.arange(1, len(q1_sorted) + 1) / (len(q1_sorted) + 1.0)) * 100.0
    p2 = (np.arange(1, len(q2_sorted) + 1) / (len(q2_sorted) + 1.0)) * 100.0

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(p1, q1_sorted, color="#2b83ba", lw=2, label=f"Era 1 (Pre-{median_date.year})")
    ax.plot(p2, q2_sorted, color="#d7191c", lw=2, label=f"Era 2 (Post-{median_date.year})")

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.set_xlim(0, 100)
    ax.set_xlabel("Exceedance Probability (%)", fontsize=10, fontweight="bold")
    ax.set_ylabel(f"Discharge ({value_col})", fontsize=10, fontweight="bold")
    ax.set_title("MOD-07: Comparative Dual-FDC (Regime Shift Evaluation)", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, which="major", color="#dddddd", linestyle="-", lw=0.7)
    ax.legend(loc="upper right", frameon=True, facecolor="#ffffff")
    plt.tight_layout()

    code = f'''# MOD-07: Standalone Comparative FDC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

df["Date"] = pd.to_datetime(df["{date_col}"])
split = df["Date"].iloc[len(df)//2]
e1 = np.sort(df[df["Date"] <= split]["{value_col}"].dropna().values)[::-1]
e2 = np.sort(df[df["Date"] > split]["{value_col}"].dropna().values)[::-1]

fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
ax.plot((np.arange(1, len(e1)+1)/(len(e1)+1))*100, e1, color="#2b83ba", lw=2, label=f"Era 1 (Pre-{{split.year}})")
ax.plot((np.arange(1, len(e2)+1)/(len(e2)+1))*100, e2, color="#d7191c", lw=2, label=f"Era 2 (Post-{{split.year}})")
ax.set_yscale("log")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{{y:g}}"))
ax.set_xlim(0, 100)
ax.set_xlabel("Exceedance Probability (%)", fontweight="bold")
ax.set_ylabel("Discharge ({value_col})", fontweight="bold")
ax.set_title("MOD-07: Comparative Dual-FDC", fontweight="bold")
ax.grid(True)
ax.legend(loc="upper right")
plt.tight_layout()
plt.show()'''
    return fig, code


# ---------------------------------------------------------
# MOD-08: Flashiness Index & Hydrograph Recession Rate
# ---------------------------------------------------------
def mod08_recession_rate(df, value_col):
    df_clean = df.dropna(subset=[value_col]).copy()
    q = df_clean[value_col].values
    q_t = q[:-1]
    q_next = q[1:]
    
    falling_idx = (q_next - q_t) < 0
    q_fall_t = q_t[falling_idx]
    dq_dt = (q_next[falling_idx] - q_fall_t)
    neg_dq_dt = -dq_dt

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(q_fall_t, neg_dq_dt, color="#2c7bb6", alpha=0.4, s=15, edgecolors="none")

    valid = (q_fall_t > 0) & (neg_dq_dt > 0)
    log_q = np.log10(q_fall_t[valid])
    log_dq = np.log10(neg_dq_dt[valid])
    slope, intercept, r_value, _, _ = stats.linregress(log_q, log_dq)
    
    x_fit = np.logspace(np.log10(np.min(q_fall_t[valid])), np.log10(np.max(q_fall_t[valid])), 50)
    y_fit = (10 ** intercept) * (x_fit ** slope)
    ax.plot(x_fit, y_fit, color="#d7191c", lw=2, 
            label=f"Recession Fit: $-dQ/dt = {10**intercept:.2f} Q^{{{slope:.2f}}}$ ($R^2={r_value**2:.2f}$)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.set_xlabel(r"Streamflow $Q$ (cfs)", fontsize=10, fontweight="bold")
    ax.set_ylabel(r"Recession Rate $-dQ/dt$ (cfs/day)", fontsize=10, fontweight="bold")
    ax.set_title("MOD-08: Hydrograph Recession Analysis (Brutsaert-Nieber)", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, which="major", color="#dddddd", linestyle="-", lw=0.7)
    ax.legend(loc="lower right", frameon=True, facecolor="#ffffff")
    plt.tight_layout()

    code = f'''# MOD-08: Standalone Recession Rate Plot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats

q = df["{value_col}"].dropna().values
falling = (q[1:] - q[:-1]) < 0
q_t, neg_dq = q[:-1][falling], -(q[1:][falling] - q[:-1][falling])
valid = (q_t > 0) & (neg_dq > 0)

slope, intercept, r_val, _, _ = stats.linregress(np.log10(q_t[valid]), np.log10(neg_dq[valid]))
x_fit = np.logspace(np.log10(np.min(q_t[valid])), np.log10(np.max(q_t[valid])), 50)
y_fit = (10 ** intercept) * (x_fit ** slope)

fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
ax.scatter(q_t, neg_dq, color="#2c7bb6", alpha=0.4, s=15)
ax.plot(x_fit, y_fit, color="#d7191c", lw=2, label=f"-dQ/dt = {{10**intercept:.2f}} Q^{{slope:.2f}}")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Streamflow Q", fontweight="bold")
ax.set_ylabel("-dQ/dt", fontweight="bold")
ax.set_title("MOD-08: Hydrograph Recession", fontweight="bold")
ax.grid(True)
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()'''
    return fig, code


# ---------------------------------------------------------
# MOD-09: Log-Pearson Type III Flood Frequency Curve
# ---------------------------------------------------------
def mod09_flood_frequency(df, date_col, value_col):
    df_clean = prepare_water_year_data(df, date_col, value_col)
    annual_max = df_clean.groupby("WaterYear")[value_col].max().dropna()
    n = len(annual_max)
    
    sorted_peaks = np.sort(annual_max.values)[::-1]
    return_periods = (n + 1.0) / np.arange(1, n + 1)
    
    log_peaks = np.log10(sorted_peaks)
    mean_lp = np.mean(log_peaks)
    std_lp = np.std(log_peaks, ddof=1)
    skew_lp = stats.skew(log_peaks)

    tr_theor = np.linspace(1.05, 100, 100)
    p_theor = 1.0 - (1.0 / tr_theor)
    quantiles = stats.pearson3.ppf(p_theor, skew=skew_lp, loc=mean_lp, scale=std_lp)
    q_theor = 10 ** quantiles

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(return_periods, sorted_peaks, color="#d7191c", s=25, label="Empirical Annual Maxima", zorder=3)
    ax.plot(tr_theor, q_theor, color="#2b83ba", lw=2, label=f"Log-Pearson III Fit (Skew={skew_lp:.2f})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.set_xlabel("Return Period $T_r$ (Years)", fontsize=10, fontweight="bold")
    ax.set_ylabel(f"Peak Discharge ({value_col})", fontsize=10, fontweight="bold")
    ax.set_title("MOD-09: Flood Frequency Analysis (LP-III Fit)", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, which="major", color="#dddddd", linestyle="-", lw=0.7)
    ax.grid(True, which="minor", color="#f0f0f0", linestyle=":", lw=0.5)
    ax.legend(loc="lower right", frameon=True, facecolor="#ffffff")
    plt.tight_layout()

    code = f'''# MOD-09: Standalone Flood Frequency Analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats

df["Date"] = pd.to_datetime(df["{date_col}"])
df["WaterYear"] = np.where(df["Date"].dt.month >= 10, df["Date"].dt.year + 1, df["Date"].dt.year)
peaks = df.groupby("WaterYear")["{value_col}"].max().dropna().sort_values(ascending=False).values
n = len(peaks)
tr_emp = (n + 1.0) / np.arange(1, n + 1)

log_p = np.log10(peaks)
mean_lp, std_lp, skew_lp = np.mean(log_p), np.std(log_p, ddof=1), stats.skew(log_p)
tr_theor = np.linspace(1.05, 100, 100)
q_theor = 10 ** stats.pearson3.ppf(1.0 - (1.0 / tr_theor), skew=skew_lp, loc=mean_lp, scale=std_lp)

fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
ax.scatter(tr_emp, peaks, color="#d7191c", s=25, label="Empirical Peaks")
ax.plot(tr_theor, q_theor, color="#2b83ba", lw=2, label="LP-III Fit")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Return Period (Years)", fontweight="bold")
ax.set_ylabel("Peak Discharge ({value_col})", fontweight="bold")
ax.set_title("MOD-09: Flood Frequency", fontweight="bold")
ax.grid(True, which="major")
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()'''
    return fig, code


# ---------------------------------------------------------
# MOD-10: Streamflow Exceedance Probability Distribution (Weibull CDF)
# ---------------------------------------------------------
def mod10_exceedance_cdf(df, value_col):
    q = df[value_col].dropna().values
    n = len(q)
    q_sorted = np.sort(q)
    cdf = np.arange(1, n + 1) / (n + 1.0)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(q_sorted, cdf * 100.0, color="#2ca02c", lw=2, label="Empirical Cumulative Probability")

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.set_ylim(0, 100)
    ax.set_xlabel(f"Discharge ({value_col})", fontsize=10, fontweight="bold")
    ax.set_ylabel("Non-Exceedance Probability $P(X \leq x)$ (%)", fontsize=10, fontweight="bold")
    ax.set_title("MOD-10: Streamflow Cumulative Distribution Function (CDF)", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, which="major", color="#dddddd", linestyle="-", lw=0.7)
    ax.legend(loc="upper left", frameon=True, facecolor="#ffffff")
    plt.tight_layout()

    code = f'''# MOD-10: Standalone Cumulative Distribution Function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

q_sorted = np.sort(df["{value_col}"].dropna().values)
cdf = (np.arange(1, len(q_sorted) + 1) / (len(q_sorted) + 1.0)) * 100.0

fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
ax.plot(q_sorted, cdf, color="#2ca02c", lw=2, label="Empirical CDF")
ax.set_xscale("log")
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{{x:g}}"))
ax.set_ylim(0, 100)
ax.set_xlabel("Discharge ({value_col})", fontweight="bold")
ax.set_ylabel("Non-Exceedance Probability (%)", fontweight="bold")
ax.set_title("MOD-10: Cumulative Distribution Function", fontweight="bold")
ax.grid(True)
ax.legend(loc="upper left")
plt.tight_layout()
plt.show()'''
    return fig, code


# ---------------------------------------------------------
# MOD-11: Circular Polar Hydrologic Seasonality Clock (Leap-Adjusted)
# ---------------------------------------------------------
def mod11_circular_seasonality(df, date_col, value_col):
    df_clean = prepare_water_year_data(df, date_col, value_col)
    
    mean_daily = df_clean.groupby("Standard_DOWY")[value_col].mean().reindex(range(1, 367))
    theta = np.linspace(0, 2 * np.pi, 366, endpoint=False)
    
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2.0)

    values = mean_daily.values
    if np.isnan(values[151]):
        values[151] = (values[150] + values[152]) / 2.0

    ax.plot(theta, values, color="#1f77b4", lw=2, label="Mean Daily Flow")
    ax.fill(theta, values, color="#1f77b4", alpha=0.25)

    month_angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    ax.set_xticks(month_angles)
    ax.set_xticklabels(MONTH_LABELS, fontsize=9, fontweight="bold")
    ax.set_title("MOD-11: Circular Polar Hydrologic Regime (Water Year Cycle)", fontsize=11, fontweight="bold", pad=20)
    plt.tight_layout()

    code = f'''# MOD-11: Standalone Circular Polar Seasonality Plot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df["Date"] = pd.to_datetime(df["{date_col}"])
month, day, year = df["Date"].dt.month, df["Date"].dt.day, df["Date"].dt.year
df["WaterYear"] = np.where(month >= 10, year + 1, year)
ref_dates = pd.date_range("2019-10-01", "2020-09-30", freq="D")
dowy_lookup = {{(d.month, d.day): idx + 1 for idx, d in enumerate(ref_dates)}}
df["Standard_DOWY"] = [dowy_lookup.get((m, d)) for m, d in zip(month, day)]

daily_mean = df.groupby("Standard_DOWY")["{value_col}"].mean().reindex(range(1, 367)).fillna(method="ffill").values
theta = np.linspace(0, 2 * np.pi, 366, endpoint=False)

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={{"projection": "polar"}}, dpi=300)
ax.set_theta_direction(-1); ax.set_theta_offset(np.pi / 2.0)
ax.plot(theta, daily_mean, color="#1f77b4", lw=2)
ax.fill(theta, daily_mean, color="#1f77b4", alpha=0.25)
ax.set_xticks(np.linspace(0, 2 * np.pi, 12, endpoint=False))
ax.set_xticklabels({MONTH_LABELS})
ax.set_title("MOD-11: Circular Polar Hydrologic Regime", fontweight="bold", pad=20)
plt.tight_layout()
plt.show()'''
    return fig, code


# ---------------------------------------------------------
# MOD-12: Composite Percentile Envelope Hydrograph (DOWY 1-366)
# ---------------------------------------------------------
def mod12_percentile_envelope(df, date_col, value_col):
    df_clean = prepare_water_year_data(df, date_col, value_col)
    
    stats_df = df_clean.groupby("Standard_DOWY")[value_col].agg(
        p10=lambda x: np.percentile(x, 10),
        p25=lambda x: np.percentile(x, 25),
        p50=lambda x: np.percentile(x, 50),
        p75=lambda x: np.percentile(x, 75),
        p90=lambda x: np.percentile(x, 90),
    ).reindex(range(1, 367))

    days = np.arange(1, 367)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    
    ax.fill_between(days, stats_df["p10"], stats_df["p90"], color="#c6dbef", alpha=0.6, label="10th–90th Percentile Range")
    ax.fill_between(days, stats_df["p25"], stats_df["p75"], color="#6baed6", alpha=0.7, label="25th–75th Percentile Range")
    ax.plot(days, stats_df["p50"], color="#08519c", lw=2.2, label="Median Streamflow ($Q_{50}$)")

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.set_xticks(MONTH_TICKS_366)
    ax.set_xticklabels(MONTH_LABELS, fontsize=9)
    ax.set_xlim(1, 366)
    ax.set_xlabel("Day of Water Year (Leap-Adjusted)", fontsize=10, fontweight="bold")
    ax.set_ylabel(f"Discharge ({value_col})", fontsize=10, fontweight="bold")
    ax.set_title("MOD-12: Standardized Composite Percentile Hydrograph Envelope", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, which="major", color="#dddddd", linestyle="-", lw=0.7)
    ax.legend(loc="upper right", frameon=True, facecolor="#ffffff")
    plt.tight_layout()

    code = f'''# MOD-12: Standalone Percentile Envelope Hydrograph
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

df["Date"] = pd.to_datetime(df["{date_col}"])
month, day, year = df["Date"].dt.month, df["Date"].dt.day, df["Date"].dt.year
df["WaterYear"] = np.where(month >= 10, year + 1, year)
ref_dates = pd.date_range("2019-10-01", "2020-09-30", freq="D")
dowy_lookup = {{(d.month, d.day): idx + 1 for idx, d in enumerate(ref_dates)}}
df["Standard_DOWY"] = [dowy_lookup.get((m, d)) for m, d in zip(month, day)]

env = df.groupby("Standard_DOWY")["{value_col}"].agg(
    p10=lambda x: np.percentile(x, 10),
    p25=lambda x: np.percentile(x, 25),
    p50=lambda x: np.percentile(x, 50),
    p75=lambda x: np.percentile(x, 75),
    p90=lambda x: np.percentile(x, 90),
).reindex(range(1, 367))

days = np.arange(1, 367)
fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
ax.fill_between(days, env["p10"], env["p90"], color="#c6dbef", alpha=0.6, label="10th-90th %ile")
ax.fill_between(days, env["p25"], env["p75"], color="#6baed6", alpha=0.7, label="25th-75th %ile")
ax.plot(days, env["p50"], color="#08519c", lw=2, label="Median (Q50)")
ax.set_yscale("log")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{{y:g}}"))
ax.set_xticks({MONTH_TICKS_366})
ax.set_xticklabels({MONTH_LABELS})
ax.set_xlim(1, 366)
ax.set_xlabel("Day of Water Year", fontweight="bold")
ax.set_ylabel("Discharge ({value_col})", fontweight="bold")
ax.set_title("MOD-12: Composite Percentile Hydrograph Envelope", fontweight="bold")
ax.grid(True)
ax.legend(loc="upper right")
plt.tight_layout()
plt.show()'''
    return fig, code


# ---------------------------------------------------------
# Sidebar Controls & Data Ingestion
# ---------------------------------------------------------
st.sidebar.title("🌊 Hydro-Studio")
st.sidebar.markdown("**Deterministic Hydrological Engine**")
st.sidebar.markdown("---")

data_source = st.sidebar.radio("Data Ingestion", ["Use Benchmark Dataset", "Upload Custom CSV"])

if data_source == "Upload Custom CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV (Date, Discharge)", type=["csv"])
    if uploaded_file is not None:
        try:
            df_current = pd.read_csv(uploaded_file)
            st.sidebar.success("CSV Loaded Successfully")
        except Exception as e:
            st.sidebar.error(f"Error reading CSV: {e}")
            df_current = get_benchmark_data()
    else:
        st.sidebar.info("Awaiting CSV. Displaying benchmark data.")
        df_current = get_benchmark_data()
else:
    df_current = get_benchmark_data()
    st.sidebar.caption("Benchmark Data: Synthetic 7-year multi-regime daily streamflow series.")

col_names = list(df_current.columns)
date_candidates = [c for c in col_names if "date" in c.lower() or "time" in c.lower()]
val_candidates = [c for c in col_names if c not in date_candidates]

date_col = st.sidebar.selectbox("Date Column", col_names, index=col_names.index(date_candidates[0]) if date_candidates else 0)
val_col = st.sidebar.selectbox("Discharge Column", col_names, index=col_names.index(val_candidates[0]) if val_candidates else (1 if len(col_names) > 1 else 0))

st.sidebar.markdown("---")
modules_dict = {
    "MOD-01: Flow Duration Curve (FDC)": ("Flow Duration Curve", mod01_fdc, [df_current, val_col]),
    "MOD-02: Lag-1 Rate-of-Change Scatter": ("Lag-1 Rate-of-Change Scatterplot", mod02_lag1_scatter, [df_current, val_col]),
    "MOD-03: 2D Raster Hydrograph": ("2D Raster Hydrograph", mod03_raster_hydrograph, [df_current, date_col, val_col]),
    "MOD-04: Enhanced FDC (eFDC Partitioning)": ("Enhanced Flow Duration Curve", mod04_efdc, [df_current, val_col]),
    "MOD-05: Discrete State Transition Matrix": ("Discrete State Transition Matrix", mod05_transition_matrix, [df_current, val_col]),
    "MOD-06: Annual Regime Carpet (Ranked)": ("Annual Regime Carpet Hydrograph", mod06_ranked_carpet, [df_current, date_col, val_col]),
    "MOD-07: Comparative Dual-FDC Overlay": ("Comparative Dual-FDC", mod07_comparative_fdc, [df_current, date_col, val_col]),
    "MOD-08: Flashiness & Recession Rate": ("Hydrograph Recession Rate Analysis", mod08_recession_rate, [df_current, val_col]),
    "MOD-09: Flood Frequency Analysis (LP-III)": ("Flood Frequency Analysis (LP-III)", mod09_flood_frequency, [df_current, date_col, val_col]),
    "MOD-10: Exceedance Probability Distribution (CDF)": ("Non-Exceedance Probability CDF", mod10_exceedance_cdf, [df_current, val_col]),
    "MOD-11: Circular Polar Seasonality Clock": ("Circular Polar Hydrologic Regime", mod11_circular_seasonality, [df_current, date_col, val_col]),
    "MOD-12: Composite Percentile Envelope": ("Composite Percentile Envelope Hydrograph", mod12_percentile_envelope, [df_current, date_col, val_col]),
}

selected_module_key = st.sidebar.selectbox("Visual Analytics Routine", list(modules_dict.keys()))
prompt_name, module_func, module_args = modules_dict[selected_module_key]

# ---------------------------------------------------------
# Main Panel: Plot Display & Exporter
# ---------------------------------------------------------
st.subheader(f"Baseline Rendering: {selected_module_key}")

fig, current_code = module_func(*module_args)
st.pyplot(fig)

# ---------------------------------------------------------
# Exporter & Workshop AI Copilot Workspace
# ---------------------------------------------------------
st.markdown("---")
st.subheader("🛠️ Export & AI Copilot Workspace")
st.markdown(
    "Hydro-Studio establishes mathematical ground truth. Use the panels below to export "
    "reproducible Python scripts, or copy a structured prompt into **Google Gemini** to customize the visualization."
)

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("#### 1. Pure Python Baseline Script")
    st.caption("Deterministic, publication-grade code ready to execute locally.")
    st.code(current_code, language="python")

with col_right:
    st.markdown("#### 2. Workshop AI Copilot Prompt")
    st.caption("Copy this entire prompt into Gemini to request modifications safely.")
    
    custom_goal = st.selectbox(
        "Select an AI Modification Goal:",
        [
            "Add shaded ribbons for 25th-75th flow percentiles and annotate the median.",
            "Highlight extreme flash flood transitions (+dQ/dt) exceeding 90th percentile.",
            "Apply ASCE journal typography standards (300 DPI, single-column width).",
            "Add an interactive threshold line for historical bankfull discharge.",
            "Overlay PDSI drought classifications as shaded background horizontal spans.",
        ]
    )

    structured_prompt = f"""I have working, verified Python code for a hydrological {prompt_name}.

Here is the baseline script:
```python
{current_code}
