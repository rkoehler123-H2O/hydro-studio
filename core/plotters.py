import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap, LogNorm
import numpy as np
import pandas as pd

# Global styling configuration per VDA standards
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#000000'
plt.rcParams['axes.linewidth'] = 0.8

def configure_log_axis(ax, axis='y'):
    """Applies strict base-10 numerical labeling and major/minor grids."""
    target_axis = ax.yaxis if axis == 'y' else ax.xaxis
    target_axis.set_major_locator(ticker.LogLocator(base=10.0, numticks=12))
    target_axis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12))
    target_axis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.grid(True, which='major', linestyle='-', color='#000000', linewidth=0.8, alpha=0.7)
    ax.grid(True, which='minor', linestyle='--', color='#808080', linewidth=0.5, alpha=0.7)

# ==========================================
# MOD-01: Enhanced Flow Duration Curve (eFDC)
# ==========================================
def get_mod01_code(station_name: str) -> str:
    return f"""import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
plot_df = df[(df['Q_t'] > 0) & (df['Q_next'] > 0)].copy()
fdc_curve = plot_df.sort_values(by='exceedance_pct')

ax.plot(fdc_curve['exceedance_pct'], fdc_curve['Q_t'], color='#000000', linewidth=1.5, zorder=3, label='Baseline FDC [Q(t)]')

rising = plot_df[plot_df['dQ_dt'] > 0]
falling = plot_df[plot_df['dQ_dt'] < 0]
steady = plot_df[plot_df['dQ_dt'] == 0]

ax.scatter(rising['exceedance_pct'], rising['Q_next'], color='#0000FF', alpha=0.30, s=12, edgecolors='none', zorder=4, label='Rising Limb (+dQ/dt)')
ax.scatter(falling['exceedance_pct'], falling['Q_next'], color='#FF0000', alpha=0.30, s=12, edgecolors='none', zorder=4, label='Falling Limb (-dQ/dt)')
ax.scatter(steady['exceedance_pct'], steady['Q_next'], color='#000000', alpha=0.30, s=12, edgecolors='none', zorder=4, label='Steady State (dQ/dt = 0)')

ax.set_yscale('log')
ax.set_xlim(0, 100)
ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=12))
ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{{y:g}}"))
ax.grid(True, which='major', linestyle='-', color='#000000', linewidth=0.8, alpha=0.7)
ax.grid(True, which='minor', linestyle='--', color='#808080', linewidth=0.5, alpha=0.7)

ax.set_xlabel("Exceedance Percentage P(Q_t) (%)", fontsize=10, fontweight='bold')
ax.set_ylabel("Discharge Q(t+1)", fontsize=10, fontweight='bold')
ax.set_title("MOD-01: Enhanced Flow Duration Curve (eFDC)\\n{station_name}", fontsize=11, fontweight='bold', pad=32)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=True, fontsize=8, edgecolor='#808080')
"""

def plot_mod01_efdc(df: pd.DataFrame, station_name: str = "Streamflow Station"):
    code = get_mod01_code(station_name)
    local_env = {"df": df.copy(), "plt": plt, "ticker": ticker, "np": np}
    exec(code, local_env)
    return local_env.get("fig", plt.gcf()), code

# ==========================================
# MOD-02: Lag-1 Differential Phase Space
# ==========================================
def get_mod02_code(station_name: str) -> str:
    return f"""import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
plot_df = df[(df['Q_t'] > 0) & (df['Q_next'] > 0)].copy()
q_min = min(plot_df['Q_t'].min(), plot_df['Q_next'].min())
q_max = max(plot_df['Q_t'].max(), plot_df['Q_next'].max())

ref_vals = np.logspace(np.log10(q_min * 0.8), np.log10(q_max * 1.2), 100)
ax.plot(ref_vals, ref_vals, color='#000000', linewidth=1.2, linestyle='-', zorder=2, label='Steady State (y = x)')
ax.plot(ref_vals, ref_vals * 2.0, color='#808080', linewidth=0.8, linestyle='--', zorder=2, label='Doubling (y = 2x)')
ax.plot(ref_vals, ref_vals * 0.5, color='#808080', linewidth=0.8, linestyle='--', zorder=2, label='Halving (y = 0.5x)')

rising = plot_df[plot_df['dQ_dt'] > 0]
falling = plot_df[plot_df['dQ_dt'] < 0]
steady = plot_df[plot_df['dQ_dt'] == 0]

ax.scatter(rising['Q_t'], rising['Q_next'], color='#0000FF', alpha=0.30, s=12, edgecolors='none', zorder=3, label='+dQ/dt')
ax.scatter(falling['Q_t'], falling['Q_next'], color='#FF0000', alpha=0.30, s=12, edgecolors='none', zorder=3, label='-dQ/dt')
ax.scatter(steady['Q_t'], steady['Q_next'], color='#000000', alpha=0.30, s=12, edgecolors='none', zorder=3, label='dQ/dt = 0')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(q_min * 0.8, q_max * 1.2)
ax.set_ylim(q_min * 0.8, q_max * 1.2)
ax.set_aspect('equal', adjustable='box')

for axis_obj in [ax.xaxis, ax.yaxis]:
    axis_obj.set_major_locator(ticker.LogLocator(base=10.0, numticks=12))
    axis_obj.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12))
    axis_obj.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{{y:g}}"))

ax.grid(True, which='major', linestyle='-', color='#000000', linewidth=0.8, alpha=0.7)
ax.grid(True, which='minor', linestyle='--', color='#808080', linewidth=0.5, alpha=0.7)

ax.set_xlabel("Discharge Today Q(t)", fontsize=10, fontweight='bold')
ax.set_ylabel("Discharge Tomorrow Q(t+1)", fontsize=10, fontweight='bold')
ax.set_title("MOD-02: Lag-1 Differential Phase Space\\n{station_name}", fontsize=11, fontweight='bold', pad=36)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=True, fontsize=8, edgecolor='#808080')
"""

def plot_mod02_lag1(df: pd.DataFrame, station_name: str = "Streamflow Station"):
    code = get_mod02_code(station_name)
    local_env = {"df": df.copy(), "plt": plt, "ticker": ticker, "np": np}
    exec(code, local_env)
    return local_env.get("fig", plt.gcf()), code

# ==========================================
# Helper: Discrete Transition Matrix Engine
# ==========================================
def _compute_transition_bins(df: pd.DataFrame, delta_log10: float = 0.10):
    plot_df = df[(df['Q_t'] > 0) & (df['Q_next'] > 0)].copy()
    min_flow = min(plot_df['Q_t'].min(), plot_df['Q_next'].min())
    max_flow = max(plot_df['Q_t'].max(), plot_df['Q_next'].max())
    
    start_dec = np.floor(np.log10(min_flow))
    end_dec = np.ceil(np.log10(max_flow))
    log_bins = np.arange(start_dec, end_dec + delta_log10, delta_log10)
    bin_edges = 10.0 ** log_bins
    
    plot_df['x_bin'] = np.digitize(plot_df['Q_t'], bin_edges) - 1
    plot_df['y_bin'] = np.digitize(plot_df['Q_next'], bin_edges) - 1
    
    num_bins = len(bin_edges) - 1
    matrix = np.zeros((num_bins, num_bins), dtype=float)
    for _, row in plot_df.iterrows():
        xi, yi = int(row['x_bin']), int(row['y_bin'])
        if 0 <= xi < num_bins and 0 <= yi < num_bins:
            matrix[yi, xi] += 1
            
    return matrix, bin_edges, min_flow, max_flow

# ==========================================
# MOD-03: Discrete Transition Matrix (Counts)
# ==========================================
def get_mod03_code(station_name: str) -> str:
    return f"""import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

fig, ax = plt.subplots(figsize=(7, 7), dpi=300)
num_bins = len(bin_edges) - 1

# 1:1 Identity Line
ax.plot([bin_edges[0], bin_edges[-1]], [bin_edges[0], bin_edges[-1]], color='#000000', linewidth=1.0, linestyle='-', zorder=2)

for i in range(num_bins):
    for j in range(num_bins):
        cnt = matrix[j, i]
        if cnt > 0:
            cx = np.sqrt(bin_edges[i] * bin_edges[i+1])
            cy = np.sqrt(bin_edges[j] * bin_edges[j+1])
            if j > i:
                col = '#0000FF' # Rising
            elif j < i:
                col = '#FF0000' # Falling
            else:
                col = '#000000' # Steady
            bbox_dict = dict(boxstyle='square,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85) if i == j else None
            ax.text(cx, cy, f"{{int(cnt)}}", ha='center', va='center', fontsize=6, fontweight='bold', color=col, bbox=bbox_dict, zorder=4)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(bin_edges[0], bin_edges[-1])
ax.set_ylim(bin_edges[0], bin_edges[-1])
ax.set_aspect('equal', adjustable='box')

for axis_obj in [ax.xaxis, ax.yaxis]:
    axis_obj.set_major_locator(ticker.LogLocator(base=10.0, numticks=12))
    axis_obj.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12))
    axis_obj.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{{y:g}}"))

ax.grid(True, which='major', linestyle='-', color='#000000', linewidth=0.8, alpha=0.7)
ax.grid(True, which='minor', linestyle='--', color='#808080', linewidth=0.5, alpha=0.7)

ax.set_xlabel("Antecedent Discharge State Q(t)", fontsize=10, fontweight='bold')
ax.set_ylabel("Transition Discharge State Q(t+1)", fontsize=10, fontweight='bold')
ax.set_title("MOD-03: Discrete Transition Matrix (Raw Counts)\\n{station_name}", fontsize=11, fontweight='bold', pad=32)
"""

def plot_mod03_matrix(df: pd.DataFrame, station_name: str = "Streamflow Station"):
    matrix, bin_edges, _, _ = _compute_transition_bins(df)
    code = get_mod03_code(station_name)
    local_env = {"matrix": matrix, "bin_edges": bin_edges, "plt": plt, "ticker": ticker, "np": np}
    exec(code, local_env)
    return local_env.get("fig", plt.gcf()), code

# ==========================================
# MOD-04: Column-Normalized Forecast Matrix
# ==========================================
def get_mod04_code(station_name: str) -> str:
    return f"""import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

fig, ax = plt.subplots(figsize=(7, 7), dpi=300)
num_bins = len(bin_edges) - 1

ax.plot([bin_edges[0], bin_edges[-1]], [bin_edges[0], bin_edges[-1]], color='#000000', linewidth=1.0, linestyle='-', zorder=2)

for i in range(num_bins):
    for j in range(num_bins):
        pct = norm_matrix[j, i]
        if pct > 0:
            cx = np.sqrt(bin_edges[i] * bin_edges[i+1])
            cy = np.sqrt(bin_edges[j] * bin_edges[j+1])
            col = '#0000FF' if j > i else ('#FF0000' if j < i else '#000000')
            txt = f"{{pct:.1f}}%" if pct >= 1.0 else f"{{pct:.2f}}%"
            bbox_dict = dict(boxstyle='square,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85) if i == j else None
            ax.text(cx, cy, txt, ha='center', va='center', fontsize=5.5, fontweight='bold', color=col, bbox=bbox_dict, zorder=4)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(bin_edges[0], bin_edges[-1])
ax.set_ylim(bin_edges[0], bin_edges[-1])
ax.set_aspect('equal', adjustable='box')

for axis_obj in [ax.xaxis, ax.yaxis]:
    axis_obj.set_major_locator(ticker.LogLocator(base=10.0, numticks=12))
    axis_obj.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12))
    axis_obj.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{{y:g}}"))

ax.grid(True, which='major', linestyle='-', color='#000000', linewidth=0.8, alpha=0.7)
ax.grid(True, which='minor', linestyle='--', color='#808080', linewidth=0.5, alpha=0.7)

ax.set_xlabel("Observed State Today Q(t)", fontsize=10, fontweight='bold')
ax.set_ylabel("Forecast State Tomorrow Q(t+1)", fontsize=10, fontweight='bold')
ax.set_title("Forecast / Column-Normalized Matrix [P(Q(t+1) | Q(t))]\\n{station_name}", fontsize=11, fontweight='bold', pad=32)
"""

def plot_mod04_forecast(df: pd.DataFrame, station_name: str = "Streamflow Station"):
    matrix, bin_edges, _, _ = _compute_transition_bins(df)
    col_sums = matrix.sum(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        norm_matrix = np.where(col_sums > 0, (matrix / col_sums) * 100.0, 0.0)
    code = get_mod04_code(station_name)
    local_env = {"norm_matrix": norm_matrix, "bin_edges": bin_edges, "plt": plt, "ticker": ticker, "np": np}
    exec(code, local_env)
    return local_env.get("fig", plt.gcf()), code

# ==========================================
# MOD-05: Row-Normalized Antecedent Matrix
# ==========================================
def get_mod05_code(station_name: str) -> str:
    return f"""import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

fig, ax = plt.subplots(figsize=(7, 7), dpi=300)
num_bins = len(bin_edges) - 1

ax.plot([bin_edges[0], bin_edges[-1]], [bin_edges[0], bin_edges[-1]], color='#000000', linewidth=1.0, linestyle='-', zorder=2)

for i in range(num_bins):
    for j in range(num_bins):
        pct = norm_matrix[j, i]
        if pct > 0:
            cx = np.sqrt(bin_edges[i] * bin_edges[i+1])
            cy = np.sqrt(bin_edges[j] * bin_edges[j+1])
            col = '#0000FF' if j > i else ('#FF0000' if j < i else '#000000')
            txt = f"{{pct:.1f}}%" if pct >= 1.0 else f"{{pct:.2f}}%"
            bbox_dict = dict(boxstyle='square,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85) if i == j else None
            ax.text(cx, cy, txt, ha='center', va='center', fontsize=5.5, fontweight='bold', color=col, bbox=bbox_dict, zorder=4)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(bin_edges[0], bin_edges[-1])
ax.set_ylim(bin_edges[0], bin_edges[-1])
ax.set_aspect('equal', adjustable='box')

for axis_obj in [ax.xaxis, ax.yaxis]:
    axis_obj.set_major_locator(ticker.LogLocator(base=10.0, numticks=12))
    axis_obj.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12))
    axis_obj.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{{y:g}}"))

ax.grid(True, which='major', linestyle='-', color='#000000', linewidth=0.8, alpha=0.7)
ax.grid(True, which='minor', linestyle='--', color='#808080', linewidth=0.5, alpha=0.7)

ax.set_xlabel("Antecedent State Yesterday Q(t)", fontsize=10, fontweight='bold')
ax.set_ylabel("Observed State Today Q(t+1)", fontsize=10, fontweight='bold')
ax.set_title("Antecedent / Row-Normalized Matrix [P(Q(t) | Q(t+1))]\\n{station_name}", fontsize=11, fontweight='bold', pad=32)
"""

def plot_mod05_antecedent(df: pd.DataFrame, station_name: str = "Streamflow Station"):
    matrix, bin_edges, _, _ = _compute_transition_bins(df)
    row_sums = matrix.sum(axis=1)[:, np.newaxis]
    with np.errstate(divide='ignore', invalid='ignore'):
        norm_matrix = np.where(row_sums > 0, (matrix / row_sums) * 100.0, 0.0)
    code = get_mod05_code(station_name)
    local_env = {"norm_matrix": norm_matrix, "bin_edges": bin_edges, "plt": plt, "ticker": ticker, "np": np}
    exec(code, local_env)
    return local_env.get("fig", plt.gcf()), code

# ==========================================
# MOD-06: Sequential Flow Duration & Persistence
# ==========================================
def get_mod06_code(station_name: str) -> str:
    return f"""import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

fig, ax = plt.subplots(figsize=(11, 7), dpi=300)

for (cx, dur), count in counts.items():
    ax.text(cx, dur, f"{{count}}", ha='center', va='center', fontsize=7, fontweight='bold', color='#000000')

ax.set_xscale('log')
ax.set_xlim(bin_edges[0], bin_edges[-1])
ax.set_ylim(0, max_dur + 2)

# Grids: 7-day solid major, 1-day dashed minor
ax.yaxis.set_major_locator(ticker.MultipleLocator(7))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
ax.grid(True, which='major', axis='y', linestyle='-', color='#000000', linewidth=0.8)
ax.grid(True, which='minor', axis='y', linestyle='--', color='#B0B0B0', linewidth=0.45)

ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=12))
ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12))
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{{y:g}}"))
ax.grid(True, which='major', axis='x', linestyle='-', color='#000000', linewidth=0.8)

ax.set_xlabel("Discharge Class (Categorized Bins)", fontsize=10, fontweight='bold')
ax.set_ylabel("Sequence Duration (Continuous Days)", fontsize=10, fontweight='bold')
ax.set_title("MOD-06: Sequential Flow Duration & Persistence Analysis\\n{station_name}", fontsize=11, fontweight='bold', pad=25)
"""

def plot_mod06_persistence(df: pd.DataFrame, station_name: str = "Streamflow Station"):
    _, bin_edges, _, _ = _compute_transition_bins(df)
    valid_df = df[df['Q_t'] > 0].copy()
    valid_df['bin_idx'] = np.digitize(valid_df['Q_t'], bin_edges) - 1
    
    runs = []
    current_bin = None
    cur_len = 0
    for b in valid_df['bin_idx']:
        if b == current_bin:
            cur_len += 1
        else:
            if current_bin is not None and 0 <= current_bin < len(bin_edges) - 1:
                cx = np.sqrt(bin_edges[current_bin] * bin_edges[current_bin + 1])
                runs.append((cx, cur_len))
            current_bin = b
            cur_len = 1
    if current_bin is not None and 0 <= current_bin < len(bin_edges) - 1:
        cx = np.sqrt(bin_edges[current_bin] * bin_edges[current_bin + 1])
        runs.append((cx, cur_len))
        
    counts = pd.Series(runs).value_counts().to_dict()
    max_dur = max([r[1] for r in runs]) if runs else 14
    
    code = get_mod06_code(station_name)
    local_env = {"counts": counts, "bin_edges": bin_edges, "max_dur": max_dur, "plt": plt, "ticker": ticker, "np": np}
    exec(code, local_env)
    return local_env.get("fig", plt.gcf()), code

# ==========================================
# Helper: USGS Palette & Raster Matrix Builder
# ==========================================
USGS_HEX_LIST = [
    "#730000", "#C40000", "#FF5500", "#FFAA00", 
    "#E6E600", "#55FF00", "#00AA00", "#00FFFF", 
    "#0070FF", "#0000FF", "#000080"
]

def _build_raster_grid(df: pd.DataFrame):
    valid = df[df['Q_t'] >= 0].copy()
    years = sorted(valid['water_year'].unique())
    grid = np.full((len(years), 366), np.nan)
    
    for i, y in enumerate(years):
        sub = valid[valid['water_year'] == y]
        for _, row in sub.iterrows():
            d = int(row['DOWY'])
            if 1 <= d <= 366:
                grid[i, d - 1] = row['Q_t']
                
    return grid, years

# ==========================================
# MOD-07: Chronological Raster Hydrograph
# ==========================================
def get_mod07_code(station_name: str, cmap_name: str) -> str:
    return f"""import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm, ListedColormap
import numpy as np

fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

pos_min = np.nanmin(grid[grid > 0]) if np.any(grid > 0) else 0.1
pos_max = np.nanmax(grid) if np.any(~np.isnan(grid)) else 1000

if "{cmap_name}" == "USGS Discrete":
    cmap = ListedColormap({USGS_HEX_LIST})
    norm = LogNorm(vmin=pos_min, vmax=pos_max)
else:
    cmap = plt.get_cmap("{cmap_name}")
    norm = LogNorm(vmin=pos_min, vmax=pos_max)

im = ax.imshow(grid, aspect='auto', origin='lower', cmap=cmap, norm=norm, extent=[0.5, 366.5, -0.5, len(years) - 0.5])

# Exterior right colorbar
cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.04)
cbar.set_label("Discharge Q", fontsize=10, fontweight='bold')
cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{{y:g}}"))

# Water Year vertical axis
step = max(1, len(years) // 8)
y_ticks = np.arange(0, len(years), step)
ax.set_yticks(y_ticks)
ax.set_yticklabels([str(years[i]) for i in y_ticks])
ax.grid(True, which='major', axis='y', linestyle='-', color='#000000', linewidth=0.8)

# Month boundaries (Water Year Oct 1 -> Sep 30)
month_dowy = [1, 32, 62, 93, 124, 152, 183, 213, 244, 274, 305, 335]
month_names = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
for m_start in month_dowy:
    ax.axvline(m_start, color='#000000', linestyle='--', linewidth=0.75)

ax.set_xticks(month_dowy)
ax.set_xticklabels(month_names, fontweight='bold')
ax.set_xlim(1, 366)
ax.set_xlabel("Water Year Calendar Month", fontsize=10, fontweight='bold')
ax.set_ylabel("Water Year (Chronological)", fontsize=10, fontweight='bold')
ax.set_title("MOD-07: Chronological Raster Hydrograph\\n{station_name}", fontsize=11, fontweight='bold', pad=25)
"""

def plot_mod07_raster(df: pd.DataFrame, station_name: str = "Streamflow Station", cmap: str = "USGS Discrete"):
    grid, years = _build_raster_grid(df)
    code = get_mod07_code(station_name, cmap)
    local_env = {"grid": grid, "years": years, "plt": plt, "ticker": ticker, "np": np, "ListedColormap": ListedColormap}
    exec(code, local_env)
    return local_env.get("fig", plt.gcf()), code

# ==========================================
# MOD-08: Volumetric Raster Hydrograph
# ==========================================
def get_mod08_code(station_name: str, cmap_name: str) -> str:
    return f"""import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm, ListedColormap
import numpy as np

fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

pos_min = np.nanmin(ranked_grid[ranked_grid > 0]) if np.any(ranked_grid > 0) else 0.1
pos_max = np.nanmax(ranked_grid) if np.any(~np.isnan(ranked_grid)) else 1000

if "{cmap_name}" == "USGS Discrete":
    cmap = ListedColormap({USGS_HEX_LIST})
    norm = LogNorm(vmin=pos_min, vmax=pos_max)
else:
    cmap = plt.get_cmap("{cmap_name}")
    norm = LogNorm(vmin=pos_min, vmax=pos_max)

im = ax.imshow(ranked_grid, aspect='auto', origin='upper', cmap=cmap, norm=norm, extent=[0.5, 366.5, len(ranked_years) + 0.5, 0.5])

cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.04)
cbar.set_label("Discharge Q", fontsize=10, fontweight='bold')
cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{{y:g}}"))

major_ranks = [1] + [r for r in range(10, len(ranked_years) + 1, 10)]
ax.set_yticks(major_ranks)
ax.set_yticklabels([str(r) for r in major_ranks])
ax.grid(True, which='major', axis='y', linestyle='-', color='#000000', linewidth=0.8)

month_dowy = [1, 32, 62, 93, 124, 152, 183, 213, 244, 274, 305, 335]
month_names = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
for m_start in month_dowy:
    ax.axvline(m_start, color='#000000', linestyle='--', linewidth=0.75)

ax.set_xticks(month_dowy)
ax.set_xticklabels(month_names, fontweight='bold')
ax.set_xlim(1, 366)
ax.set_ylim(len(ranked_years) + 0.5, 0.5)
ax.set_xlabel("Water Year Calendar Month", fontsize=10, fontweight='bold')
ax.set_ylabel("Volumetric Rank (1 = Greatest Annual Volume)", fontsize=10, fontweight='bold')
ax.set_title("MOD-08: Volumetric Raster Hydrograph\\n{station_name}", fontsize=11, fontweight='bold', pad=25)
"""

def plot_mod08_volumetric_raster(df: pd.DataFrame, station_name: str = "Streamflow Station", cmap: str = "USGS Discrete"):
    grid, years = _build_raster_grid(df)
    annual_vols = np.nansum(grid, axis=1)
    rank_order = np.argsort(-annual_vols)
    ranked_grid = grid[rank_order]
    ranked_years = [years[i] for i in rank_order]
    
    code = get_mod08_code(station_name, cmap)
    local_env = {"ranked_grid": ranked_grid, "ranked_years": ranked_years, "plt": plt, "ticker": ticker, "np": np, "ListedColormap": ListedColormap}
    exec(code, local_env)
    return local_env.get("fig", plt.gcf()), code

# ==========================================
# MOD-09: Annual FDC Spaghetti Plot
# ==========================================
def get_mod09_code(station_name: str, cmap_name: str) -> str:
    return f"""import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
cmap = plt.get_cmap("{cmap_name if cmap_name != 'USGS Discrete' else 'Spectral_r'}")
norm = plt.Normalize(vmin=min(unique_years), vmax=max(unique_years))

for y in unique_years:
    sub = df[(df['water_year'] == y) & (df['Q_t'] > 0)].copy()
    if len(sub) > 20:
        sub = sub.sort_values(by='Q_t', ascending=False)
        ranks = np.arange(1, len(sub) + 1)
        exc = (ranks / (len(sub) + 1)) * 100.0
        ax.plot(exc, sub['Q_t'], color=cmap(norm(y)), alpha=0.35, linewidth=1.0)

ax.set_yscale('log')
ax.set_xlim(0, 100)
ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=12))
ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{{y:g}}"))
ax.grid(True, which='major', linestyle='-', color='#000000', linewidth=0.8, alpha=0.7)
ax.grid(True, which='minor', linestyle='--', color='#808080', linewidth=0.5, alpha=0.7)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.03, fraction=0.04)
cbar.set_label("Water Year", fontsize=10, fontweight='bold')

ax.set_xlabel("Exceedance Percentage (%)", fontsize=10, fontweight='bold')
ax.set_ylabel("Discharge Q", fontsize=10, fontweight='bold')
ax.set_title("MOD-09: Annual FDC Spaghetti Plot\\n{station_name}", fontsize=11, fontweight='bold', pad=25)
"""

def plot_mod09_spaghetti(df: pd.DataFrame, station_name: str = "Streamflow Station", cmap: str = "Spectral_r"):
    unique_years = sorted(df['water_year'].unique())
    code = get_mod09_code(station_name, cmap)
    local_env = {"df": df.copy(), "unique_years": unique_years, "plt": plt, "ticker": ticker, "np": np}
    exec(code, local_env)
    return local_env.get("fig", plt.gcf()), code

# ==========================================
# Helper: Annual FDC Threshold Computer
# ==========================================
def _compute_annual_thresholds(df: pd.DataFrame):
    records = []
    for y in sorted(df['water_year'].unique()):
        sub = df[(df['water_year'] == y) & (df['Q_t'] > 0)]
        if len(sub) > 60:
            q10 = np.percentile(sub['Q_t'], 90)
            q50 = np.percentile(sub['Q_t'], 50)
            q90 = np.percentile(sub['Q_t'], 10)
            vol = sub['Q_t'].sum()
            records.append({'water_year': y, 'Q10': q10, 'Q50': q50, 'Q90': q90, 'vol': vol})
    tdf = pd.DataFrame(records)
    tdf = tdf.sort_values(by='vol', ascending=False).reset_index(drop=True)
    tdf['vol_rank'] = tdf.index + 1
    return tdf

# ==========================================
# MOD-10: Annual FDC Chronological Thresholds
# ==========================================
def get_mod010_code(station_name: str) -> str:
    return f"""import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
chrono_df = tdf.sort_values(by='water_year')

ax.plot(chrono_df['water_year'], chrono_df['Q10'], color='#0000FF', marker='o', markersize=4, linestyle='-', linewidth=1.0, label='Q10 (High Flow)')
ax.plot(chrono_df['water_year'], chrono_df['Q50'], color='#FFD700', marker='s', markersize=4, linestyle='-', linewidth=1.0, label='Q50 (Median Flow)')
ax.plot(chrono_df['water_year'], chrono_df['Q90'], color='#FF0000', marker='^', markersize=4, linestyle='-', linewidth=1.0, label='Q90 (Low Flow)')

ax.set_yscale('log')
ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=12))
ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{{y:g}}"))
ax.grid(True, which='major', linestyle='-', color='#000000', linewidth=0.8, alpha=0.7)
ax.grid(True, which='minor', linestyle='--', color='#808080', linewidth=0.5, alpha=0.7)

ax.set_xlabel("Water Year", fontsize=10, fontweight='bold')
ax.set_ylabel("Discharge Q", fontsize=10, fontweight='bold')
ax.set_title("MOD-10: Annual FDC Threshold Trends (Chronological)\\n{station_name}", fontsize=11, fontweight='bold', pad=32)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=True, fontsize=8, edgecolor='#808080')
"""

def plot_mod10_chrono_thresholds(df: pd.DataFrame, station_name: str = "Streamflow Station"):
    tdf = _compute_annual_thresholds(df)
    code = get_mod010_code(station_name)
    local_env = {"tdf": tdf, "plt": plt, "ticker": ticker, "np": np}
    exec(code, local_env)
    return local_env.get("fig", plt.gcf()), code

# ==========================================
# MOD-11: Annual FDC Volumetric Thresholds
# ==========================================
def get_mod011_code(station_name: str) -> str:
    return f"""import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
rank_df = tdf.sort_values(by='vol_rank')

ax.plot(rank_df['vol_rank'], rank_df['Q10'], color='#0000FF', marker='o', markersize=4, linestyle='-', linewidth=1.0, label='Q10 (High Flow)')
ax.plot(rank_df['vol_rank'], rank_df['Q50'], color='#FFD700', marker='s', markersize=4, linestyle='-', linewidth=1.0, label='Q50 (Median Flow)')
ax.plot(rank_df['vol_rank'], rank_df['Q90'], color='#FF0000', marker='^', markersize=4, linestyle='-', linewidth=1.0, label='Q90 (Low Flow)')

ax.set_yscale('log')
ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=12))
ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{{y:g}}"))
ax.grid(True, which='major', linestyle='-', color='#000000', linewidth=0.8, alpha=0.7)
ax.grid(True, which='minor', linestyle='--', color='#808080', linewidth=0.5, alpha=0.7)

ax.set_xlabel("Volumetric Rank (1 = Wettest)", fontsize=10, fontweight='bold')
ax.set_ylabel("Discharge Q", fontsize=10, fontweight='bold')
ax.set_title("MOD-11: Annual FDC Volumetric Thresholds (Rank-Ordered)\\n{station_name}", fontsize=11, fontweight='bold', pad=32)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=True, fontsize=8, edgecolor='#808080')
"""

def plot_mod11_vol_thresholds(df: pd.DataFrame, station_name: str = "Streamflow Station"):
    tdf = _compute_annual_thresholds(df)
    code = get_mod011_code(station_name)
    local_env = {"tdf": tdf, "plt": plt, "ticker": ticker, "np": np}
    exec(code, local_env)
    return local_env.get("fig", plt.gcf()), code

# ==========================================
# MOD-12: Composite Hydroinformatics Dashboard
# ==========================================
def get_mod012_code(station_name: str) -> str:
    return f"""import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm, ListedColormap
import numpy as np

fig, axs = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
plt.subplots_adjust(hspace=0.35, wspace=0.25)

# Panel A: MOD-02 Lag-1
axA = axs[0, 0]
q_min = min(plot_df['Q_t'].min(), plot_df['Q_next'].min())
q_max = max(plot_df['Q_t'].max(), plot_df['Q_next'].max())
ref_vals = np.logspace(np.log10(q_min * 0.8), np.log10(q_max * 1.2), 100)
axA.plot(ref_vals, ref_vals, color='#000000', linewidth=1.0)
axA.scatter(plot_df[plot_df['dQ_dt'] > 0]['Q_t'], plot_df[plot_df['dQ_dt'] > 0]['Q_next'], color='#0000FF', alpha=0.25, s=8, label='+dQ/dt')
axA.scatter(plot_df[plot_df['dQ_dt'] < 0]['Q_t'], plot_df[plot_df['dQ_dt'] < 0]['Q_next'], color='#FF0000', alpha=0.25, s=8, label='-dQ/dt')
axA.set_xscale('log'); axA.set_yscale('log')
axA.set_xlim(q_min * 0.8, q_max * 1.2); axA.set_ylim(q_min * 0.8, q_max * 1.2)
axA.set_aspect('equal', adjustable='box')
for axis_obj in [axA.xaxis, axA.yaxis]:
    axis_obj.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{{y:g}}"))
axA.grid(True, which='major', color='#000000', linewidth=0.8); axA.grid(True, which='minor', color='#808080', linewidth=0.5, linestyle='--')
axA.set_title("A. Lag-1 Phase Space", loc='left', fontweight='bold', fontsize=11)
axA.set_xlabel("Q(t)", fontweight='bold'); axA.set_ylabel("Q(t+1)", fontweight='bold')
axA.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=8)

# Panel B: MOD-03 Transition Matrix
axB = axs[0, 1]
num_bins = len(bin_edges) - 1
axB.plot([bin_edges[0], bin_edges[-1]], [bin_edges[0], bin_edges[-1]], color='#000000', linewidth=1.0)
for i in range(num_bins):
    for j in range(num_bins):
        cnt = matrix[j, i]
        if cnt > 0:
            cx = np.sqrt(bin_edges[i] * bin_edges[i+1])
            cy = np.sqrt(bin_edges[j] * bin_edges[j+1])
            col = '#0000FF' if j > i else ('#FF0000' if j < i else '#000000')
            axB.text(cx, cy, f"{{int(cnt)}}", ha='center', va='center', fontsize=5, fontweight='bold', color=col)
axB.set_xscale('log'); axB.set_yscale('log')
axB.set_xlim(bin_edges[0], bin_edges[-1]); axB.set_ylim(bin_edges[0], bin_edges[-1])
axB.set_aspect('equal', adjustable='box')
for axis_obj in [axB.xaxis, axB.yaxis]:
    axis_obj.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{{y:g}}"))
axB.grid(True, which='major', color='#000000', linewidth=0.8); axB.grid(True, which='minor', color='#808080', linewidth=0.5, linestyle='--')
axB.set_title("B. Transition Frequency Matrix", loc='left', fontweight='bold', fontsize=11)
axB.set_xlabel("Q(t)", fontweight='bold'); axB.set_ylabel("Q(t+1)", fontweight='bold')

# Panel C: MOD-07 Chronological Raster
axC = axs[1, 0]
pos_min = np.nanmin(grid[grid > 0]) if np.any(grid > 0) else 0.1
pos_max = np.nanmax(grid) if np.any(~np.isnan(grid)) else 1000
cmap_usgs = ListedColormap({USGS_HEX_LIST})
norm_usgs = LogNorm(vmin=pos_min, vmax=pos_max)
imC = axC.imshow(grid, aspect='auto', origin='lower', cmap=cmap_usgs, norm=norm_usgs, extent=[0.5, 366.5, -0.5, len(years) - 0.5])
cbarC = fig.colorbar(imC, ax=axC, pad=0.02, fraction=0.04)
cbarC.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{{y:g}}"))
axC.set_title("C. Chronological Raster Hydrograph", loc='left', fontweight='bold', fontsize=11)
axC.set_xlabel("Day of Water Year (Oct 1 -> Sep 30)", fontweight='bold'); axC.set_ylabel("Water Year", fontweight='bold')

# Panel D: MOD-10 Annual Threshold Trends
axD = axs[1, 1]
chrono_df = tdf.sort_values(by='water_year')
axD.plot(chrono_df['water_year'], chrono_df['Q10'], color='#0000FF', marker='o', markersize=3, label='Q10')
axD.plot(chrono_df['water_year'], chrono_df['Q50'], color='#FFD700', marker='s', markersize=3, label='Q50')
axD.plot(chrono_df['water_year'], chrono_df['Q90'], color='#FF0000', marker='^', markersize=3, label='Q90')
axD.set_yscale('log')
axD.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{{y:g}}"))
axD.grid(True, which='major', color='#000000', linewidth=0.8); axD.grid(True, which='minor', color='#808080', linewidth=0.5, linestyle='--')
axD.set_title("D. Annual Exceedance Trends", loc='left', fontweight='bold', fontsize=11)
axD.set_xlabel("Water Year", fontweight='bold'); axD.set_ylabel("Discharge Q", fontweight='bold')
axD.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=8)

fig.suptitle("MOD-12: Composite Hydroinformatics Dashboard\\n{station_name}", fontsize=14, fontweight='bold', y=0.98)
"""

def plot_mod12_composite(df: pd.DataFrame, station_name: str = "Streamflow Station"):
    plot_df = df[(df['Q_t'] > 0) & (df['Q_next'] > 0)].copy()
    matrix, bin_edges, _, _ = _compute_transition_bins(df)
    grid, years = _build_raster_grid(df)
    tdf = _compute_annual_thresholds(df)
    
    code = get_mod012_code(station_name)
    local_env = {
        "plot_df": plot_df, "matrix": matrix, "bin_edges": bin_edges,
        "grid": grid, "years": years, "tdf": tdf, "plt": plt,
        "ticker": ticker, "np": np, "ListedColormap": ListedColormap
    }
    exec(code, local_env)
    return local_env.get("fig", plt.gcf()), code