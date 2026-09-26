import numpy as np
import pandas as pd

def ingest_and_process_streamflow(df: pd.DataFrame, date_col: str, q_col: str) -> pd.DataFrame:
    """
    Applies physical flow bounds, temporal continuity, and core hydroinformatics
    metric calculations according to Visual Data Analytics operational standards.
    """
    data = df[[date_col, q_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values(by=date_col).reset_index(drop=True)
    
    # 1. Enforce physical flow bounds: Q >= 0 (remove negative flows)
    data = data[data[q_col] >= 0].copy()
    
    # 2. Temporal continuity check: enforce strict dt = 1 day without synthetic interpolation
    data = data.set_index(date_col)
    full_date_range = pd.date_range(start=data.index.min(), end=data.index.max(), freq='D')
    data = data.reindex(full_date_range)
    data.index.name = 'date'
    data = data.reset_index()
    
    # 3. Axiomatic Rate of Change: dQ/dt = Q(t+1) - Q(t)
    data['Q_t'] = data[q_col]
    data['Q_next'] = data[q_col].shift(-1)
    data['dQ_dt'] = data['Q_next'] - data['Q_t']
    
    # 4. Water Year & Day of Water Year (DOWY) calculations
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['water_year'] = np.where(data['month'] >= 10, data['year'] + 1, data['year'])
    
    def get_dowy(d):
        wy_start = pd.Timestamp(year=d.year if d.month >= 10 else d.year - 1, month=10, day=1)
        return (d - wy_start).days + 1

    data['DOWY'] = data['date'].apply(get_dowy)
    
    # 5. Baseline empirical exceedance: P(Q_t) = [m / (n + 1)] * 100
    valid_pairs = data.dropna(subset=['Q_t', 'Q_next']).copy()
    valid_pairs['rank'] = valid_pairs['Q_t'].rank(ascending=False, method='first')
    n = len(valid_pairs)
    valid_pairs['exceedance_pct'] = (valid_pairs['rank'] / (n + 1)) * 100.0
    
    return valid_pairs

def generate_log_bins(q_series: pd.Series, delta_log10: float = 0.10) -> np.ndarray:
    """Generates exact delta log10 = 0.10 logarithmic bin edges."""
    q_positive = q_series[q_series > 0]
    min_exp = np.floor(np.log10(q_positive.min()))
    max_exp = np.ceil(np.log10(q_positive.max()))
    log_edges = np.arange(min_exp, max_exp + delta_log10, delta_log10)
    return 10.0 ** log_edges