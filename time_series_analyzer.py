"""
time_series_analyzer.py
Performs time-series analysis on engine sensor data:
  - Rolling statistics (mean, std, min, max)
  - Trend detection via linear regression on rolling windows
  - Rule-based emission threshold alerts (NOx, PM)
  - Correlation analysis between sensors
"""

import pandas as pd
import numpy as np
from scipy import stats


# ── Emission & sensor thresholds (EU Stage V / RR internal limits) ──────────
THRESHOLDS = {
    "nox_ppm":          400,
    "pm_mg_m3":         25,
    "exhaust_temp_c":   580,
    "vibration_g":      1.8,
    "oil_pressure_bar": 1.5,   # min pressure; alert if below
}


def rolling_stats(df: pd.DataFrame, window: str = "1h") -> pd.DataFrame:
    """Compute rolling mean, std, min, max for all numeric sensors."""
    numeric = df.drop(columns=["anomaly_label"], errors="ignore")
    rolled = numeric.rolling(window)
    stats_df = pd.concat([
        rolled.mean().add_suffix("_mean"),
        rolled.std() .add_suffix("_std"),
        rolled.min() .add_suffix("_min"),
        rolled.max() .add_suffix("_max"),
    ], axis=1)
    return stats_df


def detect_trends(df: pd.DataFrame, sensor: str, window: int = 100) -> pd.Series:
    """
    Slide a window across the time series and compute the linear slope.
    Positive slope → rising trend; negative → falling.
    """
    values = df[sensor].values
    slopes = np.full(len(values), np.nan)
    for i in range(window, len(values)):
        segment = values[i - window: i]
        x = np.arange(window)
        slope, *_ = stats.linregress(x, segment)
        slopes[i] = slope
    return pd.Series(slopes, index=df.index, name=f"{sensor}_slope")


def threshold_alerts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a DataFrame of threshold-breach events with timestamp,
    sensor name, measured value, and limit.
    """
    events = []
    for col, limit in THRESHOLDS.items():
        if col not in df.columns:
            continue
        if col == "oil_pressure_bar":
            breaches = df[df[col] < limit][[col]]
        else:
            breaches = df[df[col] > limit][[col]]

        for ts, row in breaches.iterrows():
            events.append({
                "timestamp": ts,
                "sensor":    col,
                "value":     round(row[col], 3),
                "limit":     limit,
                "breach":    "BELOW" if col == "oil_pressure_bar" else "ABOVE",
            })

    alerts = pd.DataFrame(events)
    if not alerts.empty:
        alerts.sort_values("timestamp", inplace=True)
    return alerts


def sensor_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix for all numeric sensors."""
    return df.drop(columns=["anomaly_label"], errors="ignore").corr().round(3)


def degradation_index(df: pd.DataFrame) -> pd.Series:
    """
    Composite degradation score (0–1) from z-score normalised sensors.
    Higher score → higher likelihood of impending failure.
    """
    sensors = ["exhaust_temp_c", "nox_ppm", "pm_mg_m3", "vibration_g"]
    z_scores = df[sensors].apply(lambda col: (col - col.mean()) / col.std())
    # Weight: emission sensors weighted higher
    weights = np.array([1.5, 2.0, 1.5, 1.0])
    score = (z_scores * weights).sum(axis=1) / weights.sum()
    # Normalise to 0–1
    score = (score - score.min()) / (score.max() - score.min())
    return score.rename("degradation_index")


if __name__ == "__main__":
    from data_generator import generate_engine_data
    import os; os.makedirs("data", exist_ok=True)

    df = generate_engine_data()

    alerts = threshold_alerts(df)
    print(f"\n── Threshold Alerts ({len(alerts)} breaches) ──")
    print(alerts.head(10).to_string(index=False))

    corr = sensor_correlation(df)
    print("\n── Sensor Correlation Matrix ──")
    print(corr)

    deg = degradation_index(df)
    print(f"\n── Degradation Index: max={deg.max():.3f} | mean={deg.mean():.3f} ──")
