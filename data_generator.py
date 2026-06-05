"""
data_generator.py
Generates realistic diesel engine sensor time-series data including
normal operation, gradual degradation, and failure events.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_engine_data(n_hours: int = 720, seed: int = 42) -> pd.DataFrame:
    """
    Simulate 720 hours (~30 days) of diesel engine sensor readings.

    Sensors modelled:
      - engine_rpm        : rotational speed (RPM)
      - coolant_temp_c    : coolant temperature (°C)
      - exhaust_temp_c    : exhaust gas temperature (°C)
      - oil_pressure_bar  : lubrication oil pressure (bar)
      - nox_ppm           : NOx emission level (ppm)
      - pm_mg_m3          : particulate matter (mg/m³)
      - vibration_g       : engine block vibration (g)
      - fuel_flow_lph     : fuel flow rate (L/hr)
    """
    np.random.seed(seed)
    freq = 10                          # readings per hour
    n = n_hours * freq

    timestamps = [datetime(2026, 1, 1) + timedelta(minutes=6 * i) for i in range(n)]

    # ── Normal operating baselines ──────────────────────────────────────────
    rpm        = 1800 + 200 * np.sin(np.linspace(0, 20 * np.pi, n)) + np.random.normal(0, 15, n)
    coolant    = 88  + 4   * np.sin(np.linspace(0, 8  * np.pi, n)) + np.random.normal(0, 1, n)
    exhaust    = 420 + 30  * np.sin(np.linspace(0, 12 * np.pi, n)) + np.random.normal(0, 8, n)
    oil_press  = 4.5 - 0.5 * np.sin(np.linspace(0, 6  * np.pi, n)) + np.random.normal(0, 0.1, n)
    nox        = 280 + 20  * np.sin(np.linspace(0, 10 * np.pi, n)) + np.random.normal(0, 10, n)
    pm         = 12  + 2   * np.sin(np.linspace(0, 8  * np.pi, n)) + np.random.normal(0, 0.5, n)
    vibration  = 0.8 + 0.1 * np.sin(np.linspace(0, 15 * np.pi, n)) + np.random.normal(0, 0.05, n)
    fuel_flow  = 45  + 5   * np.sin(np.linspace(0, 10 * np.pi, n)) + np.random.normal(0, 1, n)

    # ── Gradual degradation: 600–680 h ──────────────────────────────────────
    degrad_start, degrad_end = 600 * freq, 680 * freq
    slope = np.linspace(0, 1, degrad_end - degrad_start)
    exhaust[degrad_start:degrad_end] += slope * 60
    nox    [degrad_start:degrad_end] += slope * 80
    pm     [degrad_start:degrad_end] += slope * 8
    vibration[degrad_start:degrad_end] += slope * 0.4
    oil_press[degrad_start:degrad_end] -= slope * 0.8

    # ── Failure events (anomalies) ───────────────────────────────────────────
    anomaly_mask = np.zeros(n, dtype=int)
    for center in [1500, 3800, 5200, 6700]:          # 4 injected failure events
        span = range(max(0, center - 15), min(n, center + 15))
        exhaust  [span] += np.random.uniform(80, 150)
        nox      [span] += np.random.uniform(100, 200)
        pm       [span] += np.random.uniform(10, 25)
        vibration[span] += np.random.uniform(0.6, 1.2)
        oil_press[span] -= np.random.uniform(1.0, 2.0)
        anomaly_mask[span] = 1

    df = pd.DataFrame({
        "timestamp":       timestamps,
        "engine_rpm":      np.clip(rpm,       800, 2500),
        "coolant_temp_c":  np.clip(coolant,   60,  120),
        "exhaust_temp_c":  np.clip(exhaust,   300, 700),
        "oil_pressure_bar":np.clip(oil_press, 0.5, 6.5),
        "nox_ppm":         np.clip(nox,       100, 650),
        "pm_mg_m3":        np.clip(pm,        2,   60),
        "vibration_g":     np.clip(vibration, 0.1, 3.0),
        "fuel_flow_lph":   np.clip(fuel_flow, 20,  80),
        "anomaly_label":   anomaly_mask,
    })
    df.set_index("timestamp", inplace=True)
    return df


if __name__ == "__main__":
    df = generate_engine_data()
    df.to_csv("data/engine_sensor_data.csv")
    print(f"Generated {len(df)} records | Anomalies: {df['anomaly_label'].sum()}")
    print(df.describe().round(2))
