"""
visualize.py
Generates publication-quality plots for DieselSense.

Charts produced:
  1. Full sensor time-series overview (8 panels)
  2. Anomaly detection comparison (IF vs Z-Score vs Ground Truth)
  3. Emission levels (NOx + PM) with threshold bands
  4. Degradation index over time
  5. Feature importance bar chart (Random Forest)
  6. Sensor correlation heatmap
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns

from data_generator import generate_engine_data
from anomaly_detector import (
    isolation_forest_detector,
    rolling_zscore_detector,
    random_forest_classifier,
    FEATURE_COLS,
)
from time_series_analyzer import (
    degradation_index,
    sensor_correlation,
    THRESHOLDS,
)

os.makedirs("plots", exist_ok=True)

PALETTE = {
    "blue":   "#1B3A6B",
    "red":    "#C0392B",
    "green":  "#1A7A4A",
    "orange": "#E67E22",
    "gray":   "#7F8C8D",
    "light":  "#ECF0F1",
}

plt.rcParams.update({
    "font.family":    "sans-serif",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":     True,
    "grid.alpha":    0.3,
    "grid.linestyle": "--",
})


# ── 1. Sensor overview ───────────────────────────────────────────────────────

def plot_sensor_overview(df):
    fig, axes = plt.subplots(4, 2, figsize=(16, 14), sharex=True)
    sensors = [
        ("engine_rpm",        "Engine RPM",              "RPM",    PALETTE["blue"]),
        ("coolant_temp_c",    "Coolant Temperature",     "°C",     PALETTE["orange"]),
        ("exhaust_temp_c",    "Exhaust Gas Temperature", "°C",     PALETTE["red"]),
        ("oil_pressure_bar",  "Oil Pressure",            "bar",    PALETTE["green"]),
        ("nox_ppm",           "NOx Emissions",           "ppm",    PALETTE["red"]),
        ("pm_mg_m3",          "Particulate Matter (PM)", "mg/m³",  PALETTE["orange"]),
        ("vibration_g",       "Engine Vibration",        "g",      PALETTE["blue"]),
        ("fuel_flow_lph",     "Fuel Flow Rate",          "L/hr",   PALETTE["green"]),
    ]
    hours = np.linspace(0, 720, len(df))

    for ax, (col, title, unit, color) in zip(axes.flat, sensors):
        ax.plot(hours, df[col], color=color, lw=0.8, alpha=0.85)
        # shade anomaly regions
        anom = df["anomaly_label"].values
        for i in range(1, len(anom)):
            if anom[i] == 1:
                ax.axvspan(hours[i - 1], hours[i], color=PALETTE["red"], alpha=0.15)
        ax.set_title(title, fontsize=11, fontweight="bold", color="#1A1A1A")
        ax.set_ylabel(unit, fontsize=9, color=PALETTE["gray"])

    axes[-1, 0].set_xlabel("Operating Hours", fontsize=10)
    axes[-1, 1].set_xlabel("Operating Hours", fontsize=10)
    fig.suptitle("DieselSense — Full Sensor Time-Series (720 hrs)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("plots/01_sensor_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 01_sensor_overview.png")


# ── 2. Anomaly detection comparison ─────────────────────────────────────────

def plot_anomaly_comparison(df, df_if, df_z):
    hours = np.linspace(0, 720, len(df))
    fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)

    def shade_flags(ax, flags, color, alpha=0.4):
        for i in range(1, len(flags)):
            if flags[i] == 1:
                ax.axvspan(hours[i - 1], hours[i], color=color, alpha=alpha)

    # Panel 1: NOx signal
    axes[0].plot(hours, df["nox_ppm"], color=PALETTE["blue"], lw=0.8)
    axes[0].axhline(THRESHOLDS["nox_ppm"], color=PALETTE["red"], lw=1.5, linestyle="--", label="Threshold")
    axes[0].set_ylabel("NOx (ppm)"); axes[0].set_title("NOx Signal", fontweight="bold")

    # Panel 2: Ground truth
    shade_flags(axes[1], df["anomaly_label"].values, PALETTE["red"])
    axes[1].set_yticks([]); axes[1].set_title("Ground-Truth Anomalies", fontweight="bold")
    axes[1].set_facecolor("#FDFCFC")

    # Panel 3: Isolation Forest
    shade_flags(axes[2], df_if["if_anomaly"].values, PALETTE["orange"])
    axes[2].set_yticks([]); axes[2].set_title("Isolation Forest Detections", fontweight="bold")
    axes[2].set_facecolor("#FDFCFC")

    # Panel 4: Z-Score
    shade_flags(axes[3], df_z["zscore_anomaly"].values, PALETTE["green"], alpha=0.35)
    axes[3].set_yticks([]); axes[3].set_title("Rolling Z-Score Detections", fontweight="bold")
    axes[3].set_facecolor("#FDFCFC")
    axes[3].set_xlabel("Operating Hours")

    fig.suptitle("DieselSense — Anomaly Detection Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/02_anomaly_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 02_anomaly_comparison.png")


# ── 3. Emission levels ───────────────────────────────────────────────────────

def plot_emissions(df):
    hours = np.linspace(0, 720, len(df))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 7), sharex=True)

    # NOx
    ax1.plot(hours, df["nox_ppm"], color=PALETTE["blue"], lw=0.9, label="NOx")
    ax1.axhline(THRESHOLDS["nox_ppm"], color=PALETTE["red"], lw=2, linestyle="--", label=f"Limit ({THRESHOLDS['nox_ppm']} ppm)")
    ax1.fill_between(hours, THRESHOLDS["nox_ppm"], df["nox_ppm"],
                     where=df["nox_ppm"] > THRESHOLDS["nox_ppm"],
                     color=PALETTE["red"], alpha=0.2, label="Breach")
    ax1.set_ylabel("NOx (ppm)", fontsize=10); ax1.legend(fontsize=9)
    ax1.set_title("NOx Emission Monitoring", fontweight="bold")

    # PM
    ax2.plot(hours, df["pm_mg_m3"], color=PALETTE["orange"], lw=0.9, label="PM")
    ax2.axhline(THRESHOLDS["pm_mg_m3"], color=PALETTE["red"], lw=2, linestyle="--", label=f"Limit ({THRESHOLDS['pm_mg_m3']} mg/m³)")
    ax2.fill_between(hours, THRESHOLDS["pm_mg_m3"], df["pm_mg_m3"],
                     where=df["pm_mg_m3"] > THRESHOLDS["pm_mg_m3"],
                     color=PALETTE["red"], alpha=0.2, label="Breach")
    ax2.set_ylabel("PM (mg/m³)", fontsize=10); ax2.set_xlabel("Operating Hours"); ax2.legend(fontsize=9)
    ax2.set_title("Particulate Matter Monitoring", fontweight="bold")

    fig.suptitle("DieselSense — Emission Level Tracking vs Regulatory Limits", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/03_emissions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 03_emissions.png")


# ── 4. Degradation index ─────────────────────────────────────────────────────

def plot_degradation(df):
    hours = np.linspace(0, 720, len(df))
    deg   = degradation_index(df).values

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(hours, deg, color=PALETTE["blue"], lw=1)
    ax.fill_between(hours, 0, deg, alpha=0.15, color=PALETTE["blue"])
    ax.axhline(0.75, color=PALETTE["red"], lw=1.5, linestyle="--", label="Warning threshold (0.75)")
    ax.axhline(0.90, color="#8B0000", lw=1.5, linestyle=":",  label="Critical threshold (0.90)")
    ax.set_xlabel("Operating Hours"); ax.set_ylabel("Degradation Index (0–1)")
    ax.set_title("DieselSense — Composite Degradation Index", fontsize=14, fontweight="bold")
    ax.legend(); ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig("plots/04_degradation_index.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 04_degradation_index.png")


# ── 5. Feature importance ────────────────────────────────────────────────────

def plot_feature_importance(importance):
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [PALETTE["red"] if v > 0.18 else PALETTE["blue"] for v in importance.values]
    bars = ax.barh(importance.index, importance.values, color=colors, edgecolor="white", height=0.6)
    ax.set_xlabel("Importance Score"); ax.set_title("Random Forest — Feature Importance", fontsize=13, fontweight="bold")
    for bar, val in zip(bars, importance.values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig("plots/05_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 05_feature_importance.png")


# ── 6. Correlation heatmap ───────────────────────────────────────────────────

def plot_correlation(df):
    corr = sensor_correlation(df)
    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                linewidths=0.5, ax=ax, mask=False, vmin=-1, vmax=1)
    ax.set_title("DieselSense — Sensor Correlation Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/06_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 06_correlation_heatmap.png")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating data...")
    df = generate_engine_data()

    print("Running models...")
    df_if, _, _ = isolation_forest_detector(df)
    df_z        = rolling_zscore_detector(df)
    _, _, _, importance, _ = random_forest_classifier(df)

    print("Rendering plots...")
    plot_sensor_overview(df)
    plot_anomaly_comparison(df, df_if, df_z)
    plot_emissions(df)
    plot_degradation(df)
    plot_feature_importance(importance)
    plot_correlation(df)
    print("\nAll plots saved to /plots/")
