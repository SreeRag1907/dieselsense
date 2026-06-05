# 🔧 DieselSense — AI-Powered Engine Condition & Emission Monitor

> **Predictive condition monitoring system for high-power diesel engines using Python, machine learning, and time-series analytics — built to detect failures and emission breaches before they occur.**

---

## 📋 Overview

DieselSense simulates and analyses 720 hours of diesel engine sensor data to:

- **Detect anomalies** (failures, emission spikes) using Isolation Forest + Rolling Z-Score
- **Predict failures** using a supervised Random Forest classifier (F1 = 1.00 on test set)
- **Track emission levels** (NOx, PM) against regulatory limits in real time
- **Compute a composite Degradation Index** as an early warning signal
- **Visualise all insights** through 6 publication-quality charts

This project maps directly to Rolls-Royce Power Systems' condition monitoring work on emission-reduction equipment for diesel engines.

---

## 🗂️ Project Structure

```
dieselsense/
├── data_generator.py        # Simulates realistic engine sensor time-series
├── time_series_analyzer.py  # Rolling stats, trend detection, threshold alerts
├── anomaly_detector.py      # Isolation Forest, Z-Score, Random Forest ML models
├── visualize.py             # Generates all 6 analysis plots
├── plots/                   # Output charts
│   ├── 01_sensor_overview.png
│   ├── 02_anomaly_comparison.png
│   ├── 03_emissions.png
│   ├── 04_degradation_index.png
│   ├── 05_feature_importance.png
│   └── 06_correlation_heatmap.png
├── requirements.txt
└── README.md
```

---

## 🔬 Sensors Modelled

| Sensor | Unit | Description |
|---|---|---|
| `engine_rpm` | RPM | Rotational speed |
| `coolant_temp_c` | °C | Coolant temperature |
| `exhaust_temp_c` | °C | Exhaust gas temperature |
| `oil_pressure_bar` | bar | Lubrication oil pressure |
| `nox_ppm` | ppm | NOx emission level |
| `pm_mg_m3` | mg/m³ | Particulate matter |
| `vibration_g` | g | Engine block vibration |
| `fuel_flow_lph` | L/hr | Fuel consumption rate |

---

## 🤖 ML Models & Results

| Model | Type | Precision | Recall | F1 |
|---|---|---|---|---|
| Isolation Forest | Unsupervised | 0.556 | **1.000** | 0.714 |
| Rolling Z-Score | Statistical | 0.157 | 0.208 | 0.179 |
| Random Forest | Supervised | **1.000** | **1.000** | **1.000** |

> **Key insight:** Isolation Forest achieves perfect recall (catches every real failure) with no labelled data — making it ideal for real-world deployment where failure labels are scarce.

---

## 📊 Key Findings

- **Exhaust temperature** is the strongest predictor of engine failure (importance: 0.231)
- **NOx and PM** are leading indicators of emission breaches — spikes precede oil pressure drops
- **Degradation Index** rises detectably ~15 hours before threshold breach events
- Correlation analysis shows strong positive correlation between `exhaust_temp_c` and `nox_ppm` (r ≈ 0.78)

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate data + run all models + save plots
python visualize.py

# Run just the ML pipeline
python anomaly_detector.py

# Check threshold alerts
python time_series_analyzer.py
```

---

## 🛠️ Tech Stack

- **Python** — core language
- **pandas / NumPy / SciPy** — time-series processing and statistics
- **scikit-learn** — Isolation Forest, Random Forest, StandardScaler
- **matplotlib / seaborn** — visualisation

---

## 📌 Relevance to Industrial Condition Monitoring

The pipeline mirrors real-world condition monitoring workflows:

1. **Sensor ingestion** → streaming multi-channel time-series
2. **Statistical analysis** → rolling baselines, trend detection, threshold alerting
3. **ML anomaly detection** → unsupervised Isolation Forest for zero-label deployment
4. **Supervised classification** → Random Forest when labelled failure history exists
5. **Composite health scoring** → Degradation Index for maintenance scheduling

---

*Built by C Sreerag as part of a portfolio project aligned with industrial data analytics and predictive maintenance.*
