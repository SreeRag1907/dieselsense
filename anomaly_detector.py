"""
anomaly_detector.py
ML-based anomaly detection for engine condition monitoring.

Models:
  1. Isolation Forest  – unsupervised; no labels needed
  2. Rolling Z-Score   – statistical baseline comparison
  3. (Optional) Random Forest classifier when labels are available

Output:
  - Predicted anomaly flags
  - Anomaly scores (continuous)
  - Precision / Recall / F1 against injected ground-truth labels
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


FEATURE_COLS = [
    "engine_rpm", "coolant_temp_c", "exhaust_temp_c",
    "oil_pressure_bar", "nox_ppm", "pm_mg_m3",
    "vibration_g", "fuel_flow_lph",
]


# ── 1. Isolation Forest ──────────────────────────────────────────────────────

def isolation_forest_detector(df: pd.DataFrame, contamination: float = 0.03):
    """
    Train Isolation Forest on sensor readings.
    Returns the original df with two new columns:
      - if_score:   raw anomaly score (lower = more anomalous)
      - if_anomaly: 1 if anomaly, 0 if normal
    """
    X = df[FEATURE_COLS].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    scores   = model.decision_function(X_scaled)   # higher = more normal
    preds    = model.predict(X_scaled)              # -1 = anomaly, 1 = normal

    result = df.copy()
    result["if_score"]   = scores
    result["if_anomaly"] = (preds == -1).astype(int)
    return result, model, scaler


# ── 2. Rolling Z-Score ───────────────────────────────────────────────────────

def rolling_zscore_detector(df: pd.DataFrame, window: int = 60, threshold: float = 3.0):
    """
    Flag readings where any sensor deviates > threshold std devs
    from its rolling mean.
    """
    result = df.copy()
    result["zscore_anomaly"] = 0

    for col in FEATURE_COLS:
        roll_mean = df[col].rolling(window, min_periods=1).mean()
        roll_std  = df[col].rolling(window, min_periods=1).std().fillna(1)
        z = (df[col] - roll_mean) / roll_std
        result.loc[z.abs() > threshold, "zscore_anomaly"] = 1

    return result


# ── 3. Supervised Random Forest (uses ground-truth labels) ───────────────────

def random_forest_classifier(df: pd.DataFrame):
    """
    Train a Random Forest on labelled data.
    Returns trained model + evaluation report dict.
    """
    X = df[FEATURE_COLS]
    y = df["anomaly_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)

    report = classification_report(y_test, y_pred, output_dict=True)
    feature_importance = pd.Series(
        clf.feature_importances_, index=FEATURE_COLS
    ).sort_values(ascending=False)

    return clf, scaler, report, feature_importance, (y_test, y_pred)


# ── Evaluation helper ────────────────────────────────────────────────────────

def evaluate_unsupervised(df: pd.DataFrame, pred_col: str) -> dict:
    """Compare unsupervised predictions against ground-truth anomaly_label."""
    y_true = df["anomaly_label"].values
    y_pred = df[pred_col].values

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Precision": round(precision, 3),
        "Recall":    round(recall,    3),
        "F1":        round(f1,        3),
    }


if __name__ == "__main__":
    from data_generator import generate_engine_data

    print("Generating engine data...")
    df = generate_engine_data()

    # Isolation Forest
    print("\n── Isolation Forest ──")
    df_if, _, _ = isolation_forest_detector(df)
    metrics_if = evaluate_unsupervised(df_if, "if_anomaly")
    print(pd.DataFrame([metrics_if]).to_string(index=False))

    # Rolling Z-Score
    print("\n── Rolling Z-Score ──")
    df_z = rolling_zscore_detector(df)
    metrics_z = evaluate_unsupervised(df_z, "zscore_anomaly")
    print(pd.DataFrame([metrics_z]).to_string(index=False))

    # Random Forest (supervised)
    print("\n── Random Forest Classifier ──")
    _, _, report, importance, _ = random_forest_classifier(df)
    print(f"  Precision: {report['1']['precision']:.3f}")
    print(f"  Recall:    {report['1']['recall']:.3f}")
    print(f"  F1-score:  {report['1']['f1-score']:.3f}")
    print("\n  Feature Importance:")
    print(importance.to_string())
