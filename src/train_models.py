"""
train_models.py
Trains and evaluates the ML models of the project.

Task A (classification): predict the RISK LEVEL (Low / Medium / High / Critical)
                         of a scenario, based on its yearly expected loss (EAL).
                         Models compared: XGBoost, Random Forest, SVM.
Task B (regression)    : predict the EAL itself in rupees (XGBoost).

IMPORTANT: loss_inr, EAL, VaR and CVaR are NEVER used as inputs (that would be
"data leakage" = giving the model the answer). Only scenario facts are inputs.

Run from the project root:   python src/train_models.py
Outputs go to the outputs/ folder.
"""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")                      # draw charts into files, no pop-up window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             ConfusionMatrixDisplay, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier, XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
SEED = 42
LEVELS = ["Low", "Medium", "High", "Critical"]
# EAL (rupees) cut-offs between the 4 levels: <5 lakh, <15 lakh, <50 lakh, above
EAL_BINS = [0, 500_000, 1_500_000, 5_000_000, np.inf]

CATEGORICAL = ["industry", "org_size", "threat_type"]
NUMERIC = ["cvss_score", "asset_criticality", "log_records", "control_maturity",
           "patch_delay_days", "detection_hours", "event_frequency_per_year"]
FEATURES = CATEGORICAL + NUMERIC
FORBIDDEN = {"incident_id", "loss_inr", "risk_level", "eal_inr", "var95_inr",
             "cvar95_inr", "risk_exposure_score"}


def load_data():
    incidents = pd.read_csv(ROOT / "data" / "incidents.csv")
    results = pd.read_csv(ROOT / "data" / "risk_results.csv")
    df = incidents.merge(results, on="incident_id")
    df["log_records"] = np.log(df["records_exposed"])
    df["eal_level"] = pd.cut(df["eal_inr"], bins=EAL_BINS, labels=LEVELS).astype(str)
    df["eal_level_code"] = df["eal_level"].map({n: i for i, n in enumerate(LEVELS)})
    return df


def make_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


def make_classifiers():
    return {
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.08,
                                 subsample=0.9, colsample_bytree=0.9,
                                 eval_metric="mlogloss", random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
        "SVM": SVC(kernel="rbf", C=3, gamma="scale"),
    }


def train_classifiers(df):
    X, y = df[FEATURES], df["eal_level_code"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    rows, fitted = [], {}
    for name, model in make_classifiers().items():
        pipe = Pipeline([("prep", make_preprocessor()), ("model", model)])
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rows.append({
            "model": name,
            "cv_mean_accuracy": round(cv_scores.mean(), 4),
            "cv_std": round(cv_scores.std(), 4),
            "test_accuracy": round(accuracy_score(y_test, pred), 4),
            "test_precision_macro": round(precision_score(y_test, pred, average="macro", zero_division=0), 4),
            "test_recall_macro": round(recall_score(y_test, pred, average="macro", zero_division=0), 4),
            "test_f1_macro": round(f1_score(y_test, pred, average="macro", zero_division=0), 4),
            # levels are ordered, so "off by one level" is a much smaller mistake than "off by three"
            "test_within_one_level": round(float(np.mean(np.abs(np.asarray(pred) - np.asarray(y_test)) <= 1)), 4),
        })
        fitted[name] = (pipe, pred)
    table = pd.DataFrame(rows)
    return table, fitted, y_test


def save_classification_outputs(table, fitted, y_test):
    OUT.mkdir(exist_ok=True)
    table.to_csv(OUT / "model_comparison.csv", index=False)

    best_name = table.sort_values("cv_mean_accuracy", ascending=False).iloc[0]["model"]
    best_pipe, best_pred = fitted[best_name]
    cm = confusion_matrix(y_test, best_pred, labels=range(len(LEVELS)))
    ConfusionMatrixDisplay(cm, display_labels=LEVELS).plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion matrix - {best_name} (test set)")
    plt.tight_layout()
    plt.savefig(OUT / "confusion_matrix.png", dpi=150)
    plt.close()

    report = classification_report(y_test, best_pred, target_names=LEVELS, zero_division=0)
    (OUT / "classification_report.txt").write_text(f"Best model: {best_name}\n\n{report}")

    # XGBoost feature importance (which inputs the model relied on most)
    xgb_pipe = fitted["XGBoost"][0]
    names = xgb_pipe.named_steps["prep"].get_feature_names_out()
    imp = pd.Series(xgb_pipe.named_steps["model"].feature_importances_, index=names)
    imp.sort_values().tail(12).plot(kind="barh", figsize=(7, 5), color="#1f77b4")
    plt.title("XGBoost feature importance (top 12)")
    plt.tight_layout()
    plt.savefig(OUT / "feature_importance.png", dpi=150)
    plt.close()

    joblib.dump(xgb_pipe, OUT / "xgb_risk_classifier.joblib")
    return best_name, report


def train_regressor(df):
    X, y = df[FEATURES], np.log(df["eal_inr"])            # model log(EAL), easier to learn
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    pipe = Pipeline([("prep", make_preprocessor()),
                     ("model", XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.08,
                                            subsample=0.9, colsample_bytree=0.9,
                                            random_state=SEED))])
    pipe.fit(X_train, y_train)
    pred_log = pipe.predict(X_test)
    actual, pred = np.exp(y_test), np.exp(pred_log)
    metrics = {
        "r2_log_scale": round(r2_score(y_test, pred_log), 4),
        "mae_inr": round(mean_absolute_error(actual, pred), 0),
        "rmse_inr": round(float(np.sqrt(mean_squared_error(actual, pred))), 0),
        "median_abs_pct_error": round(float(np.median(np.abs(pred - actual) / actual) * 100), 1),
    }
    pd.Series(metrics).to_csv(OUT / "regression_metrics.csv", header=["value"])

    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(actual, pred, s=8, alpha=0.4)
    lim = [min(actual.min(), pred.min()), max(actual.max(), pred.max())]
    plt.plot(lim, lim, "r--", label="perfect prediction")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("Actual EAL (INR)"); plt.ylabel("Predicted EAL (INR)")
    plt.title("EAL prediction - test set"); plt.legend(); plt.tight_layout()
    plt.savefig(OUT / "eal_actual_vs_predicted.png", dpi=150)
    plt.close()

    joblib.dump(pipe, OUT / "xgb_eal_regressor.joblib")
    return metrics


def main():
    OUT.mkdir(exist_ok=True)
    df = load_data()
    print(f"Rows: {len(df)}   Features used: {len(FEATURES)}")
    print("\nRisk level counts (from EAL):")
    print(df["eal_level"].value_counts().reindex(LEVELS).to_string())

    print("\n=== Task A: risk level classification ===")
    table, fitted, y_test = train_classifiers(df)
    best_name, report = save_classification_outputs(table, fitted, y_test)
    print(table.to_string(index=False))
    print(f"\nBest model by cross-validation: {best_name}\n")
    print(report)

    print("=== Task B: EAL prediction (XGBoost regressor) ===")
    for key, value in train_regressor(df).items():
        print(f"  {key}: {value:,}")
    print("\nSaved charts, tables and models in the outputs/ folder.")


if __name__ == "__main__":
    main()
