"""Checks for the ML step. Run AFTER train_models.py:  python tests/test_models.py"""
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from train_models import FEATURES, FORBIDDEN, LEVELS, load_data  # noqa: E402

def check(name, condition):
    print(("PASS  " if condition else "FAIL  ") + name)
    assert condition, name

df = load_data()
check("TC-13 no leakage columns used as inputs", not (set(FEATURES) & FORBIDDEN))
check("TC-14 all 4 risk levels exist in the data", df["eal_level"].nunique() == 4)
for fname in ["model_comparison.csv", "confusion_matrix.png", "feature_importance.png",
              "eal_actual_vs_predicted.png", "regression_metrics.csv",
              "xgb_risk_classifier.joblib", "xgb_eal_regressor.joblib"]:
    check(f"TC-15 output exists: {fname}", (ROOT / "outputs" / fname).exists())

clf = joblib.load(ROOT / "outputs" / "xgb_risk_classifier.joblib")
pred = clf.predict(df[FEATURES].head(20))
check("TC-16 classifier predicts valid level codes for 20 samples",
      len(pred) == 20 and set(pred) <= set(range(len(LEVELS))))
print("\nAll tests passed.")
