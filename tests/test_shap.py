"""Checks for the SHAP step. Run AFTER explain_shap.py:  python tests/test_shap.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from explain_shap import group_columns, original_feature_name, to_per_class  # noqa: E402

def check(name, condition):
    print(("PASS  " if condition else "FAIL  ") + name)
    assert condition, name

check("TC-26 one-hot name maps back to its feature",
      original_feature_name("cat__threat_type_Ransomware") == "threat_type"
      and original_feature_name("cat__industry_IT_Services") == "industry")
check("TC-27 numeric name is kept", original_feature_name("num__cvss_score") == "cvss_score")

values = np.array([[1.0, 2.0, 3.0, 4.0]])
grouped = group_columns(values, ["num__cvss_score", "cat__org_size_Small",
                                 "cat__org_size_Large", "num__control_maturity"])
check("TC-28 one-hot columns are added together",
      grouped["org_size"].iloc[0] == 5.0 and grouped["cvss_score"].iloc[0] == 1.0)
check("TC-29 SHAP output shape is normalised (new and old format)",
      len(to_per_class(np.zeros((5, 3, 4)), 4)) == 4
      and len(to_per_class([np.zeros((5, 3))] * 4, 4)) == 4)

imp = pd.read_csv(ROOT / "outputs" / "shap_importance.csv", index_col=0)
check("TC-30 importance file lists all 10 inputs", len(imp) == 10)
check("TC-31 importances are positive", (imp["mean_abs_shap"] >= 0).all() and imp["mean_abs_shap"].sum() > 0)
for f in ["shap_global_importance.png", "shap_summary_critical.png"]:
    check(f"TC-32 output exists: {f}", (ROOT / "outputs" / f).exists())
check("TC-33 at least one local explanation chart exists",
      len(list((ROOT / "outputs").glob("shap_local_incident_*.png"))) >= 1)
print("\nAll tests passed.")
