"""Simple checks for the generated dataset. Run: python tests/test_data_generator.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data_generator import generate  # noqa: E402

df = generate()

def check(name, condition):
    print(("PASS  " if condition else "FAIL  ") + name)
    assert condition, name

check("TC-01 dataset has 3000 rows", len(df) == 3000)
check("TC-02 no missing values", df.isna().sum().sum() == 0)
check("TC-03 CVSS between 0 and 10", df["cvss_score"].between(0, 10).all())
check("TC-04 control maturity between 0 and 1", df["control_maturity"].between(0, 1).all())
check("TC-05 all losses are positive", (df["loss_inr"] > 0).all())
check("TC-06 all 4 risk levels present", df["risk_level"].nunique() == 4)
check("TC-07 same seed gives identical data", generate().equals(df))
print("\nAll tests passed.")
