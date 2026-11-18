"""Checks for prioritiser and optimizer. Run: python tests/test_optimizer.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from budget_optimizer import LAKH, eal_by_threat, greedy, load_controls, optimise, risk_reduction  # noqa: E402
from risk_prioritiser import build_org_portfolio, prioritise  # noqa: E402

def check(name, condition):
    print(("PASS  " if condition else "FAIL  ") + name)
    assert condition, name

portfolio = build_org_portfolio()
controls = load_controls()
ranked = prioritise(portfolio)
threat_eal = eal_by_threat(portfolio)

check("TC-17 ranking is sorted by EAL (highest first)", ranked["eal_inr"].is_monotonic_decreasing)
check("TC-18 shares add up to about 100%", abs(ranked["share_pct"].sum() - 100) < 1.0)
check("TC-19 no controls means zero reduction", risk_reduction(threat_eal, controls, []) == 0)
_, red0, cost0 = optimise(portfolio, controls, 0)
check("TC-20 zero budget buys nothing", red0 == 0 and cost0 == 0)
chosen, red, cost = optimise(portfolio, controls, 40 * LAKH)
check("TC-21 chosen controls fit the budget", cost <= 40 * LAKH)
_, g_red, _ = greedy(portfolio, controls, 40 * LAKH)
check("TC-22 optimal is at least as good as greedy", red >= g_red - 1e-6)
_, red_big, _ = optimise(portfolio, controls, 80 * LAKH)
check("TC-23 bigger budget never reduces less", red_big >= red)
a, b = 0.5, 0.4
check("TC-24 two controls combine with diminishing returns", 1 - (1 - a) * (1 - b) < a + b)
check("TC-25 reduction never exceeds total EAL", red_big <= portfolio["eal_inr"].sum())
print("\nAll tests passed.")
