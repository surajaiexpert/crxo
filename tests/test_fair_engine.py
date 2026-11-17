"""Checks for the Monte Carlo engine. Run: python tests/test_fair_engine.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fair_engine import simulate_annual_loss, risk_metrics, LOSS_SIGMA  # noqa: E402

def check(name, condition):
    print(("PASS  " if condition else "FAIL  ") + name)
    assert condition, name

# TC-08: simulated EAL must be close to the mathematical answer freq * mean loss
freq, median = 1.5, 1_000_000
rng = np.random.default_rng(1)
losses = simulate_annual_loss(freq, median, n_sims=200_000, rng=rng)
expected = freq * median * np.exp(LOSS_SIGMA ** 2 / 2)
eal, var, cvar = risk_metrics(losses)
check(f"TC-08 simulated EAL within 2% of theory ({eal:,.0f} vs {expected:,.0f})",
      abs(eal - expected) / expected < 0.02)
check("TC-09 CVaR is at least as large as VaR", cvar >= var)
check("TC-10 zero frequency gives zero loss",
      simulate_annual_loss(0.0, median, n_sims=1000).sum() == 0)
check("TC-11 higher frequency gives higher EAL",
      risk_metrics(simulate_annual_loss(3.0, median, 50_000, rng=np.random.default_rng(2)))[0]
      > risk_metrics(simulate_annual_loss(0.5, median, 50_000, rng=np.random.default_rng(2)))[0])
check("TC-12 same seed gives identical results",
      np.array_equal(simulate_annual_loss(1, median, 1000, rng=np.random.default_rng(5)),
                     simulate_annual_loss(1, median, 1000, rng=np.random.default_rng(5))))
print("\nAll tests passed.")
