"""
fair_engine.py
FAIR-style Monte Carlo risk engine.

For every incident scenario in data/incidents.csv we simulate many "possible years":
  1. How many times does the attack happen this year?   -> Poisson(event_frequency_per_year)
  2. How much does each event cost?                      -> Lognormal around loss_inr
  3. Add the costs up -> total loss for that simulated year.
After thousands of simulated years we read off:
  EAL   = average yearly loss (Expected Annual Loss)
  VaR95 = loss that is exceeded in only 5% of years
  CVaR95= average loss in those worst 5% of years
  RES   = Risk Exposure Score, 0-100 (our simple scale of log(EAL))

Run from the project root:   python src/fair_engine.py
Output:                      data/risk_results.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

N_SIMS = 10_000        # number of simulated years per scenario
LOSS_SIGMA = 0.45      # spread of one event's cost (uncertainty), log-space
SEED = 2026


def simulate_annual_loss(freq, median_loss, n_sims=N_SIMS, sigma=LOSS_SIGMA, rng=None):
    """Return an array of n_sims simulated yearly total losses (INR)."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    counts = rng.poisson(freq, size=n_sims)               # events in each simulated year
    total_events = int(counts.sum())
    if total_events == 0:
        return np.zeros(n_sims)
    event_losses = rng.lognormal(np.log(median_loss), sigma, size=total_events)
    year_index = np.repeat(np.arange(n_sims), counts)     # which year each event belongs to
    return np.bincount(year_index, weights=event_losses, minlength=n_sims)


def risk_metrics(annual_losses, confidence=0.95):
    """EAL, VaR and CVaR from simulated yearly losses."""
    eal = annual_losses.mean()
    var = np.quantile(annual_losses, confidence)
    tail = annual_losses[annual_losses >= var]
    cvar = tail.mean() if len(tail) else var
    return eal, var, cvar


def quantify_all(df, n_sims=N_SIMS, seed=SEED):
    rng = np.random.default_rng(seed)
    rows = []
    for rec in df.itertuples(index=False):
        losses = simulate_annual_loss(rec.event_frequency_per_year, rec.loss_inr,
                                      n_sims=n_sims, rng=rng)
        eal, var, cvar = risk_metrics(losses)
        rows.append((rec.incident_id, eal, var, cvar))
    out = pd.DataFrame(rows, columns=["incident_id", "eal_inr", "var95_inr", "cvar95_inr"])

    # Risk Exposure Score: log(EAL) scaled to 0-100 (simple, easy to explain)
    log_eal = np.log(out["eal_inr"].clip(lower=1))
    out["risk_exposure_score"] = (100 * (log_eal - log_eal.min())
                                  / (log_eal.max() - log_eal.min())).round(1)
    return out.round({"eal_inr": 0, "var95_inr": 0, "cvar95_inr": 0})


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    incidents = pd.read_csv(root / "data" / "incidents.csv")
    print(f"Simulating {N_SIMS:,} years for {len(incidents)} scenarios ... please wait")
    results = quantify_all(incidents)
    results.to_csv(root / "data" / "risk_results.csv", index=False)
    print(f"Saved {len(results)} rows to data/risk_results.csv\n")
    print(results.head().to_string(index=False))
    print("\nPortfolio totals (sum over all scenarios):")
    print(f"  Total EAL   : Rs {results['eal_inr'].sum():,.0f}")
    print(f"  Total VaR95 : Rs {results['var95_inr'].sum():,.0f}")
    print(f"  Total CVaR95: Rs {results['cvar95_inr'].sum():,.0f}")
