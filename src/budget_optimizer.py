"""
budget_optimizer.py
Steps 6-8 of the pipeline: INVESTMENT OPTIONS -> BUDGET CONSTRAINT -> INVESTMENT ALLOCATION.

Question answered: "I can spend at most Rs B per year on security. Which controls
should I buy so that my Expected Annual Loss (EAL) falls the most?"

How it works (simple version):
  * Each control (see data/controls.csv) has a yearly cost and a % reduction for each threat type.
  * If several chosen controls fight the same threat, their effects combine with
    diminishing returns:  total reduction = 1 - (1-a)*(1-b)*...   (never more than 100%).
  * With only 9 controls there are 2^9 = 512 possible shopping lists, so we simply
    CHECK ALL OF THEM and keep the best one that fits the budget. That is the true optimum.
  * We also run a quick "greedy" method (best value-for-money first) to compare.

NOTE: the control costs and effectiveness values in controls.csv are ILLUSTRATIVE ASSUMPTIONS
for this academic prototype. Replace them with sourced values when available.

Run from the project root:   python src/budget_optimizer.py
Optional:                    python src/budget_optimizer.py --budget 40 --industry Finance --size Medium
                             (--budget is in lakh rupees)
"""
import argparse
import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from risk_prioritiser import build_org_portfolio

ROOT = Path(__file__).resolve().parent.parent
LAKH = 100_000


def load_controls():
    return pd.read_csv(ROOT / "data" / "controls.csv")


def eal_by_threat(portfolio):
    return portfolio.groupby("threat_type")["eal_inr"].sum()


def risk_reduction(threat_eal, controls, chosen):
    """Rupees of yearly loss removed by the chosen controls (chosen = list of row positions)."""
    if len(chosen) == 0:
        return 0.0
    total = 0.0
    for threat, eal in threat_eal.items():
        remaining = np.prod([1 - controls.iloc[i][threat] for i in chosen])
        total += eal * (1 - remaining)
    return float(total)


def optimise(portfolio, controls, budget):
    """Best shopping list within budget, found by checking every combination."""
    threat_eal = eal_by_threat(portfolio)
    costs = controls["yearly_cost_inr"].to_numpy()
    best, best_reduction, best_cost = [], 0.0, 0.0
    for mask in itertools.product([0, 1], repeat=len(controls)):
        chosen = [i for i, m in enumerate(mask) if m]
        cost = costs[chosen].sum() if chosen else 0.0
        if cost > budget:
            continue
        reduction = risk_reduction(threat_eal, controls, chosen)
        if reduction > best_reduction + 1e-9 or (abs(reduction - best_reduction) <= 1e-9 and cost < best_cost):
            best, best_reduction, best_cost = chosen, reduction, cost
    return best, best_reduction, best_cost


def greedy(portfolio, controls, budget):
    """Quick method: keep adding the control with the best (extra loss removed / cost)."""
    threat_eal = eal_by_threat(portfolio)
    costs = controls["yearly_cost_inr"].to_numpy()
    chosen, spent = [], 0.0
    while True:
        base = risk_reduction(threat_eal, controls, chosen)
        best_i, best_ratio = None, 0.0
        for i in range(len(controls)):
            if i in chosen or spent + costs[i] > budget:
                continue
            gain = risk_reduction(threat_eal, controls, chosen + [i]) - base
            if gain / costs[i] > best_ratio:
                best_i, best_ratio = i, gain / costs[i]
        if best_i is None:
            break
        chosen.append(best_i)
        spent += costs[best_i]
    return chosen, risk_reduction(threat_eal, controls, chosen), spent


def summarise(portfolio, controls, chosen, reduction, cost):
    total = portfolio["eal_inr"].sum()
    return {
        "controls": [controls.iloc[i]["control_name"] for i in chosen],
        "cost_inr": cost,
        "eal_before_inr": total,
        "eal_after_inr": total - reduction,
        "reduction_inr": reduction,
        "reduction_pct": 100 * reduction / total if total else 0.0,
        # Return on security investment: (loss avoided - cost) / cost
        "rosi": (reduction - cost) / cost if cost else 0.0,
    }


def budget_sweep(portfolio, controls, budgets):
    rows = []
    for b in budgets:
        chosen, red, cost = optimise(portfolio, controls, b)
        rows.append({"budget_lakh": b / LAKH, "spent_lakh": cost / LAKH,
                     "reduction_lakh": red / LAKH, "controls_chosen": len(chosen)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=float, default=40, help="yearly budget in lakh rupees")
    parser.add_argument("--industry", default="Finance")
    parser.add_argument("--size", default="Medium")
    args = parser.parse_args()
    budget = args.budget * LAKH

    (ROOT / "outputs").mkdir(exist_ok=True)
    portfolio = build_org_portfolio(args.industry, args.size)
    controls = load_controls()

    print(f"Organisation: {args.size} {args.industry}  |  Scenarios: {len(portfolio)}  |  "
          f"Budget: Rs {budget:,.0f}")
    print(f"EAL by threat type (Rs):\n{eal_by_threat(portfolio).round(0).apply(lambda v: f'{v:,.0f}').to_string()}\n")

    results = {}
    for label, func in [("OPTIMAL (checks all combinations)", optimise), ("GREEDY (value-for-money first)", greedy)]:
        chosen, red, cost = func(portfolio, controls, budget)
        s = summarise(portfolio, controls, chosen, red, cost)
        results[label] = s
        print(f"=== {label} ===")
        for name in s["controls"]:
            print(f"  + {name}")
        print(f"  Spend           : Rs {s['cost_inr']:,.0f}")
        print(f"  EAL before/after: Rs {s['eal_before_inr']:,.0f} -> Rs {s['eal_after_inr']:,.0f}")
        print(f"  Risk reduced    : Rs {s['reduction_inr']:,.0f} ({s['reduction_pct']:.1f}%)")
        print(f"  ROSI            : {s['rosi']:.2f}\n")

    opt_chosen, _, _ = optimise(portfolio, controls, budget)
    pd.DataFrame([{"control_id": controls.iloc[i]["control_id"],
                   "control_name": controls.iloc[i]["control_name"],
                   "yearly_cost_inr": controls.iloc[i]["yearly_cost_inr"]} for i in opt_chosen]
                 ).to_csv(ROOT / "outputs" / "optimal_allocation.csv", index=False)

    sweep = budget_sweep(portfolio, controls, [x * LAKH for x in range(0, 101, 5)])
    sweep.to_csv(ROOT / "outputs" / "budget_sweep.csv", index=False)
    plt.figure(figsize=(7, 4.5))
    plt.plot(sweep["budget_lakh"], sweep["reduction_lakh"], marker="o")
    plt.xlabel("Yearly security budget (Rs lakh)")
    plt.ylabel("Loss reduced (Rs lakh per year)")
    plt.title("What each extra rupee of budget buys")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ROOT / "outputs" / "budget_vs_reduction.png", dpi=150)
    plt.close()
    print("Saved outputs/optimal_allocation.csv, budget_sweep.csv, budget_vs_reduction.png")
