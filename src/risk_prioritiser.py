"""
risk_prioritiser.py
Step 5 of the pipeline: RISK PRIORITISATION.

1. Picks the risk scenarios of one example organisation (default: a Medium-size
   Finance company, 25 scenarios) from the simulated data.
2. Ranks them by Expected Annual Loss (EAL) and shows which few risks cause most of the loss
   (Pareto / 80-20 idea).

Run from the project root:   python src/risk_prioritiser.py
Optional:                    python src/risk_prioritiser.py --industry Healthcare --size Large
Outputs:                     outputs/prioritised_risks.csv , outputs/pareto_chart.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def build_org_portfolio(industry="Finance", org_size="Medium", n=25, seed=7):
    """The risk scenarios of one example organisation."""
    incidents = pd.read_csv(ROOT / "data" / "incidents.csv")
    results = pd.read_csv(ROOT / "data" / "risk_results.csv")
    df = incidents.merge(results, on="incident_id")
    df = df[(df["industry"] == industry) & (df["org_size"] == org_size)]
    if len(df) == 0:
        raise ValueError(f"No scenarios for industry={industry}, size={org_size}")
    if len(df) > n:
        df = df.sample(n=n, random_state=seed)
    cols = ["incident_id", "industry", "org_size", "threat_type", "cvss_score",
            "control_maturity", "eal_inr", "var95_inr", "cvar95_inr", "risk_exposure_score"]
    return df[cols].reset_index(drop=True)


def prioritise(portfolio):
    """Rank scenarios by EAL and add share, cumulative share and priority tier."""
    df = portfolio.sort_values("eal_inr", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    total = df["eal_inr"].sum()
    df["share_pct"] = (100 * df["eal_inr"] / total).round(1)
    df["cumulative_pct"] = df["share_pct"].cumsum().round(1)
    # P1 = first scenarios that make up the first 50% of loss, P2 = up to 80%, P3 = rest
    prev = df["cumulative_pct"] - df["share_pct"]
    df["priority"] = prev.apply(lambda p: "P1" if p < 50 else ("P2" if p < 80 else "P3"))
    return df


def save_pareto_chart(ranked, path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(ranked["rank"], ranked["eal_inr"] / 1e5, color="#1f77b4")
    ax.set_xlabel("Risk scenario rank")
    ax.set_ylabel("EAL (Rs lakh)")
    ax2 = ax.twinx()
    ax2.plot(ranked["rank"], ranked["cumulative_pct"], color="#d62728", marker="o")
    ax2.set_ylabel("Cumulative share of total loss (%)")
    ax2.set_ylim(0, 105)
    ax.set_title("Risk prioritisation (Pareto chart)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--industry", default="Finance")
    parser.add_argument("--size", default="Medium")
    args = parser.parse_args()

    (ROOT / "outputs").mkdir(exist_ok=True)
    ranked = prioritise(build_org_portfolio(args.industry, args.size))
    ranked.to_csv(ROOT / "outputs" / "prioritised_risks.csv", index=False)
    save_pareto_chart(ranked, ROOT / "outputs" / "pareto_chart.png")

    print(f"Organisation: {args.size} {args.industry}   Scenarios: {len(ranked)}")
    print(f"Total EAL: Rs {ranked['eal_inr'].sum():,.0f}\n")
    print(ranked[["rank", "incident_id", "threat_type", "eal_inr", "share_pct",
                  "cumulative_pct", "priority"]].head(10).to_string(index=False))
    print("\nEAL by threat type (Rs):")
    print(ranked.groupby("threat_type")["eal_inr"].sum().sort_values(ascending=False)
          .round(0).apply(lambda v: f"{v:,.0f}").to_string())
    print("\nSaved outputs/prioritised_risks.csv and outputs/pareto_chart.png")
