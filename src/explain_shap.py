"""
explain_shap.py
Explainable AI with SHAP: WHY does the model give a scenario a certain risk level?

SHAP gives every input (CVSS score, control maturity, detection time ...) a "credit score":
how much did this input push the prediction up or down?
  * GLOBAL view: which inputs matter most overall.
  * LOCAL view : why one particular scenario got its risk level.

Run from the project root (after train_models.py):
    python src/explain_shap.py
Optional - explain a specific scenario:
    python src/explain_shap.py --incident_id 357
Outputs (in outputs/): shap_importance.csv, shap_global_importance.png,
                       shap_summary_critical.png, shap_local_incident_<id>.png
"""
import argparse
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train_models import CATEGORICAL, FEATURES, LEVELS, load_data

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
SEED = 42
SAMPLE_SIZE = 500          # explaining 500 scenarios is plenty and keeps it fast


def to_per_class(shap_values, n_classes):
    """Return a list with one (rows x features) array per risk level, whatever the SHAP version."""
    if isinstance(shap_values, list):
        return [np.asarray(v) for v in shap_values]
    arr = np.asarray(shap_values)
    if arr.ndim == 3 and arr.shape[2] == n_classes:        # (rows, features, classes)
        return [arr[:, :, k] for k in range(n_classes)]
    if arr.ndim == 3 and arr.shape[0] == n_classes:        # (classes, rows, features)
        return [arr[k] for k in range(n_classes)]
    raise ValueError(f"Unexpected SHAP output shape: {arr.shape}")


def original_feature_name(encoded_name):
    """'cat__threat_type_Ransomware' -> 'threat_type'; 'num__cvss_score' -> 'cvss_score'."""
    name = encoded_name.split("__", 1)[-1]
    for cat in CATEGORICAL:
        if name.startswith(cat + "_"):
            return cat
    return name


def group_columns(values, encoded_names):
    """Add the one-hot pieces of a feature back together (e.g. all threat_type_* columns)."""
    groups = {}
    for j, name in enumerate(encoded_names):
        groups.setdefault(original_feature_name(name), []).append(j)
    return pd.DataFrame({g: values[:, idx].sum(axis=1) for g, idx in groups.items()})


def main():
    import shap                                 # imported here so the helper functions above stay testable

    parser = argparse.ArgumentParser()
    parser.add_argument("--incident_id", type=int, default=None,
                        help="scenario to explain (default: the scenario with the highest EAL)")
    args = parser.parse_args()

    OUT.mkdir(exist_ok=True)
    df = load_data()
    pipe = joblib.load(OUT / "xgb_risk_classifier.joblib")
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    names = list(prep.get_feature_names_out())

    target_id = args.incident_id or int(df.loc[df["eal_inr"].idxmax(), "incident_id"])
    if target_id not in set(df["incident_id"]):
        raise SystemExit(f"incident_id {target_id} not found")
    sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=SEED)
    if target_id not in set(sample["incident_id"]):
        sample = pd.concat([sample, df[df["incident_id"] == target_id]])
    sample = sample.reset_index(drop=True)

    X = prep.transform(sample[FEATURES])
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)

    print(f"Computing SHAP values for {len(sample)} scenarios ... please wait")
    explainer = shap.TreeExplainer(model)
    per_class = to_per_class(explainer.shap_values(X), len(LEVELS))

    # ---------- GLOBAL: average importance of each input over all scenarios and levels
    importance = sum(group_columns(v, names).abs().mean() for v in per_class) / len(per_class)
    importance = importance.sort_values(ascending=False)
    importance.rename("mean_abs_shap").to_csv(OUT / "shap_importance.csv")
    importance.sort_values().plot(kind="barh", figsize=(7, 5), color="#2ca02c")
    plt.xlabel("Average impact on the prediction (mean |SHAP|)")
    plt.title("Which inputs drive the risk level? (global SHAP)")
    plt.tight_layout()
    plt.savefig(OUT / "shap_global_importance.png", dpi=150)
    plt.close()
    print("\nTop inputs (mean |SHAP|):")
    print(importance.round(3).to_string())

    # ---------- Detailed dot plot for the 'Critical' level
    shap.summary_plot(per_class[LEVELS.index("Critical")], X, feature_names=names,
                      max_display=12, show=False)
    plt.title("What pushes a scenario towards 'Critical'")
    plt.savefig(OUT / "shap_summary_critical.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---------- LOCAL: explain one scenario
    pos = int(sample.index[sample["incident_id"] == target_id][0])
    k = int(model.predict(X[pos:pos + 1])[0])
    contrib = group_columns(per_class[k][pos:pos + 1], names).iloc[0]
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(6)
    row = sample.iloc[pos]
    print(f"\nScenario {target_id}: predicted risk level = {LEVELS[k]}  "
          f"(actual level from EAL = {row['eal_level']}, EAL = Rs {row['eal_inr']:,.0f})")
    for feat, val in top.items():
        direction = "pushes TOWARDS" if val > 0 else "pushes AWAY from"
        print(f"  {feat} = {row[feat]}  {direction} '{LEVELS[k]}'  (SHAP {val:+.2f})")

    colors = ["#d62728" if v > 0 else "#1f77b4" for v in top.values[::-1]]
    plt.figure(figsize=(7, 4))
    plt.barh([f"{f} = {row[f]}" for f in top.index[::-1]], top.values[::-1], color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel(f"Push towards '{LEVELS[k]}' (red = towards, blue = away)")
    plt.title(f"Why scenario {target_id} is '{LEVELS[k]}'")
    plt.tight_layout()
    plt.savefig(OUT / f"shap_local_incident_{target_id}.png", dpi=150)
    plt.close()
    print(f"\nSaved SHAP charts and shap_importance.csv in {OUT}")


if __name__ == "__main__":
    main()
