"""
data_generator.py
Generates a synthetic (computer-made) cyber-incident dataset in INDIAN RUPEES.

Each row = one cyber incident that could happen to an organisation.
The loss is built from a FAIR-style logic: bigger vulnerability, more critical
assets, more exposed records and slower detection => bigger loss;
stronger security controls => smaller loss.

Run from the project root:   python src/data_generator.py
Output:                      data/incidents.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42              # same seed => same dataset for every teammate
N_ROWS = 3000          # same size as the base paper's simulated dataset

INDUSTRIES = ["Finance", "Healthcare", "IT_Services", "Manufacturing",
              "Retail", "Education", "Government"]
ORG_SIZES = ["Small", "Medium", "Large"]

# threat type -> (typical base loss in INR, yearly frequency per organisation)
THREATS = {
    "Ransomware":      (2_000_000, 0.60),
    "Phishing":        (  600_000, 2.50),
    "Data_Breach":     (3_000_000, 0.50),
    "DDoS":            (  400_000, 1.20),
    "Insider_Threat":  (1_000_000, 0.40),
    "Malware":         (  800_000, 1.50),
    "Web_App_Attack":  (1_200_000, 0.90),
}
INDUSTRY_MULT = {"Finance": 0.50, "Healthcare": 0.40, "IT_Services": 0.20,
                 "Manufacturing": 0.10, "Retail": 0.00, "Education": -0.20,
                 "Government": 0.30}               # added inside the log
SIZE_MULT = {"Small": -0.40, "Medium": 0.00, "Large": 0.50}


def generate(n_rows: int = N_ROWS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    threat_names = list(THREATS.keys())
    threat = rng.choice(threat_names, size=n_rows)
    industry = rng.choice(INDUSTRIES, size=n_rows)
    size = rng.choice(ORG_SIZES, size=n_rows, p=[0.35, 0.40, 0.25])

    cvss = np.clip(rng.beta(5, 3, n_rows) * 10, 0.5, 10.0)          # 0-10
    asset_crit = rng.integers(1, 6, n_rows)                         # 1-5
    records = np.exp(rng.uniform(np.log(100), np.log(5_000_000), n_rows)).astype(int)
    control_maturity = np.clip(rng.beta(4, 3, n_rows), 0.05, 0.99)  # 0-1
    patch_delay = rng.integers(0, 121, n_rows)                      # days
    detect_hours = np.clip(rng.exponential(72, n_rows), 1, 720)     # hours

    # --- yearly frequency of this kind of event (FAIR: Loss Event Frequency)
    base_freq = np.array([THREATS[t][1] for t in threat])
    freq = base_freq * (0.5 + cvss / 10) * (1.3 - control_maturity) \
           * (1 + patch_delay / 240)
    freq = np.round(freq, 3)

    # --- size of one loss (FAIR: Loss Magnitude), built in log-space
    base_loss = np.array([THREATS[t][0] for t in threat])
    log_loss = (np.log(base_loss)
                + 0.12 * (cvss - 6.0)               # centred so base loss stays typical
                + 0.22 * (asset_crit - 3)
                + 0.10 * (np.log(records) - 8.5)
                + 0.25 * (np.log1p(detect_hours / 24) - 1.1)
                - 1.00 * (control_maturity - 0.57)
                + np.array([INDUSTRY_MULT[i] for i in industry])
                + np.array([SIZE_MULT[s] for s in size])
                + rng.normal(0, 0.45, n_rows))                      # randomness
    loss_inr = np.round(np.exp(log_loss), 0)

    df = pd.DataFrame({
        "incident_id": np.arange(1, n_rows + 1),
        "industry": industry,
        "org_size": size,
        "threat_type": threat,
        "cvss_score": np.round(cvss, 1),
        "asset_criticality": asset_crit,
        "records_exposed": records,
        "control_maturity": np.round(control_maturity, 2),
        "patch_delay_days": patch_delay,
        "detection_hours": np.round(detect_hours, 1),
        "event_frequency_per_year": freq,
        "loss_inr": loss_inr,
    })

    # risk level label (used for classification and for the dashboard)
    bins = [0, 500_000, 2_500_000, 10_000_000, np.inf]
    df["risk_level"] = pd.cut(df["loss_inr"], bins=bins,
                              labels=["Low", "Medium", "High", "Critical"]).astype(str)
    return df


if __name__ == "__main__":
    out_path = Path(__file__).resolve().parent.parent / "data" / "incidents.csv"
    out_path.parent.mkdir(exist_ok=True)
    data = generate()
    data.to_csv(out_path, index=False)
    print(f"Saved {len(data)} rows to {out_path}")
    print("\nFirst 5 rows:")
    print(data.head().to_string(index=False))
    print("\nRisk level counts:")
    print(data["risk_level"].value_counts().to_string())
    print("\nLoss (INR) summary:")
    print(data["loss_inr"].describe().round(0).to_string())
