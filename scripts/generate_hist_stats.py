import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import compute_groundwater_stress, compute_agricultural_risk, compute_resilience_score
import pandas as pd
import numpy as np

df = pd.read_csv("data/historical_climate_data.csv")

gw_stress_vals = []
ag_risk_vals = []
resilience_vals = []

for _, row in df.iterrows():
    gw = compute_groundwater_stress(
        row["reservoir_storage_pct"], row["rainfall_ond"], row["rainfall_jjas"],
        445.47, 311.15, 445.47
    )
    ag = compute_agricultural_risk(
        row["rice_yield"], 3436.90,
        row["rainfall_ond"], row["rainfall_jjas"],
        445.47, 311.15
    )
    rs = compute_resilience_score(
        row["rice_yield"], row["reservoir_storage_pct"],
        ag, 3436.90, 45.33
    )
    gw_stress_vals.append(gw)
    ag_risk_vals.append(ag)
    resilience_vals.append(rs)

print("groundwater_stress mean:", round(np.mean(gw_stress_vals), 4))
print("groundwater_stress std:", round(np.std(gw_stress_vals), 4))
print("groundwater_stress min:", round(np.min(gw_stress_vals), 4))
print("groundwater_stress max:", round(np.max(gw_stress_vals), 4))
print()
print("agricultural_risk mean:", round(np.mean(ag_risk_vals), 4))
print("agricultural_risk std:", round(np.std(ag_risk_vals), 4))
print("agricultural_risk min:", round(np.min(ag_risk_vals), 4))
print("agricultural_risk max:", round(np.max(ag_risk_vals), 4))
print()
print("regional_resilience mean:", round(np.mean(resilience_vals), 4))
print("regional_resilience std:", round(np.std(resilience_vals), 4))
print("regional_resilience min:", round(np.min(resilience_vals), 4))
print("regional_resilience max:", round(np.max(resilience_vals), 4))