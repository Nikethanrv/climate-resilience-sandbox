import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from src.utils import compute_groundwater_stress, compute_agricultural_risk

df = pd.read_csv("data/historical_climate_data.csv")

gw_vals = []
ag_vals = []

for _, row in df.iterrows():
    gw = compute_groundwater_stress(
        row["reservoir_storage_pct"], row["rainfall_ond"], row["rainfall_jjas"],
        45.3265, 445.47, 311.15
    )
    ag = compute_agricultural_risk(
        row["rice_yield"], 3436.90,
        row["rainfall_ond"], row["rainfall_jjas"],
        445.47, 311.15
    )
    gw_vals.append(gw)
    ag_vals.append(ag)

# These are 0-1 scale from utils
# UI multiplies gw by 100, ag by 5
print("=== Groundwater (UI scale = *100) ===")
print(f"mean: {round(np.mean(gw_vals) * 100, 4)}")
print(f"std:  {round(np.std(gw_vals) * 100, 4)}")
print(f"min:  {round(np.min(gw_vals) * 100, 4)}")
print(f"max:  {round(np.max(gw_vals) * 100, 4)}")

print("=== Agricultural Risk (UI scale = *5) ===")
print(f"mean: {round(np.mean(ag_vals) * 5, 4)}")
print(f"std:  {round(np.std(ag_vals) * 5, 4)}")
print(f"min:  {round(np.min(ag_vals) * 5, 4)}")
print(f"max:  {round(np.max(ag_vals) * 5, 4)}")