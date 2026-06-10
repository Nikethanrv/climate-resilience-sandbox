import pandas as pd
import joblib
import json
import numpy as np
from src.utils import calculate_zscore, classify_risk

df = pd.read_csv("data/historical_climate_data.csv")
with open("data/historical_statistics.json") as f:
    stats = json.load(f)

crop_mean = stats["crop_yield_stability"]["mean"]
crop_std  = stats["crop_yield_stability"]["std"]

df["crop_z"] = df["rice_yield"].apply(lambda x: calculate_zscore(x, crop_mean, crop_std))
df["crop_risk"] = df["crop_z"].apply(classify_risk)

print(df[["year", "rice_yield", "crop_z", "crop_risk"]].to_string())
print("\nCategory distribution:")
print(df["crop_risk"].value_counts())


water_mean = stats["water_reservoir_security"]["mean"]
water_std  = stats["water_reservoir_security"]["std"]

df["water_z"] = df["reservoir_storage_pct"].apply(
    lambda x: calculate_zscore(x, water_mean, water_std))
df["water_risk"] = df["water_z"].apply(classify_risk)

print(df[["year", "reservoir_storage_pct", "water_z", "water_risk"]].to_string())
print("\nCategory distribution:")
print(df["water_risk"].value_counts())