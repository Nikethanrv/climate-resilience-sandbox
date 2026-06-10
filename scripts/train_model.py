import pandas as pd
import numpy as np
import json
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────

df = pd.read_csv("data/historical_climate_data.csv")

# ── 2. DEFINE FEATURES & TARGETS ─────────────────────────────────────────────

FEATURES = [
    "oni_djf",
    "rainfall_jjas",
    "rainfall_ond",
    "reservoir_storage_pct",
    "year",
    "groundnut_yield"  # correlated agricultural indicator
]

X = df[FEATURES]

# Targets
y_crop = df["rice_yield"]
y_water = df["reservoir_storage_pct"]

# ── 3. TRAIN MODELS ───────────────────────────────────────────────────────────

crop_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    max_depth=4,
    min_samples_leaf=3
)

water_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    max_depth=4,
    min_samples_leaf=3
)

crop_model.fit(X, y_crop)
water_model.fit(X, y_water)

# ── 4. EVALUATE — LEAVE-ONE-OUT CV ───────────────────────────────────────────
# LOO is appropriate for small datasets (n=27)

loo = LeaveOneOut()

def evaluate(model, X, y, name):
    preds = []
    actuals = []
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        m = RandomForestRegressor(
            n_estimators=100, random_state=42,
            max_depth=4, min_samples_leaf=3
        )
        m.fit(X_train, y_train)
        preds.append(m.predict(X_test)[0])
        actuals.append(y_test.values[0])

    mae  = mean_absolute_error(actuals, preds)
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    r2   = r2_score(actuals, preds)

    print(f"\n── {name} ──")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²:   {r2:.4f}")

    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}

print("=== Model Evaluation (Leave-One-Out CV) ===")
crop_metrics  = evaluate(crop_model,  X, y_crop,  "Crop Yield Model")
water_metrics = evaluate(water_model, X, y_water, "Water Reservoir Model")

# ── 5. SAVE MODELS ────────────────────────────────────────────────────────────

import os
os.makedirs("models", exist_ok=True)

joblib.dump(crop_model,  "models/crop_model.joblib")
joblib.dump(water_model, "models/water_model.joblib")
print("\n✅ Models saved to models/")

# ── 6. GENERATE HISTORICAL STATISTICS ────────────────────────────────────────

crop_preds_full  = crop_model.predict(X)
water_preds_full = water_model.predict(X)

def compute_stats(values, name):
    return {
        "name":   name,
        "mean":   round(float(np.mean(values)), 4),
        "std":    round(float(np.std(values)), 4),
        "min":    round(float(np.min(values)), 4),
        "max":    round(float(np.max(values)), 4),
        "p25":    round(float(np.percentile(values, 25)), 4),
        "p50":    round(float(np.percentile(values, 50)), 4),
        "p75":    round(float(np.percentile(values, 75)), 4),
    }

historical_stats = {
    "crop_yield_stability":    compute_stats(crop_preds_full,  "Crop Yield Stability"),
    "water_reservoir_security": compute_stats(water_preds_full, "Water Reservoir Security"),
    "evaluation_metrics": {
        "crop_model":  crop_metrics,
        "water_model": water_metrics
    }
}

with open("data/historical_statistics.json", "w") as f:
    json.dump(historical_stats, f, indent=2)

print("✅ Saved: data/historical_statistics.json")
print("\n=== Historical Statistics ===")
print(json.dumps(historical_stats, indent=2))