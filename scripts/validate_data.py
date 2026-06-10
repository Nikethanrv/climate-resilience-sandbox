import pandas as pd
import sys

# ── LOAD ──────────────────────────────────────────────────────────────────────

df = pd.read_csv("data/historical_climate_data.csv")

errors = []

# ── 1. EXPECTED COLUMNS ───────────────────────────────────────────────────────

expected_columns = [
    "year", "oni_djf", "rainfall_jjas", "rainfall_ond",
    "rice_yield", "groundnut_yield", "reservoir_storage_pct"
]

missing_cols = [c for c in expected_columns if c not in df.columns]
if missing_cols:
    errors.append(f"Missing columns: {missing_cols}")
else:
    print("✅ Column check passed")

# ── 2. MISSING VALUES ─────────────────────────────────────────────────────────

null_counts = df.isnull().sum()
if null_counts.any():
    errors.append(f"Missing values found:\n{null_counts[null_counts > 0]}")
else:
    print("✅ No missing values")

# ── 3. DATASET DIMENSIONS ────────────────────────────────────────────────────

if len(df) < 20:
    errors.append(f"Too few rows: {len(df)} (expected at least 20)")
else:
    print(f"✅ Row count: {len(df)}")

# ── 4. YEAR RANGE ─────────────────────────────────────────────────────────────

if df["year"].min() < 1991 or df["year"].max() > 2017:
    errors.append(f"Year range out of bounds: {df['year'].min()}–{df['year'].max()}")
else:
    print(f"✅ Year range: {df['year'].min()}–{df['year'].max()}")

# ── 5. EXPECTED FEATURE RANGES ───────────────────────────────────────────────

checks = {
    "oni_djf":                (-3.0, 3.0),
    "rainfall_jjas":          (0, 1000),
    "rainfall_ond":           (0, 1500),
    "rice_yield":             (500, 8000),
    "groundnut_yield":        (200, 5000),
    "reservoir_storage_pct":  (0, 100),
}

for col, (low, high) in checks.items():
    out = df[(df[col] < low) | (df[col] > high)]
    if not out.empty:
        errors.append(f"{col} has {len(out)} out-of-range values (expected {low}–{high})")
    else:
        print(f"✅ {col} range OK ({low}–{high})")

# ── RESULT ────────────────────────────────────────────────────────────────────

print("\n" + "─" * 50)
if errors:
    print("❌ Validation failed:")
    for e in errors:
        print(f"  • {e}")
    sys.exit(1)
else:
    print("✅ All checks passed. Dataset is ready for training.")