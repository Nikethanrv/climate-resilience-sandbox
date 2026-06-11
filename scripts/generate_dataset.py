import pandas as pd
import numpy as np

# ── 1. LOAD ──────────────────────────────────────────────────────────────────

oni = pd.read_csv("data/ONI.csv")
rainfall = pd.read_csv("data/Tamil Nadu Rainfall.csv")
icrisat = pd.read_csv("data/ICRISAT-District Level Data.csv")
reservoir = pd.read_csv("data/Yearly Reservoir Level & Storage Timeseries Timeseries data for All agency for period 1991 to 2026.csv")

# ── 2. ONI ───────────────────────────────────────────────────────────────────

oni_clean = oni[["Year", "DJF"]].rename(columns={"Year": "year", "DJF": "oni_djf"})

# ── 3. RAINFALL ──────────────────────────────────────────────────────────────

rainfall_clean = rainfall[["YEAR", "JJAS", "OND"]].rename(columns={
    "YEAR": "year",
    "JJAS": "rainfall_jjas",
    "OND": "rainfall_ond"
})

# ── 4. ICRISAT ───────────────────────────────────────────────────────────────

# State-level mean across all districts, per year
icrisat_clean = (
    icrisat
    .groupby("Year")[["RICE YIELD (Kg per ha)", "GROUNDNUT YIELD (Kg per ha)"]]
    .mean()
    .reset_index()
    .rename(columns={
        "Year": "year",
        "RICE YIELD (Kg per ha)": "rice_yield",
        "GROUNDNUT YIELD (Kg per ha)": "groundnut_yield"
    }))

# ── 5. RESERVOIR ─────────────────────────────────────────────────────────────

# Date column is already an integer year — no parsing needed
reservoir = reservoir[reservoir["Live Cap FRL"] > 0].copy()

res_agg = (
    reservoir
    .groupby("Date")[["Current Live Storage", "Live Cap FRL"]]
    .sum()
    .reset_index()
    .rename(columns={"Date": "year"})
)

res_agg["reservoir_storage_pct"] = (
    res_agg["Current Live Storage"] / res_agg["Live Cap FRL"] * 100
).clip(0, 100)

reservoir_clean = res_agg[["year", "reservoir_storage_pct"]]

mean_storage = reservoir_clean[reservoir_clean["reservoir_storage_pct"] > 0]["reservoir_storage_pct"].mean()
reservoir_clean["reservoir_storage_pct"] = reservoir_clean["reservoir_storage_pct"].replace(0, mean_storage)
print(f"Replaced zeros with mean: {mean_storage:.2f}%")
print(reservoir_clean[reservoir_clean["year"] <= 1999])

# ── 6. MERGE ─────────────────────────────────────────────────────────────────

df = (
    oni_clean
    .merge(rainfall_clean, on="year", how="inner")
    .merge(icrisat_clean, on="year", how="inner")
    .merge(reservoir_clean, on="year", how="inner")
)

# ── 7. TRIM TO INTERSECTION RANGE ────────────────────────────────────────────

df = df[(df["year"] >= 1991) & (df["year"] <= 2017)].reset_index(drop=True)

# ── 8. VALIDATION ─────────────────────────────────────────────────────────────

print(f"Rows: {len(df)}")
print(f"Year range: {df['year'].min()} – {df['year'].max()}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nSample:\n{df.head()}")
print(f"\nDescriptive stats:\n{df.describe()}")

# ── 9. SAVE ───────────────────────────────────────────────────────────────────

df.to_csv("data/historical_climate_data.csv", index=False)
print("\n✅ Saved: data/historical_climate_data.csv")

print(reservoir_clean[reservoir_clean["reservoir_storage_pct"] == 0])