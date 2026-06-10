import json
import numpy as np

# ── 1. LOAD KNOWLEDGE BASE FILES ─────────────────────────────────────────────

def load_policy_rules(path="data/policy_rules.json"):
    with open(path, "r") as f:
        return json.load(f)

def load_thresholds(path="data/risk_threshold.json"):
    with open(path, "r") as f:
        return json.load(f)

def load_metadata(path="data/metadata.json"):
    with open(path, "r") as f:
        return json.load(f)

def load_historical_stats(path="data/historical_statistics.json"):
    with open(path, "r") as f:
        return json.load(f)

# ── 2. Z-SCORE COMPUTATION ────────────────────────────────────────────────────

def calculate_zscore(value, mean, std):
    """
    Compute Z-score of a predicted value against historical distribution.
    Returns 0 if std is zero to avoid division errors.
    """
    if std == 0:
        return 0.0
    return (value - mean) / std

# ── 3. RISK CLASSIFICATION ────────────────────────────────────────────────────

def classify_risk(zscore):
    """
    Map Z-score to risk category.
    Thresholds follow standard statistical conventions:
      Z > -0.5  → Normal
      Z > -1.0  → Elevated Risk
      Z > -2.0  → Severe Stress
      Z <= -2.0 → Critical Resource Alert
    These will be calibrated and stored in risk_thresholds.json in Phase 3.
    """
    if zscore > -0.5:
        return "Normal"
    elif zscore > -1.0:
        return "Elevated Risk"
    elif zscore > -2.0:
        return "Severe Stress"
    else:
        return "Critical Resource Alert"

# ── 4. GROUNDWATER STRESS PROXY ───────────────────────────────────────────────

def compute_groundwater_stress(reservoir_storage_pct, rainfall_ond, 
                                 reservoir_mean, rainfall_mean):
    """
    Derived proxy for groundwater stress.
    Lower reservoir storage + lower rainfall = higher stress.
    Returns a value between 0 (no stress) and 1 (maximum stress).
    """
    reservoir_norm = np.clip(reservoir_storage_pct / max(reservoir_mean, 1), 0, 2)
    rainfall_norm  = np.clip(rainfall_ond / max(rainfall_mean, 1), 0, 2)
    stress = 1 - (0.6 * reservoir_norm + 0.4 * rainfall_norm) / 2
    return round(float(np.clip(stress, 0, 1)), 4)

# ── 5. AGRICULTURAL RISK EXPOSURE ─────────────────────────────────────────────

def compute_agricultural_risk(crop_yield, crop_mean, rainfall_ond, rainfall_mean):
    """
    Composite of crop yield deviation and rainfall deficit severity.
    Returns a value between 0 (no risk) and 1 (maximum risk).
    """
    yield_deviation  = np.clip((crop_mean - crop_yield) / max(crop_mean, 1), 0, 1)
    rainfall_deficit = np.clip((rainfall_mean - rainfall_ond) / max(rainfall_mean, 1), 0, 1)
    return round(float(0.6 * yield_deviation + 0.4 * rainfall_deficit), 4)

# ── 6. REGIONAL RESILIENCE SCORE ──────────────────────────────────────────────

def compute_resilience_score(crop_yield_stability, water_reservoir_security,
                              agricultural_risk_exposure,
                              crop_mean, water_mean):
    """
    Weighted composite:
      40% Crop Yield Stability
      40% Water Reservoir Security
      20% Agricultural Risk Exposure (inverted — higher risk = lower resilience)
    Returns a score between 0 and 100.
    """
    crop_norm  = np.clip(crop_yield_stability / max(crop_mean, 1), 0, 1)
    water_norm = np.clip(water_reservoir_security / 100, 0, 1)
    risk_norm  = 1 - agricultural_risk_exposure  # invert

    score = (0.4 * crop_norm + 0.4 * water_norm + 0.2 * risk_norm) * 100
    return round(float(score), 2)

# ── 7. POLICY RECOMMENDATION ROUTING ─────────────────────────────────────────

def get_recommendations(risk_category, policy_rules):
    """
    Route a risk category to its corresponding policy recommendations.
    Falls back to Normal guidance if category not found.
    """
    category_map = {
        "Normal":                 "normal",
        "Elevated Risk":          "elevated",
        "Severe Stress":          "severe",
        "Critical Resource Alert": "critical"
    }
    key = category_map.get(risk_category, "normal")
    return policy_rules.get(key, policy_rules.get("normal", {}))

# ── 8. OUT-OF-DISTRIBUTION WARNING ───────────────────────────────────────────

def check_out_of_distribution(value, min_val, max_val, label):
    """
    Warn if a scenario input exceeds the historical training range.
    """
    if value < min_val or value > max_val:
        return (f"⚠️ Warning: {label} value of {value:.2f} exceeds the historical "
                f"range ({min_val:.2f}–{max_val:.2f}) represented in the training "
                f"dataset. Results should be interpreted as exploratory estimates.")
    return None

# ── 9. FULL SCENARIO EVALUATION ───────────────────────────────────────────────

def evaluate_scenario(crop_pred, water_pred, rainfall_ond, stats, policy_rules):
    """
    Given model predictions, compute all indicators and return
    a complete scenario result dictionary.
    """
    crop_mean  = stats["crop_yield_stability"]["mean"]
    crop_std   = stats["crop_yield_stability"]["std"]
    water_mean = stats["water_reservoir_security"]["mean"]
    water_std  = stats["water_reservoir_security"]["std"]
    rainfall_mean = 445.47  # historical OND mean from dataset

    # Z-scores
    crop_z  = calculate_zscore(crop_pred,  crop_mean,  crop_std)
    water_z = calculate_zscore(water_pred, water_mean, water_std)

    # Risk classifications
    crop_risk  = classify_risk(crop_z)
    water_risk = classify_risk(water_z)

    # Derived indicators
    groundwater_stress = compute_groundwater_stress(
        water_pred, rainfall_ond, water_mean, rainfall_mean
    )
    agricultural_risk = compute_agricultural_risk(
        crop_pred, crop_mean, rainfall_ond, rainfall_mean
    )
    resilience_score = compute_resilience_score(
        crop_pred, water_pred, agricultural_risk, crop_mean, water_mean
    )

    # Recommendations
    overall_risk = crop_risk if crop_z <= water_z else water_risk
    recommendations = get_recommendations(overall_risk, policy_rules)

    return {
        "crop_yield_stability":    round(float(crop_pred), 2),
        "water_reservoir_security": round(float(water_pred), 2),
        "groundwater_stress":      groundwater_stress,
        "agricultural_risk_exposure": agricultural_risk,
        "regional_resilience_score":  resilience_score,
        "crop_risk_category":      crop_risk,
        "water_risk_category":     water_risk,
        "overall_risk_category":   overall_risk,
        "crop_zscore":             round(crop_z, 4),
        "water_zscore":            round(water_z, 4),
        "recommendations":         recommendations,
    }