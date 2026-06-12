import json
import numpy as np
import pandas as pd

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
    Thresholds based on IMD's SPI-based drought classification framework.
    
    Normal:        Z > -0.5
    Elevated Risk: -1.0 < Z <= -0.5
    Severe Stress: -1.5 < Z <= -1.0
    Critical:      Z <= -1.5
    """
    if zscore > -0.5:
        return "Normal"
    elif zscore > -1.0:
        return "Elevated Risk"
    elif zscore > -1.5:
        return "Severe Stress"
    else:
        return "Critical Resource Alert"

# ── 4. GROUNDWATER STRESS PROXY ───────────────────────────────────────────────

def compute_groundwater_stress(reservoir_storage_pct, 
                               rainfall_ond, 
                               rainfall_jjas,
                                 reservoir_mean, rainfall_ond_mean, 
                                 rainfall_jjas_mean):
    """
    Derived proxy for groundwater stress.
    Lower reservoir storage + lower rainfall = higher stress.
    Uses both OND and JJAS rainfall for Tamil Nadu
    Returns a value between 0 (no stress) and 1 (maximum stress).
    Source: Nair et al. 2021 (GRL); PMC rainfall-groundwater study
    """
    reservoir_norm = np.clip(reservoir_storage_pct / max(reservoir_mean, 1), 0, 2)
    rainfall_ond_norm  = np.clip(rainfall_ond / max(rainfall_ond_mean, 1), 0, 2)
    rainfall_jjas_norm  = np.clip(rainfall_jjas / max(rainfall_jjas_mean, 1), 0, 2)

    # OND weighted higher for Tamil Nadu (primary monsoon season)
    rainfall_norm = (0.6 * rainfall_ond_norm) + (0.4 * rainfall_jjas_norm)

    stress = 1 - (0.6 * reservoir_norm + 0.4 * rainfall_norm) / 2
    return round(float(np.clip(stress, 0, 1)), 4)

# ── 5. AGRICULTURAL RISK EXPOSURE ─────────────────────────────────────────────

def compute_agricultural_risk(crop_yield, crop_mean, rainfall_ond, rainfall_jjas, rainfall_ond_mean, rainfall_jjas_mean):
    """
    Composite of crop yield deviation and rainfall deficit severity.
    Returns a value between 0 (no risk) and 1 (maximum risk).
    Source: FAO ASIS; Tandfonline 2025 agricultural drought mapping
    """
    yield_deviation  = np.clip((crop_mean - crop_yield) / max(crop_mean, 1), 0, 1)

    # Combined rainfall deficit — OND weighted higher for Tamil Nadu
    ond_deficit = np.clip((rainfall_ond_mean - rainfall_ond) / max(rainfall_ond_mean, 1), 0, 1)
    jjas_deficit = np.clip((rainfall_jjas_mean - rainfall_jjas) / max(rainfall_jjas_mean, 1), 0, 1)

    rainfall_deficit = (0.6 * ond_deficit) + (0.4 * jjas_deficit)

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

def evaluate_scenario(crop_pred, water_pred, rainfall_ond, rainfall_jjas, stats, policy_rules):
    """
    Given model predictions, compute all indicators and return
    a complete scenario result dictionary.
    """
    crop_mean  = stats["crop_yield_stability"]["mean"]
    crop_std   = stats["crop_yield_stability"]["std"]
    water_mean = stats["water_reservoir_security"]["mean"]
    water_std  = stats["water_reservoir_security"]["std"]
    rainfall_ond_mean  = stats["rainfall_ond"]["mean"]
    rainfall_jjas_mean = stats["rainfall_jjas"]["mean"]

    # Z-scores
    crop_z  = calculate_zscore(crop_pred,  crop_mean,  crop_std)
    water_z = calculate_zscore(water_pred, water_mean, water_std)

    # Risk classifications
    crop_risk  = classify_risk(crop_z)
    water_risk = classify_risk(water_z)

    # Derived indicators
    groundwater_stress = compute_groundwater_stress(
        water_pred, rainfall_ond, rainfall_jjas, water_mean, rainfall_ond_mean, rainfall_jjas_mean
    )
    agricultural_risk = compute_agricultural_risk(
        crop_pred, crop_mean, rainfall_ond, rainfall_jjas, rainfall_ond_mean, rainfall_jjas_mean
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

# ── 10. PREPROCESSING MAPPING LAYER ──────────────────────────────────────────

def map_ui_inputs_to_features(sst_anomaly, rainfall_deficit_pct, 
                               drought_duration, heat_stress, 
                               reservoir_storage_pct, year=2004):
    """
    Converts user-facing UI slider values to model input features.
    
    UI Inputs:
        sst_anomaly         : float, range 0.0 to +3.5°C (maps to ONI)
        rainfall_deficit_pct: float, 0–100% deficit (maps to rainfall depression)
        drought_duration    : int,   0–12 months (multiplier on rainfall deficit)
        heat_stress         : int,   1–5 scale (composite with ONI)
        reservoir_storage_pct: float, 0–100% (passed through directly)
        year                : int,   fixed at 2004 (historical midpoint)
    
    Returns:
        dict of model-ready features
    """
    # ONI: SST anomaly maps directly
    # Historical ONI range in dataset: -1.7 to 2.6
    oni_djf = sst_anomaly * (2.6 / 3.5)

    # Rainfall JJAS: apply deficit % to historical mean
    # Historical mean JJAS: 311.1mm
    rainfall_jjas_mean = 311.1
    rainfall_jjas = rainfall_jjas_mean * (1 - rainfall_deficit_pct / 100)

    # Rainfall OND: apply deficit % with drought duration multiplier
    # Historical mean OND: 445.5mm
    # Longer drought duration = deeper OND deficit
    rainfall_ond_mean = 445.5
    duration_multiplier = 1 + (drought_duration / 12) * 0.5
    rainfall_ond = rainfall_ond_mean * (1 - (rainfall_deficit_pct / 100) * duration_multiplier)
    rainfall_ond = max(rainfall_ond, 0)

    return {
        "oni_djf":               round(oni_djf, 4),
        "rainfall_jjas":         round(rainfall_jjas, 2),
        "rainfall_ond":          round(rainfall_ond, 2),
        "reservoir_storage_pct": round(reservoir_storage_pct, 2),
        "year":                  year
    }

# 11. POLICY COMPARISON ENGINE

def evaluate_all_policies(climate_inputs, crop_model, water_model, stats, policy_rules, top_n=3):
    """
    Evaluates all 16 binary policy combinations and returns
    the top N combinations ranked by Regional Resilience Score.

    Policy flags (all binary 0/1):
        drought_resistant_crops : Short-cycle drought-resistant crop program
        groundwater_rationing   : Emergency groundwater rationing
        supplemental_irrigation : Supplemental irrigation support
        water_conservation      : Water conservation initiative
    
    Parameters:
        climate_inputs : dict of preprocessed model features
        crop_model     : trained RandomForest crop model
        water_model    : trained RandomForest water model
        stats          : historical_statistics.json contents
        policy_rules   : policy_rules.json contents
        top_n          : number of top combinations to return

    Returns:
        list of dicts, sorted by resilience score descending
    """ 
    import itertools

    policy_names = [
        "drought_resistant_crops",
        "groundwater_rationing",
        "supplemental_irrigation",
        "water_conservation"
    ]

    results = []

    thresholds = load_thresholds()

    # Generate all 16 binary combinations
    for combo in itertools.product([0, 1], repeat=4):
        policy_flags = dict(zip(policy_names, combo))

        # Build feature vector with policy flags
        # Policy flags act as additive adjustments to climate inputs
        adjusted_inputs = climate_inputs.copy()

        # 8% yield improvement from drought tolerant varieties
        # Source: Economic Impact of Drought Tolerant Rice Varieties in South India
        # ResearchGate, 2015 — Tamil Nadu specific field study
        if policy_flags["drought_resistant_crops"]:
            adjusted_inputs["rainfall_jjas"] = min(
                adjusted_inputs["rainfall_jjas"] * 1.08,
                thresholds["out_of_distribution_bounds"]["rainfall_jjas"]["max"]
            )
            adjusted_inputs["rainfall_ond"] = min(
                adjusted_inputs["rainfall_ond"] * 1.08,
                thresholds["out_of_distribution_bounds"]["rainfall_ond"]["max"]
            )
            # Drought resistant crops also improve overall agricultural conditions, directly boosting the correlated groundnut yield feature
            groundnut_mean = stats["groundnut_yield"]["mean"]
            adjusted_inputs["groundnut_yield"] = min(
                adjusted_inputs.get("groundnut_yield", groundnut_mean) * 1.15,
                stats["groundnut_yield"]["max"]
            )
        
        # 15% effective water availability improvement from supplemental irrigation
        # Conservative estimate based on FAO Crop Yield Response to Water (FAO, 2012)
        # Full irrigation yields 76% more than rainfed (FAO 2025); supplemental effect modeled conservatively
        if policy_flags["supplemental_irrigation"]:
            adjusted_inputs["reservoir_storage_pct"] = min(
                adjusted_inputs["reservoir_storage_pct"] * 1.15,
                100.0
            )

        # 10% reservoir retention improvement from water conservation measures
        # Conservative estimate; drip/micro-irrigation can improve efficiency up to 70%
        # Source: Drishti IAS citing Maharashtra drip irrigation mandate
        # 10% reflects policy-level partial adoption, not full system conversion
        if policy_flags["water_conservation"]:
            adjusted_inputs["reservoir_storage_pct"] = min(
                adjusted_inputs["reservoir_storage_pct"] * 1.10,
                100.0
            )
        # 5% reservoir storage benefit from reduced groundwater extraction pressure
        # Rationing reduces demand on surface water reserves during drought periods
        # Source: NDMA Drought Management Manual (2020) — groundwater rationing directives
        if policy_flags["groundwater_rationing"]:
            adjusted_inputs["reservoir_storage_pct"] = min(
                adjusted_inputs["reservoir_storage_pct"] * 1.05,
                100.0
            )
        
        # Model input array
        feature_order = [
            "oni_djf",
            "rainfall_jjas",
            "rainfall_ond",
            "reservoir_storage_pct",
            "year",
            "groundnut_yield"
        ]

        # groundnut_yield not adjusted by policy - using historical mean
        adjusted_inputs.setdefault("groundnut_yield", load_historical_stats()["groundnut_yield"]["mean"])

        X = pd.DataFrame([{k: adjusted_inputs[k] for k in feature_order}])

        # Predict
        crop_pred = crop_model.predict(X)[0]
        water_pred = water_model.predict(X)[0]

        # Evaluate full scenario
        scenario = evaluate_scenario(
            crop_pred=crop_pred,
            water_pred=water_pred,
            rainfall_ond=adjusted_inputs["rainfall_ond"],
            rainfall_jjas=adjusted_inputs["rainfall_jjas"],
            stats=stats,
            policy_rules=policy_rules
        )

        results.append({
            "policy_flags": policy_flags,
            "active_policies": [k for k, v in policy_flags.items() if v == 1],
            "crop_yield_stability": scenario["crop_yield_stability"],
            "water_reservoir_security": scenario["water_reservoir_security"],
            "groundwater_stress": scenario["groundwater_stress"],
            "agricultural_risk": scenario["agricultural_risk_exposure"],
            "resilience_score": scenario["regional_resilience_score"],
            "overall_risk": scenario["overall_risk_category"],
            "recommendations": scenario["recommendations"]
        })

    # Sort by resilience score descending
    results.sort(key=lambda x: x["resilience_score"], reverse=True)

    return results[:top_n]
