import joblib
import json
from src.utils import load_historical_stats, evaluate_scenario, load_policy_rules, evaluate_all_policies, map_ui_inputs_to_features

stats = load_historical_stats()
policy_rules = load_policy_rules()
crop_model = joblib.load("models/crop_model.joblib")
water_model = joblib.load("models/water_model.joblib")

# Test with mean values — should return Normal conditions
# result = evaluate_scenario(
#     crop_pred=3440.0,
#     water_pred=45.0,
#     rainfall_ond=445.0,
#     rainfall_jjas=311.0,
#     stats=stats,
#     policy_rules=policy_rules
# )

# import json
# print(json.dumps(result, indent=2))

# Simulating a severe drought scenario
# climate_inputs = map_ui_inputs_to_features(
#     sst_anomaly=2.5,
#     rainfall_deficit_pct=60,
#     drought_duration=8,
#     heat_stress=4,
#     reservoir_storage_pct=20.0
# )

climate_inputs = map_ui_inputs_to_features(
    sst_anomaly=1.5,
    rainfall_deficit_pct=40,
    drought_duration=4,
    heat_stress=3,
    reservoir_storage_pct=35.0
)

top_policies = evaluate_all_policies(
    climate_inputs=climate_inputs,
    crop_model=crop_model,
    water_model=water_model,
    stats=stats,
    policy_rules=policy_rules,
    top_n=3
)

for i, p in enumerate(top_policies, 1):
    print(f"\n-- Rank {i} -- ")
    print(f"    Active policies: {p['active_policies']}")
    print(f"    Resilience Score: {p['resilience_score']}")
    print(f"    Overall risk: {p['overall_risk']}")
    print(f"    Crop yield: {p['crop_yield_stability']}")
    print(f"    Water security: {p['water_reservoir_security']}")