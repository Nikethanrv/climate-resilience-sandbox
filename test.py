from utils import load_historical_stats, evaluate_scenario, load_policy_rules

stats = load_historical_stats()
policy_rules = load_policy_rules()

# Test with mean values — should return Normal conditions
result = evaluate_scenario(
    crop_pred=3440.0,
    water_pred=45.0,
    rainfall_ond=445.0,
    stats=stats,
    policy_rules=policy_rules
)

import json
print(json.dumps(result, indent=2))