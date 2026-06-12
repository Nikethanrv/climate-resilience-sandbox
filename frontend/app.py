"""
AI Climate Resilience Sandbox — Frontend Phase 3

This module is the Streamlit UI for the sandbox. Phase 3 wires the existing,
already-built backend (``src/utils.py`` decision-support engine, the trained
``models/*.joblib`` surrogates, and the ``data/*.json`` knowledge base) to the
Phase 2 dashboard. NO backend logic is reimplemented here — every live capability
is consumed exclusively through the backend adapter (``get_backend()``), and every
live-path failure degrades gracefully to a local stub with a visible, non-blocking
notice. The app must never traceback on a missing or broken teammate module.

PHASE 2 FIXES (retained):
  FIX-1  RISK_* thresholds are module constants — never inline literals.
  FIX-2  Every section is a render_*() function; main() is the sole caller.
  FIX-3  All Plotly charts carry config={"displayModeBar": False}.
  FIX-4  Groundwater Stress uses delta_color="inverse" + reversed status scale.
  FIX-5  get_backend() adapter with a sidebar Stub/Partial/Live engine badge.
  FIX-6  load_historical_stats() replaces hardcoded stats everywhere.
  FIX-7  _is_ood() uses historical min/max bounds with hardcoded fallbacks.

PHASE 3 WIRING (this revision):
  P3-1  Fixed core wiring so Live mode works end-to-end:
        • historical_statistics.json read from data/ (was project root).
        • adapter imports src.utils (the real module) with absolute-path loaders.
        • model emulator feeds the model's true 6-feature order
          [oni_djf, rainfall_jjas, rainfall_ond, reservoir_storage_pct, year,
          groundnut_yield] via utils.map_ui_inputs_to_features — the previous
          8-feature vector would have tracebacked on the first prediction.
  P3-2  Recommendations consume a LIVE policy-comparison engine through
        backend["compare"] (policy_engine.compare_policies → utils.evaluate_all_policies
        → local 16-combo fallback), normalized to one schema. (Step 3.5)
  P3-3  Z-score → directive routing: per-indicator directive bands plus an
        "Active Directive" card for the worst indicator quoting policy-rules
        text from the adapter. (Step 3.6)
  P3-4  One get_ood_warnings() feeds the alert bar, the transparency expander,
        and the TXT/PDF exports identically; confidence ranges go live when the
        engine exposes them. (Step 3.7)
  P3-5  perf_counter instrumentation, slow-response caption, cached hot path,
        a hidden Dev/QA panel, and the Step 3.12 manual-QA checklist.
"""

# ── Stdlib imports ────────────────────────────────────────────────────────────
import contextlib
import inspect
import json
import logging
import os
import sys
import time

# ── Third-party imports ───────────────────────────────────────────────────────
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# ═════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

# Path resolution — works regardless of the CWD the user launches from.
_HERE = os.path.dirname(os.path.abspath(__file__))   # frontend/
_ROOT = os.path.dirname(_HERE)                        # project root
_DATA_DIR   = os.path.join(_ROOT, "data")
_MODELS_DIR = os.path.join(_ROOT, "models")

# P3-1: data files live under data/.
_HIST_STATS_PATH   = os.path.join(_DATA_DIR, "historical_statistics.json")
_POLICY_RULES_PATH = os.path.join(_DATA_DIR, "policy_rules.json")
_RISK_THRESH_PATH  = os.path.join(_DATA_DIR, "risk_threshold.json")
_REQUIRED_STAT_KEYS = {"mean", "std", "min", "max"}

# FIX-1: risk thresholds defined exactly once.
RISK_NORMAL   = 0.35
RISK_ELEVATED = 0.50
RISK_SEVERE   = 0.70

# Raw-model-output → 0–100 UI normalization bounds (kg/ha and % capacity).
# These mirror the already-shipped backend integration and keep emulator output
# on the same scale as the *_ui historical statistics.
CROP_NORM_MIN, CROP_NORM_MAX   = 2690.11, 4663.37
WATER_NORM_MIN, WATER_NORM_MAX = 24.14, 66.50

# OOD boundary fallbacks for the UI inputs (historical training range 1991–2017).
# heat_stress > 4 means == 5 (top of the slider), matching the original check.
OOD_DEFAULTS: dict = {
    "sst_anomaly":      3.0,
    "rainfall_deficit": 80.0,
    "drought_months":   10.0,
    "heat_stress":       4.0,
}

# P3-4: the EXACT, single OOD copy required to appear identically in the alert
# bar, the transparency expander, and the exported reports' warnings list.
OOD_WARNING_TEXT = (
    "Warning: This scenario exceeds the historical range represented in the "
    "training dataset. Results should be interpreted as exploratory estimates."
)

# P3-5: a full rerun slower than this (seconds) surfaces a caching notice.
SLOW_RESPONSE_SECS = 1.5

# Fallback stats — used when historical_statistics.json is absent or partial.
# Keyed to match the *_ui scales the emulator outputs so the Z-score panel and
# metric cards stay meaningful even in stub mode.
MOCK_STATS: dict = {
    "crop_yield_stability_ui":     {"mean": 50.0, "std": 20.0, "min": 0.0, "max": 100.0},
    "water_reservoir_security_ui": {"mean": 50.0, "std": 20.0, "min": 0.0, "max": 100.0},
    "groundwater_stress_ui":       {"mean": 50.0, "std": 15.0, "min": 0.0, "max": 100.0},
    "agricultural_risk_ui":        {"mean":  2.5, "std":  1.0, "min": 0.0, "max":   5.0},
    "regional_resilience":         {"mean": 65.0, "std": 11.0, "min": 20.0, "max": 100.0},
    # Raw-scale keys the live policy engine reads when models are present.
    "crop_yield_stability":     {"mean": 3436.90, "std": 391.55, "min": 2914.97, "max": 4115.59},
    "water_reservoir_security": {"mean": 45.33,   "std": 13.45,  "min": 24.14,   "max": 66.50},
    "groundnut_yield":          {"mean": 1892.23, "std": 450.73, "min": 1366.46, "max": 3053.47},
    "rainfall_ond":             {"mean": 445.47,  "std": 170.23, "min": 149.3,   "max": 782.3},
    "rainfall_jjas":            {"mean": 311.15,  "std": 86.79,  "min": 94.2,    "max": 434.3},
}

# Preset scenario definitions.
MODERATE_EL_NINO = {"sst_anomaly": 1.2, "rainfall_deficit": 20, "drought_months":  2, "heat_stress": 2}
SEVERE_DROUGHT   = {"sst_anomaly": 2.0, "rainfall_deficit": 55, "drought_months":  6, "heat_stress": 3}
EXTREME_ENSO     = {"sst_anomaly": 3.2, "rainfall_deficit": 85, "drought_months": 10, "heat_stress": 5}

PRESET_LABELS: dict = {
    "moderate_el_nino": "Moderate El Niño",
    "severe_drought":   "Severe Drought",
    "extreme_enso":     "Extreme ENSO Shock",
}

SESSION_DEFAULTS: dict = {
    "sst_anomaly":       1.5,
    "rainfall_deficit":  30,
    "drought_months":    3,
    "heat_stress":       2,
    "adapt_crops":       False,
    "adapt_groundwater": False,
    "adapt_irrigation":  False,
    "adapt_water":       False,
    "active_preset":     "custom",
}

# Frontend adaptation keys and their display names.
STRATEGY_NAMES: dict = {
    "crops": "Drought-Resistant Crops",
    "gw":    "Groundwater Rationing",
    "irr":   "Supplemental Irrigation",
    "water": "Water Conservation",
}
ORDERED_KEYS: tuple = ("crops", "gw", "irr", "water")

# Maps the live engine's policy-flag names to frontend adaptation keys.
ENGINE_KEY_MAP: dict = {
    "drought_resistant_crops": "crops",
    "groundwater_rationing":   "gw",
    "supplemental_irrigation": "irr",
    "water_conservation":      "water",
}

# ── Directive routing (Step 3.6 display contract) ────────────────────────────
# Z-band cut points. For "reversed" indicators (positive z = worse) the sign of
# z is flipped before classification.
Z_DIRECTIVE_CRITICAL = -2.0
Z_DIRECTIVE_ELEVATED = -1.0
Z_DIRECTIVE_MONITOR  =  0.0

DIRECTIVE_CRITICAL = "🔴 Critical Drought Guidance"
DIRECTIVE_ELEVATED = "🟠 Elevated Stress Protocol"
DIRECTIVE_MONITOR  = "🟡 Enhanced Monitoring Required"
DIRECTIVE_NORMAL   = "🟢 Normal Operating Range"

DIRECTIVE_COLORS: dict = {
    DIRECTIVE_CRITICAL: "#dc2626",
    DIRECTIVE_ELEVATED: "#ea580c",
    DIRECTIVE_MONITOR:  "#ca8a04",
    DIRECTIVE_NORMAL:   "#16a34a",
}

# Maps a directive band to the policy_rules.json planning category.
DIRECTIVE_POLICY_KEY: dict = {
    DIRECTIVE_CRITICAL: "critical",
    DIRECTIVE_ELEVATED: "severe",
    DIRECTIVE_MONITOR:  "elevated",
    DIRECTIVE_NORMAL:   "normal",
}

# Stub policy_rules use flat human-readable keys; map the JSON categories to them.
_STUB_POLICY_KEYS: dict = {
    "critical": "Critical Alert",
    "severe":   "Severe Stress",
    "elevated": "Elevated Risk",
    "normal":   "Normal",
}

# Per-indicator metadata for the Z-score panel:
#   result_key -> (display name, unit, reversed?, historical-stats key)
ZSCORE_META: dict = {
    "crop_yield_stability":     ("🌾 Crop Yield Stability",     "%",     False, "crop_yield_stability_ui"),
    "water_reservoir_security": ("💧 Water Reservoir Security",  "%",     False, "water_reservoir_security_ui"),
    "groundwater_stress":       ("🏜 Groundwater Stress",        "%",     True,  "groundwater_stress_ui"),
    "agricultural_risk":        ("⚠️ Agricultural Risk",         "/ 5.0", True,  "agricultural_risk_ui"),
    "regional_resilience":      ("🛡 Regional Resilience",       "%",     False, "regional_resilience"),
}


@contextlib.contextmanager
def _chdir(path: str):
    """Temporarily run with ``path`` as the CWD (restored on exit).

    The backend's ``evaluate_all_policies`` loads a couple of knowledge-base
    files with paths relative to the project root, so the live policy engine is
    invoked from ``_ROOT`` regardless of where Streamlit was launched.
    """
    prev = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(prev)


# ═════════════════════════════════════════════════════════════════════════════
# LOCAL STUB FUNCTIONS  (fallbacks used when teammate files are absent)
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=60)
def run_climate_emulator(
    sst: float,
    rainfall: float,
    drought: float,
    heat: float,
    adapt_dict: dict,
) -> dict:
    """Stub emulator — replaced by the model emulator via the backend adapter.

    Returns a dict with the five UI indicators plus the adjusted risk score.
    """
    base  = sst / 3.5 * 0.4 + rainfall / 100 * 0.4 + drought / 12 * 0.2
    bonus = (
        adapt_dict.get("crops", 0) * 0.08
        + adapt_dict.get("gw",    0) * 0.06
        + adapt_dict.get("irr",   0) * 0.07
        + adapt_dict.get("water", 0) * 0.05
    )
    adj = max(0.0, min(1.0, base - bonus))
    return {
        "crop_yield_stability":     round((1 - adj) * 100, 1),
        "water_reservoir_security": round((1 - adj * 0.9) * 100, 1),
        "groundwater_stress":       round(adj * 100, 1),
        "agricultural_risk":        round(adj * 5, 2),
        "regional_resilience":      round((1 - adj) * 100, 1),
        "adjusted_risk_score":      adj,
    }


def calculate_zscore(value: float, mean: float, std: float) -> float:
    """Stub — replaced by utils.calculate_zscore via the backend adapter."""
    return round((value - mean) / std if std > 0 else 0.0, 2)


@st.cache_data(ttl=3600)
def load_policy_rules() -> dict:
    """Stub — replaced by utils.load_policy_rules via the backend adapter."""
    return {
        "Normal":         "Maintain standard monitoring. No immediate action required.",
        "Elevated Risk":  "Activate early-warning protocols. Review irrigation schedules.",
        "Severe Stress":  "Implement water rationing. Distribute drought-resistant seeds.",
        "Critical Alert": "Emergency response required. Activate all adaptation measures.",
    }


def load_risk_thresholds() -> dict:
    """Stub — replaced by utils.load_thresholds via the backend adapter."""
    return {"normal": RISK_NORMAL, "elevated": RISK_ELEVATED, "severe": RISK_SEVERE}


def stub_map_inputs(
    sst_anomaly: float,
    rainfall_deficit_pct: float,
    drought_duration: float,
    heat_stress: float,
    reservoir_storage_pct: float,
    year: int = 2004,
) -> dict:
    """Stub feature-mapping — replaced by utils.map_ui_inputs_to_features."""
    heat_multiplier = 1 + (heat_stress - 1) * 0.10
    oni_djf = min(sst_anomaly * (2.6 / 3.5) * heat_multiplier, 2.6)
    rainfall_jjas = 311.1 * (1 - rainfall_deficit_pct / 100)
    duration_multiplier = 1 + (drought_duration / 12) * 0.5
    rainfall_ond = max(0.0, 445.5 * (1 - (rainfall_deficit_pct / 100) * duration_multiplier))
    return {
        "oni_djf":               round(oni_djf, 4),
        "rainfall_jjas":         round(rainfall_jjas, 2),
        "rainfall_ond":          round(rainfall_ond, 2),
        "reservoir_storage_pct": round(reservoir_storage_pct, 2),
        "year":                  year,
        "groundnut_yield":       1892.23,
    }


# ═════════════════════════════════════════════════════════════════════════════
# BACKEND ADAPTER  (FIX-5 / P3-1 / P3-2)
# ═════════════════════════════════════════════════════════════════════════════

# Model feature order — must match the trained models' feature_names_in_.
MODEL_FEATURE_ORDER: tuple = (
    "oni_djf", "rainfall_jjas", "rainfall_ond",
    "reservoir_storage_pct", "year", "groundnut_yield",
)


def make_model_emulator(crop_model, water_model, map_fn, stats: dict):
    """Build a drop-in emulator backed by the trained surrogate models.

    P3-1: UI slider values are converted to the model's true 6-feature vector via
    ``map_fn`` (utils.map_ui_inputs_to_features or the stub), then crop yield and
    water security are predicted and normalized to the 0–100 UI scale. The three
    derived indicators reuse the stub's adaptation-bonus formula so adaptation
    checkboxes still move the radar / map / status alerts.
    """
    water_max = stats.get("water_reservoir_security", {}).get("max", WATER_NORM_MAX)

    def _emulator(
        sst: float,
        rainfall: float,
        drought: float,
        heat: float,
        adapt_dict: dict,
    ) -> dict:
        reservoir = max(10.0, water_max * (1 - rainfall / 100))
        feats = map_fn(
            sst_anomaly=sst,
            rainfall_deficit_pct=rainfall,
            drought_duration=drought,
            heat_stress=heat,
            reservoir_storage_pct=reservoir,
        )
        X = pd.DataFrame([[feats[f] for f in MODEL_FEATURE_ORDER]], columns=list(MODEL_FEATURE_ORDER))
        crop_yield = float(crop_model.predict(X)[0])
        water_sec  = float(water_model.predict(X)[0])

        crop_norm  = max(0.0, min(100.0, (crop_yield - CROP_NORM_MIN) / (CROP_NORM_MAX - CROP_NORM_MIN) * 100))
        water_norm = max(0.0, min(100.0, (water_sec - WATER_NORM_MIN) / (WATER_NORM_MAX - WATER_NORM_MIN) * 100))

        base  = sst / 3.5 * 0.4 + rainfall / 100 * 0.4 + drought / 12 * 0.2
        bonus = (
            adapt_dict.get("crops", 0) * 0.08
            + adapt_dict.get("gw",    0) * 0.06
            + adapt_dict.get("irr",   0) * 0.07
            + adapt_dict.get("water", 0) * 0.05
        )
        adj = max(0.0, min(1.0, base - bonus))

        return {
            "crop_yield_stability":     round(crop_norm, 1),
            "water_reservoir_security": round(water_norm, 1),
            "groundwater_stress":       round(adj * 100, 1),
            "agricultural_risk":        round(adj * 5, 2),
            "regional_resilience":      round((1 - adj) * 100, 1),
            "adjusted_risk_score":      adj,
        }

    return _emulator


def _normalize_combo(combo_keys: frozenset, score: float, description=None) -> dict:
    """Project any comparison row onto the single recommendation schema.

    Schema: ``{"strategies": [display names], "score": float,
              "combo_keys": frozenset, "description": str | None}``.
    """
    return {
        "combo_keys": combo_keys,
        "strategies": [STRATEGY_NAMES[k] for k in ORDERED_KEYS if k in combo_keys],
        "score":      round(float(score), 1),
        "description": description,
    }


def _normalize_engine_row(row: dict) -> dict:
    """Normalize one row from utils.evaluate_all_policies to the shared schema."""
    flags = row.get("policy_flags", {})
    combo_keys = frozenset(
        ENGINE_KEY_MAP[k] for k, v in flags.items() if v and k in ENGINE_KEY_MAP
    )
    score = row.get("resilience_score", row.get("score", 0.0))
    return _normalize_combo(combo_keys, score, description=None)


def _normalize_policy_engine_row(row: dict) -> dict:
    """Normalize one row from a (future) policy_engine.compare_policies result.

    Accepts a permissive shape so a teammate engine can supply either frontend
    keys or engine flag-names, an explicit score, and an optional description.
    """
    raw_keys = row.get("combo_keys") or row.get("strategies") or row.get("active_policies") or []
    combo_keys = frozenset(
        ENGINE_KEY_MAP.get(k, k) for k in raw_keys if ENGINE_KEY_MAP.get(k, k) in STRATEGY_NAMES
    )
    score = row.get("score", row.get("resilience_score", 0.0))
    return _normalize_combo(combo_keys, score, description=row.get("description"))


def _make_live_compare(compare_policies, normalizer, needs_chdir: bool):
    """Wrap a live comparison callable so it returns the normalized schema."""
    def _compare(scenario: dict) -> list:
        if needs_chdir:
            with _chdir(_ROOT):
                rows = compare_policies(scenario)
        else:
            rows = compare_policies(scenario)
        normalized = [normalizer(r) for r in rows]
        return sorted(normalized, key=lambda x: x["score"], reverse=True)
    return _compare


@st.cache_data(ttl=120, show_spinner=False)
def _fallback_compare_combos(sst: float, rainfall: float, drought: float, heat: float) -> list:
    """Local 16-combo ranking fallback, cached on the four climate inputs.

    Evaluates every adaptation combination through the current emulator (model
    or stub) and ranks by Regional Resilience. Used when no live engine is wired
    or when the live call raises.
    """
    emulator = get_backend()["emulator"]
    rows = []
    for c in range(16):
        combo_keys = frozenset(k for bit, k in zip((1, 2, 4, 8), ORDERED_KEYS) if c & bit)
        adapt = {k: int(k in combo_keys) for k in ORDERED_KEYS}
        r = emulator(sst, rainfall, drought, heat, adapt)
        rows.append(_normalize_combo(combo_keys, r["regional_resilience"]))
    return sorted(rows, key=lambda x: x["score"], reverse=True)


@st.cache_data(ttl=120, show_spinner=False)
def _live_compare_cached(sst: float, rainfall: float, drought: float, heat: float) -> list:
    """Cached wrapper around the live engine (hot path). May raise; not caught here."""
    backend = get_backend()
    scenario = {"sst": sst, "rainfall": rainfall, "drought": drought, "heat": heat}
    return backend["compare"](scenario)


def _utils_fn_ok(fn, min_params: int) -> bool:
    """Return True if fn is callable with at least min_params positional args."""
    try:
        return len(inspect.signature(fn).parameters) >= min_params
    except (TypeError, ValueError):
        return False


@st.cache_resource
def get_backend() -> dict:
    """Build the backend adapter: real teammate modules where available, stubs otherwise.

    Logged, never raises. Keys:
      source          — "stub" | "partial" | "live"
      emulator        — (sst, rainfall, drought, heat, adapt_dict) -> dict
      zscore          — (value, mean, std) -> float
      policy_rules    — () -> dict
      thresholds      — () -> dict
      map_inputs      — (sst, rainfall, drought, heat, reservoir, [year]) -> dict
      compare         — (scenario) -> normalized ranked list
      compare_is_live — bool (True only when backed by a real engine)
      detail          — human-readable wiring notes for the Dev/QA panel
    """
    backend: dict = {
        "source":          "stub",
        "emulator":        run_climate_emulator,
        "zscore":          calculate_zscore,
        "policy_rules":    load_policy_rules,
        "thresholds":      load_risk_thresholds,
        "map_inputs":      stub_map_inputs,
        "compare":         None,           # set below (fallback by default)
        "compare_is_live": False,
        "detail":          {},
    }
    detail = backend["detail"]

    utils = None
    # ── Teammate 3: src/utils.py decision-support functions ───────────────────
    try:
        if _ROOT not in sys.path:
            sys.path.insert(0, _ROOT)
        import src.utils as _utils  # noqa: PLC0415

        z_ok   = _utils_fn_ok(getattr(_utils, "calculate_zscore", None), 3)
        pr_ok  = _utils_fn_ok(getattr(_utils, "load_policy_rules", None), 0)
        th_ok  = _utils_fn_ok(getattr(_utils, "load_thresholds", None), 0)
        map_ok = _utils_fn_ok(getattr(_utils, "map_ui_inputs_to_features", None), 5)

        if z_ok and pr_ok and th_ok:
            utils = _utils
            backend.update(
                zscore=_utils.calculate_zscore,
                policy_rules=lambda: _utils.load_policy_rules(_POLICY_RULES_PATH),
                thresholds=lambda: _utils.load_thresholds(_RISK_THRESH_PATH),
                source="partial",
            )
            if map_ok:
                backend["map_inputs"] = _utils.map_ui_inputs_to_features
            detail["utils"] = "loaded (zscore/policy/thresholds" + (" +map_inputs)" if map_ok else ")")
            logging.info("src.utils loaded — live decision-support functions active")
        else:
            detail["utils"] = "found but incomplete — using stubs"
            logging.info("src.utils found but functions not fully implemented")
    except Exception as exc:  # noqa: BLE001 — never break the UI on a bad import
        detail["utils"] = f"not loaded ({type(exc).__name__})"
        logging.info("src.utils not usable — using stubs: %s", exc)

    # Stats needed by the model emulator + live policy engine.
    stats = load_historical_stats()

    # ── Teammate 1: trained surrogate models ──────────────────────────────────
    crop_model = water_model = None
    try:
        import joblib  # lazy — app works without scikit-learn / joblib installed

        crop_path  = os.path.join(_MODELS_DIR, "crop_model.joblib")
        water_path = os.path.join(_MODELS_DIR, "water_model.joblib")
        if (
            os.path.exists(crop_path) and os.path.getsize(crop_path) > 0
            and os.path.exists(water_path) and os.path.getsize(water_path) > 0
        ):
            crop_model  = joblib.load(crop_path)
            water_model = joblib.load(water_path)
            backend["emulator"] = make_model_emulator(
                crop_model, water_model, backend["map_inputs"], stats
            )
            backend["source"] = "live"
            detail["models"] = "loaded (crop + water surrogates)"
            logging.info("Live models loaded from %s", _MODELS_DIR)
        else:
            detail["models"] = "absent/empty — stub emulator"
            logging.info("Model files absent or empty — using stub emulator")
    except Exception as exc:  # noqa: BLE001
        detail["models"] = f"load failed ({type(exc).__name__}) — stub emulator"
        logging.warning("Model load failed — using stub emulator: %s", exc)

    # ── Teammate 1: policy-comparison engine (Step 3.5) ───────────────────────
    # Preference order: policy_engine.compare_policies → utils.evaluate_all_policies.
    compare_wired = False
    try:
        from policy_engine import compare_policies as _compare_policies  # noqa: PLC0415

        backend["compare"] = _make_live_compare(
            lambda scenario: _compare_policies(scenario),
            _normalize_policy_engine_row,
            needs_chdir=True,
        )
        backend["compare_is_live"] = True
        compare_wired = True
        detail["compare"] = "live (policy_engine.compare_policies)"
        logging.info("policy_engine.compare_policies wired as live comparison engine")
    except Exception:  # noqa: BLE001 — module simply not present in this build
        pass

    if not compare_wired and utils is not None and crop_model is not None:
        eval_ok = _utils_fn_ok(getattr(utils, "evaluate_all_policies", None), 5)
        if eval_ok:
            policy_rules = backend["policy_rules"]()

            def _engine(scenario: dict) -> list:
                climate_inputs = backend["map_inputs"](
                    sst_anomaly=scenario["sst"],
                    rainfall_deficit_pct=scenario["rainfall"],
                    drought_duration=scenario["drought"],
                    heat_stress=scenario["heat"],
                    reservoir_storage_pct=max(
                        10.0, stats.get("water_reservoir_security", {}).get("max", WATER_NORM_MAX)
                        * (1 - scenario["rainfall"] / 100)
                    ),
                )
                return utils.evaluate_all_policies(
                    climate_inputs, crop_model, water_model, stats, policy_rules, top_n=16
                )

            backend["compare"] = _make_live_compare(_engine, _normalize_engine_row, needs_chdir=True)
            backend["compare_is_live"] = True
            compare_wired = True
            detail["compare"] = "live (utils.evaluate_all_policies)"
            logging.info("utils.evaluate_all_policies wired as live comparison engine")

    if not compare_wired:
        # Fallback: the cached local 16-combo loop. Wrapped so the call signature
        # matches the live engines (takes a scenario dict).
        backend["compare"] = lambda scenario: _fallback_compare_combos(
            scenario["sst"], scenario["rainfall"], scenario["drought"], scenario["heat"]
        )
        backend["compare_is_live"] = False
        detail.setdefault("compare", "local 16-combo fallback")

    return backend


# ═════════════════════════════════════════════════════════════════════════════
# HISTORICAL STATISTICS LOADER  (FIX-6 / FIX-7)
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_historical_stats() -> dict:
    """Load data/historical_statistics.json, falling back to MOCK_STATS per key.

    Every dict entry with the required mean/std/min/max keys is absorbed (this
    includes the *_ui display-scale stats the Z-score panel needs and the raw
    target stats the policy engine needs). ``_using_mock`` is True when any
    MOCK_STATS key had to be substituted. Input-feature bounds (sst_anomaly,
    rainfall_deficit, …) are absorbed when present so they drive the OOD check.
    """
    result: dict = {k: dict(v) for k, v in MOCK_STATS.items()}
    result["_using_mock"] = True

    try:
        with open(_HIST_STATS_PATH, encoding="utf-8") as fh:
            raw: dict = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return result  # all mock

    for indicator, entry in raw.items():
        if indicator.startswith("_"):
            continue
        if isinstance(entry, dict) and _REQUIRED_STAT_KEYS.issubset(entry):
            result[indicator] = {k: entry[k] for k in _REQUIRED_STAT_KEYS}

    any_mock = any(key not in raw for key in MOCK_STATS)

    for input_key in ("sst_anomaly", "rainfall_deficit", "drought_months", "heat_stress"):
        if input_key in raw and isinstance(raw[input_key], dict):
            result[input_key] = raw[input_key]

    result["_using_mock"] = any_mock
    return result


def _is_ood(sst: float, rainfall: float, drought: float, heat: float, hist_stats: dict) -> bool:
    """True if any input exceeds the historical training max (FIX-7)."""
    def _max(key: str, fallback: float) -> float:
        return hist_stats.get(key, {}).get("max", fallback)

    return (
        sst       > _max("sst_anomaly",      OOD_DEFAULTS["sst_anomaly"])
        or rainfall > _max("rainfall_deficit", OOD_DEFAULTS["rainfall_deficit"])
        or drought  > _max("drought_months",   OOD_DEFAULTS["drought_months"])
        or heat     > _max("heat_stress",      OOD_DEFAULTS["heat_stress"])
    )


def get_ood_warnings(
    sst: float, rainfall: float, drought: float, heat: float, hist_stats: dict
) -> list:
    """Single source of truth for OOD warnings (Step 3.7).

    Returns the exact required copy in a list when the scenario is out of
    distribution, else an empty list. The same list feeds the alert bar, the
    transparency expander, and the exported reports so the copy is identical
    everywhere.
    """
    if _is_ood(sst, rainfall, drought, heat, hist_stats):
        return [OOD_WARNING_TEXT]
    return []


# ═════════════════════════════════════════════════════════════════════════════
# DIRECTIVE + POLICY HELPERS  (Step 3.6)
# ═════════════════════════════════════════════════════════════════════════════

def _effective_z(z: float, reversed_dir: bool) -> float:
    """Flip the sign for indicators where a higher value is worse."""
    return -z if reversed_dir else z


def classify_directive(z: float, reversed_dir: bool) -> str:
    """Map a Z-score to its directive band (display contract)."""
    effective = _effective_z(z, reversed_dir)
    if effective < Z_DIRECTIVE_CRITICAL:
        return DIRECTIVE_CRITICAL
    if effective < Z_DIRECTIVE_ELEVATED:
        return DIRECTIVE_ELEVATED
    if effective < Z_DIRECTIVE_MONITOR:
        return DIRECTIVE_MONITOR
    return DIRECTIVE_NORMAL


def _policy_guidance(policy_rules: dict, category: str) -> str:
    """Extract human-readable guidance for a planning category from policy rules.

    Handles both the live nested schema (``{"normal": {"description", "recommendations"}}``)
    and the flat stub schema (``{"Normal": "text"}``).
    """
    entry = policy_rules.get(category)
    if isinstance(entry, dict):
        desc = entry.get("description", "").strip()
        recs = entry.get("recommendations", [])
        if recs:
            first = recs[0]
            text = first.get("text", "") if isinstance(first, dict) else str(first)
            if text:
                return f"{desc} First action: {text}".strip()
        return desc
    if isinstance(entry, str):
        return entry
    stub_key = _STUB_POLICY_KEYS.get(category)
    if stub_key and isinstance(policy_rules.get(stub_key), str):
        return policy_rules[stub_key]
    return "No guidance available for this directive."


def _directive_cell_style(val: str) -> str:
    """pandas Styler callback: colour the Directive cell by its band."""
    for label, color in DIRECTIVE_COLORS.items():
        if isinstance(val, str) and val.startswith(label):
            return f"background-color: {color}; color: white; font-weight: 600;"
    return ""


# ═════════════════════════════════════════════════════════════════════════════
# REPORT GENERATORS
# ═════════════════════════════════════════════════════════════════════════════

def generate_txt_report(
    scenario: dict,
    results: dict,
    top_strategy: dict,
    warnings: list,
) -> str:
    """Generate the plain-text action briefing. ``warnings`` is the OOD list."""
    from datetime import datetime

    active = [k for k, v in scenario["adapt_dict"].items() if v] or ["None selected"]
    if warnings:
        warning_block = "\n".join("⚠️  " + w for w in warnings)
    else:
        warning_block = "✅  Scenario within historical training range (1991–2017)."

    return f"""
═══════════════════════════════════════════════════
AI CLIMATE RESILIENCE SANDBOX — ACTION BRIEFING
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M UTC")}
═══════════════════════════════════════════════════

SCENARIO PARAMETERS
───────────────────
Sea Surface Temperature Anomaly : {scenario["sst"]}°C
Monsoon Rainfall Deficit        : {scenario["rainfall"]}%
Drought Duration                : {scenario["drought"]} months
Heat Stress Index               : {scenario["heat"]} / 5

ADAPTATION MEASURES ACTIVE
──────────────────────────
{chr(10).join("  ✓ " + a for a in active)}

KEY RISK INDICATORS
───────────────────
Crop Yield Stability       : {results["crop_yield_stability"]}%
Water Reservoir Security   : {results["water_reservoir_security"]}%
Groundwater Stress         : {results["groundwater_stress"]}%
Agricultural Risk Exposure : {results["agricultural_risk"]} / 5.0
Regional Resilience Score  : {results["regional_resilience"]}%

TOP RECOMMENDED STRATEGY
─────────────────────────
Strategies : {", ".join(top_strategy["strategies"]) or "None"}
Projected Resilience : {top_strategy["score"]}%

WARNINGS
────────
{warning_block}
Model: Random Forest Regressor | Training period: 1991–2017

───────────────────────────────────────────────────
DISCLAIMER
This briefing is generated by an AI surrogate model for
scenario planning purposes only. It does not constitute
an operational forecast or official government guidance.
═══════════════════════════════════════════════════
"""


def generate_pdf_report(
    scenario: dict,
    results: dict,
    top_strategy: dict,
    warnings: list,
) -> bytes:
    """Generate the PDF action briefing. ``warnings`` is the OOD list. Returns bytes."""
    from datetime import datetime
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()

    pdf.set_fill_color(26, 107, 107)
    pdf.rect(0, 0, 210, 22, "F")
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(10, 6)
    pdf.cell(0, 10, "AI Climate Resilience Sandbox - Action Briefing")

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(200, 240, 240)
    pdf.set_xy(10, 15)
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")

    pdf.set_text_color(30, 30, 30)
    pdf.set_y(28)

    if warnings:
        pdf.set_fill_color(254, 226, 226)
        pdf.set_draw_color(220, 38, 38)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(153, 27, 27)
        for w in warnings:
            pdf.multi_cell(0, 7, w, border=1, fill=True)
        pdf.set_text_color(30, 30, 30)
        pdf.ln(3)

    def section(title: str) -> None:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_fill_color(240, 253, 250)
        pdf.set_text_color(26, 107, 107)
        pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT", fill=True)
        pdf.set_text_color(30, 30, 30)
        pdf.set_font("Helvetica", "", 10)
        pdf.ln(1)

    def row(label: str, value, unit: str = "") -> None:
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(90, 7, label)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, f"{value}{unit}", new_x="LMARGIN", new_y="NEXT")

    section("SCENARIO PARAMETERS")
    row("SST Anomaly",       scenario["sst"],      " C")
    row("Rainfall Deficit",  scenario["rainfall"], "%")
    row("Drought Duration",  scenario["drought"],  " months")
    row("Heat Stress Index", scenario["heat"],     " / 5")
    pdf.ln(3)

    section("ADAPTATION MEASURES ACTIVE")
    active = [k for k, v in scenario["adapt_dict"].items() if v]
    names = {
        "crops": "Drought-Resistant Crop Program",
        "gw":    "Emergency Groundwater Rationing",
        "irr":   "Supplemental Irrigation Support",
        "water": "Water Conservation Initiative",
    }
    if active:
        for k in active:
            pdf.cell(0, 7, f"  -  {names[k]}", new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.cell(0, 7, "  No measures active", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    section("KEY RISK INDICATORS")
    row("Crop Yield Stability",       results["crop_yield_stability"],     "%")
    row("Water Reservoir Security",   results["water_reservoir_security"], "%")
    row("Groundwater Stress",         results["groundwater_stress"],       "%")
    row("Agricultural Risk Exposure", results["agricultural_risk"],        " / 5.0")
    row("Regional Resilience Score",  results["regional_resilience"],      "%")
    pdf.ln(3)

    section("TOP RECOMMENDED STRATEGY")
    strats = ", ".join(top_strategy["strategies"]) or "None"
    pdf.multi_cell(0, 7, f"Strategies: {strats}")
    row("Projected Resilience", top_strategy["score"], "%")
    pdf.ln(3)

    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(
        0, 5,
        "DISCLAIMER: This briefing is generated by an AI surrogate model "
        "for scenario planning purposes only. It does not constitute an "
        "operational forecast or official government guidance.",
    )
    return bytes(pdf.output())


# ═════════════════════════════════════════════════════════════════════════════
# SESSION STATE + CSS HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _init_session_state() -> None:
    """Seed session_state with defaults on first run."""
    for key, val in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _inject_css() -> None:
    """Inject global theme CSS."""
    st.markdown(
        """
        <style>
        :root {
            --color-primary:  #1a6b6b;
            --color-normal:   #16a34a;
            --color-elevated: #ca8a04;
            --color-severe:   #ea580c;
            --color-critical: #dc2626;
        }
        .stApp { background-color: #ffffff; }
        section[data-testid="stSidebar"] { background-color: #f0fdfa; }
        .crs-card {
            background-color: #ffffff;
            border-radius: 0.625rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.07);
            padding: 1.1rem 1.4rem;
            margin: 0.6rem 0 1rem 0;
            border: 1px solid #e2e8f0;
        }
        .crs-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background-color: var(--color-primary);
            border-radius: 0.625rem 0.625rem 0 0;
            padding: 1.1rem 1.5rem;
        }
        .crs-header h1 {
            color: #ffffff;
            font-weight: 700;
            font-size: 1.75rem;
            margin: 0;
            line-height: 1.2;
        }
        .crs-badge {
            background-color: var(--color-normal);
            color: #ffffff;
            font-size: 0.85rem;
            font-weight: 600;
            padding: 0.35rem 0.9rem;
            border-radius: 999px;
            white-space: nowrap;
        }
        .crs-disclaimer {
            background-color: #fef3c7;
            color: #92400e;
            font-size: 0.85rem;
            font-style: italic;
            padding: 0.6rem 1.5rem;
            border-radius: 0 0 0.625rem 0.625rem;
            border: 1px solid #fde68a;
            margin-bottom: 1rem;
        }
        .crs-summary-climate {
            font-size: 1.05rem;
            font-weight: 600;
            color: var(--color-primary);
            margin-bottom: 0.4rem;
        }
        .crs-summary-adapt { font-size: 0.95rem; color: #334155; }
        .crs-preset-pill {
            display: inline-block;
            background-color: var(--color-primary);
            color: #ffffff;
            font-size: 0.8rem;
            font-weight: 600;
            padding: 0.3rem 0.85rem;
            border-radius: 999px;
            margin-top: 0.6rem;
        }
        /* Preset buttons: color by sidebar column position */
        section[data-testid="stSidebar"] div[data-testid="column"]:nth-of-type(1) button {
            background-color: var(--color-normal); color: #ffffff; border: none;
        }
        section[data-testid="stSidebar"] div[data-testid="column"]:nth-of-type(2) button {
            background-color: var(--color-severe); color: #ffffff; border: none;
        }
        section[data-testid="stSidebar"] div[data-testid="column"]:nth-of-type(3) button {
            background-color: var(--color-critical); color: #ffffff; border: none;
        }
        section[data-testid="stSidebar"] div[data-testid="column"] button:hover {
            filter: brightness(1.08); color: #ffffff;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header[data-testid="stHeader"] {background: transparent;}
        html, body, [class*="css"] { font-size: 1rem; }
        @media (max-width: 768px) {
            [data-testid="column"] { min-width: 100% !important; margin-bottom: 0.5rem; }
            h1 { font-size: 1.3rem !important; }
            h2 { font-size: 1.1rem !important; }
            .crs-card { padding: 10px !important; }
            .stDownloadButton > button { width: 100% !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# PRESET CALLBACKS
# ═════════════════════════════════════════════════════════════════════════════

def _apply_preset(preset: dict, preset_name: str) -> None:
    """on_click callback — loads preset values into session_state."""
    for key, val in preset.items():
        st.session_state[key] = val
    st.session_state["active_preset"] = preset_name


def _reset_defaults() -> None:
    """on_click callback — restores all controls to defaults."""
    for key, val in SESSION_DEFAULTS.items():
        st.session_state[key] = val
    st.session_state["active_preset"] = "custom"


# ═════════════════════════════════════════════════════════════════════════════
# HOT-PATH COMPUTE HELPERS  (P3-5: cache the emulator)
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=120, show_spinner=False)
def _emulate_cached(
    sst: float, rainfall: float, drought: float, heat: float,
    crops: int, gw: int, irr: int, water: int,
) -> dict:
    """Cached single emulator evaluation — covers the dashboard hot path."""
    emulator = get_backend()["emulator"]
    adapt = {"crops": crops, "gw": gw, "irr": irr, "water": water}
    return emulator(sst, rainfall, drought, heat, adapt)


# ═════════════════════════════════════════════════════════════════════════════
# STATUS-PILL HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _status_higher_better(value: float) -> tuple:
    """Status tier for metrics where higher = better (e.g. crop yield)."""
    if value >= 70:
        return "Normal",   "#16a34a"
    if value >= 50:
        return "Elevated", "#ca8a04"
    if value >= 30:
        return "Severe",   "#ea580c"
    return "Critical", "#dc2626"


def _status_gw_stress(value: float) -> tuple:
    """FIX-4: Status tier for groundwater stress — lower is better (reversed scale)."""
    if value <= 30:
        return "Normal",   "#16a34a"
    if value <= 50:
        return "Elevated", "#ca8a04"
    if value <= 69:
        return "Severe",   "#ea580c"
    return "Critical", "#dc2626"


def _status_ag_risk(value: float) -> tuple:
    """Status tier for agricultural risk on 0–5 scale (lower = better)."""
    if value <= 1.5:
        return "Normal",   "#16a34a"
    if value <= 2.5:
        return "Elevated", "#ca8a04"
    if value <= 3.5:
        return "Severe",   "#ea580c"
    return "Critical", "#dc2626"


# ═════════════════════════════════════════════════════════════════════════════
# RENDER FUNCTIONS  (FIX-2: all UI here, called from main())
# ═════════════════════════════════════════════════════════════════════════════

def render_sidebar(backend: dict) -> tuple:
    """Render the sidebar controls + engine badge. Returns the scenario inputs."""
    source = backend.get("source", "stub")
    badge_map = {
        "live":    "🟢 Live models",
        "partial": "🟡 Partial (utils only)",
        "stub":    "⚪ Stub mode",
    }

    with st.sidebar:
        st.markdown("### 🌍 Climate Resilience Sandbox")
        st.caption(f"Engine: {badge_map.get(source, '⚪ Stub mode')}")
        st.markdown("---")

        st.subheader("🌡️ Climate Scenario Inputs")

        st.slider(
            "Sea Surface Temperature Anomaly (°C)",
            min_value=0.0, max_value=3.5, step=0.1,
            key="sst_anomaly",
            help="SST anomaly above baseline. >2.5°C = extreme El Niño.",
        )
        st.slider(
            "Monsoon Rainfall Deficit (%)",
            min_value=0, max_value=100, step=1,
            key="rainfall_deficit",
            help="Reduction in expected seasonal monsoon rainfall.",
        )
        st.slider(
            "Drought Duration (months)",
            min_value=0, max_value=12, step=1,
            key="drought_months",
            help="Consecutive months of below-normal rainfall.",
        )
        st.slider(
            "Heat Stress Index (1–5)",
            min_value=1, max_value=5, step=1,
            key="heat_stress",
            help="1=Mild  2=Low  3=Moderate  4=High  5=Extreme",
        )

        st.subheader("🌾 Adaptation Measures")
        st.checkbox("Drought-Resistant Crop Program", key="adapt_crops",
                    help="Short-cycle, drought-tolerant seed deployment.")
        st.checkbox("Emergency Groundwater Rationing", key="adapt_groundwater",
                    help="Volumetric water-use limits across irrigation zones.")
        st.checkbox("Supplemental Irrigation Support", key="adapt_irrigation",
                    help="Activate supplemental micro-irrigation infrastructure.")
        st.checkbox("Water Conservation Initiative", key="adapt_water",
                    help="Community rainwater harvesting and conservation.")

        st.subheader("🚀 Quick Presets")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.button("🌤 Moderate", use_container_width=True,
                      on_click=_apply_preset, args=(MODERATE_EL_NINO, "moderate_el_nino"))
        with col2:
            st.button("🔥 Severe", use_container_width=True,
                      on_click=_apply_preset, args=(SEVERE_DROUGHT, "severe_drought"))
        with col3:
            st.button("⚡ Extreme", use_container_width=True,
                      on_click=_apply_preset, args=(EXTREME_ENSO, "extreme_enso"))

        st.button("↺ Reset to Default", use_container_width=True, on_click=_reset_defaults)

        with st.expander("ℹ️ About this tool"):
            st.markdown(
                """
                **AI Climate Resilience Sandbox** is a web-based decision-support
                platform for exploring extreme El Niño–Southern Oscillation (ENSO)
                climate scenarios and comparing adaptation strategies.

                **Target users:** District administrators, agricultural cooperatives,
                water management boards, rural development agencies, and climate NGOs.

                **Note:** This platform is for scenario planning only. It does not
                provide authoritative forecasts and should not replace operational
                weather or emergency-management guidance.
                """
            )

    adapt_dict = {
        "crops": int(st.session_state.adapt_crops),
        "gw":    int(st.session_state.adapt_groundwater),
        "irr":   int(st.session_state.adapt_irrigation),
        "water": int(st.session_state.adapt_water),
    }
    return (
        st.session_state.sst_anomaly,
        st.session_state.rainfall_deficit,
        st.session_state.drought_months,
        st.session_state.heat_stress,
        adapt_dict,
    )


def render_header() -> None:
    """Render the teal header banner and amber disclaimer bar."""
    st.markdown(
        """
        <div class="crs-header">
            <h1>🌍&nbsp; AI Climate Resilience Sandbox</h1>
            <span class="crs-badge">SDG 13: Climate Action</span>
        </div>
        <div class="crs-disclaimer">
            ⚠️ Decision-support tool only — not an operational forecast.
            All results are exploratory scenario estimates.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_scenario_summary(
    sst: float, rainfall: float, drought: float, heat: float, adapt_dict: dict,
) -> None:
    """Render the scenario summary bar below the header."""
    climate_desc = (
        f"SST +{sst}°C  |  {rainfall}% Rainfall Deficit  |  "
        f"{drought} Month Drought  |  Heat Level {heat}/5"
    )
    active_adaptations = [
        name for name, key in {
            "Drought-Resistant Crops": "crops",
            "Groundwater Rationing":   "gw",
            "Supplemental Irrigation": "irr",
            "Water Conservation":      "water",
        }.items()
        if adapt_dict.get(key)
    ]
    adapt_desc = (
        "Adaptations Active: " + ", ".join(active_adaptations)
        if active_adaptations else "⚠️ No adaptation measures active"
    )
    preset_html = ""
    if st.session_state["active_preset"] != "custom":
        label = PRESET_LABELS.get(st.session_state["active_preset"], "Custom")
        preset_html = f'<div class="crs-preset-pill">Preset: {label}</div>'

    st.markdown(
        f"""
        <div class="crs-card">
            <div class="crs-summary-climate">📋&nbsp; {climate_desc}</div>
            <div class="crs-summary-adapt">{adapt_desc}</div>
            {preset_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_alerts(adj: float, policy_rules: dict, ood_warnings: list) -> None:
    """Render the OOD warning(s) then the risk-level status alert, above the cards.

    P3-4: ``ood_warnings`` is the single shared list — its exact copy is rendered
    here, in the transparency expander, and in the exports.
    """
    for w in ood_warnings:
        st.error("⚠️ " + w)

    if adj >= RISK_SEVERE:
        st.error("🔴 Critical Resource Alert — " + _policy_guidance(policy_rules, "critical"))
    elif adj >= RISK_ELEVATED:
        st.warning("🟠 Severe Stress — " + _policy_guidance(policy_rules, "severe"))
    elif adj >= RISK_NORMAL:
        st.warning("🟡 Elevated Risk — " + _policy_guidance(policy_rules, "elevated"))
    else:
        st.success("🟢 Normal Conditions — " + _policy_guidance(policy_rules, "normal"))


def render_metric_cards(results: dict, baseline: dict) -> None:
    """Render the five st.metric() cards inside a bordered container."""
    st.subheader("📊 Key Resilience Indicators")

    metric_defs = [
        {"label": "🌾 Crop Yield Stability",       "value": results["crop_yield_stability"],
         "base": baseline["crop_yield_stability"],     "unit": "%",     "delta_color": "normal",  "status_fn": _status_higher_better},
        {"label": "💧 Water Reservoir Security",   "value": results["water_reservoir_security"],
         "base": baseline["water_reservoir_security"], "unit": "%",     "delta_color": "normal",  "status_fn": _status_higher_better},
        {"label": "🏜 Groundwater Stress",         "value": results["groundwater_stress"],
         "base": baseline["groundwater_stress"],       "unit": "%",     "delta_color": "inverse", "status_fn": _status_gw_stress},
        {"label": "⚠️ Agricultural Risk Exposure", "value": results["agricultural_risk"],
         "base": baseline["agricultural_risk"],        "unit": "/ 5.0", "delta_color": "inverse", "status_fn": _status_ag_risk},
        {"label": "🛡 Regional Resilience Score",   "value": results["regional_resilience"],
         "base": baseline["regional_resilience"],      "unit": "%",     "delta_color": "normal",  "status_fn": _status_higher_better},
    ]

    with st.container(border=True):
        cols = st.columns(5)
        for col, m in zip(cols, metric_defs):
            delta = m["value"] - m["base"]
            status_label, status_color = m["status_fn"](m["value"])
            with col:
                st.metric(
                    label=m["label"],
                    value=f"{m['value']}{m['unit']}",
                    delta=f"{delta:+.1f}{m['unit']}",
                    delta_color=m["delta_color"],
                )
                st.markdown(
                    f'<span style="background:{status_color};color:white;'
                    f'padding:2px 10px;border-radius:12px;font-size:0.75rem;'
                    f'font-weight:600;">{status_label}</span>',
                    unsafe_allow_html=True,
                )


def render_risk_map(adj: float) -> None:
    """Render the interactive district risk map (or treemap fallback)."""
    st.subheader("📍 Regional Risk Distribution")

    DISTRICTS = [
        {"name": "Kanchipuram",      "lat": 12.83, "lon": 79.70},
        {"name": "Cuddalore",        "lat": 11.75, "lon": 79.77},
        {"name": "Vellore",          "lat": 12.92, "lon": 79.13},
        {"name": "Salem",            "lat": 11.67, "lon": 78.15},
        {"name": "Coimbatore",       "lat": 11.01, "lon": 76.97},
        {"name": "Tiruchirappalli",  "lat": 10.79, "lon": 78.70},
        {"name": "Thanjavur",        "lat": 10.79, "lon": 79.14},
        {"name": "Madurai",          "lat":  9.93, "lon": 78.12},
        {"name": "Ramanathapuram",   "lat":  9.37, "lon": 78.83},
        {"name": "Tirunelveli",      "lat":  8.73, "lon": 77.70},
        {"name": "The Nilgiris",     "lat": 11.49, "lon": 76.73},
        {"name": "Kanyakumari",      "lat":  8.09, "lon": 77.55},
    ]
    OFFSETS = [0.10, -0.12, 0.08, -0.15, 0.05, -0.09, 0.13, -0.07, 0.11, -0.10, 0.06, -0.08]
    COLOR_MAP = {
        "Normal":         "#16a34a",
        "Elevated Risk":  "#ca8a04",
        "Severe Stress":  "#ea580c",
        "Critical Alert": "#dc2626",
    }

    rows = []
    for i, d in enumerate(DISTRICTS):
        risk_i = max(0.0, min(1.0, adj + OFFSETS[i]))
        if risk_i >= RISK_SEVERE:
            label_i = "Critical Alert"
        elif risk_i >= RISK_ELEVATED:
            label_i = "Severe Stress"
        elif risk_i >= RISK_NORMAL:
            label_i = "Elevated Risk"
        else:
            label_i = "Normal"
        rows.append({
            "District":             d["name"],
            "lat":                  d["lat"],
            "lon":                  d["lon"],
            "Risk Level":           label_i,
            "Resilience Score (%)": round((1 - risk_i) * 100, 1),
            "size":                 18,
        })

    map_df = pd.DataFrame(rows)

    try:
        fig_map = px.scatter_map(
            map_df, lat="lat", lon="lon",
            color="Risk Level",
            color_discrete_map=COLOR_MAP,
            size="size", size_max=18,
            hover_name="District",
            hover_data={
                "Risk Level": True, "Resilience Score (%)": True,
                "lat": False, "lon": False, "size": False,
            },
            zoom=6,
            center={"lat": 10.5, "lon": 78.5},
            height=420,
        )
        fig_map.update_layout(
            map_style="carto-positron",
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            legend_title_text="Risk Level",
        )
        st.plotly_chart(
            fig_map, use_container_width=True,
            config={"scrollZoom": True, "displayModeBar": False, "displaylogo": False},
            key="risk_map",
        )
    except Exception:  # noqa: BLE001
        fig_tree = px.treemap(
            map_df, path=["District"], values=[1] * len(map_df),
            color="Resilience Score (%)",
            color_continuous_scale=["#dc2626", "#ca8a04", "#16a34a"],
            hover_data={"Risk Level": True, "Resilience Score (%)": True},
            height=420,
        )
        fig_tree.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0})
        st.plotly_chart(
            fig_tree, use_container_width=True,
            config={"displayModeBar": False, "displaylogo": False},
            key="risk_map",
        )

    st.markdown(
        "🟢 Normal&nbsp;&nbsp; 🟡 Elevated Risk&nbsp;&nbsp; "
        "🟠 Severe Stress&nbsp;&nbsp; 🔴 Critical Alert"
    )


# Frontend strategy descriptions — used when the live engine supplies none.
STRATEGY_DESCRIPTIONS: dict = {
    frozenset(["crops", "water"]):
        "Reduces crop failure while preserving groundwater. Effective for moderate droughts.",
    frozenset(["crops", "gw", "irr", "water"]):
        "Maximum resilience. Requires significant coordination and resources across all sectors.",
    frozenset(["gw", "irr"]):
        "Balances short-term water access with long-term aquifer health.",
    frozenset(["crops", "gw"]):
        "Cost-effective pairing for early-stage drought response.",
    frozenset(["irr", "water"]):
        "Strong for water security; moderate crop protection.",
}
DEFAULT_STRATEGY_DESC = (
    "Moderate improvement. Consider pairing with additional measures for stronger outcomes."
)


def render_recommendations(adapt_dict: dict, all_combos: list) -> None:
    """Render the top-3 ranked strategy recommendation cards (Step 3.5).

    Consumes the normalized comparison schema. Live-supplied descriptions are
    preferred; otherwise the frontend STRATEGY_DESCRIPTIONS dict is used. The
    card matching the user's current checkboxes gets the "Currently Active"
    highlight.
    """
    st.subheader("🏆 Recommended Strategies")

    RANK_BADGES = ["🥇", "🥈", "🥉"]
    RANK_LABELS = ["Best Strategy", "Runner-Up", "Third Option"]

    active_keys = frozenset(k for k, v in adapt_dict.items() if v)
    top3 = all_combos[:3]

    for i, item in enumerate(top3):
        combo_keys = item["combo_keys"]
        strategies = item["strategies"]
        desc = item.get("description") or STRATEGY_DESCRIPTIONS.get(combo_keys, DEFAULT_STRATEGY_DESC)
        is_active = combo_keys == active_keys

        border = "3px solid #0d9488" if is_active else "1px solid #ccfbf1"
        bg     = "#f0fdfa"          if is_active else "#ffffff"

        chips = "  ".join(
            f'<span style="background:#dcfce7;color:#166534;'
            f'padding:2px 8px;border-radius:10px;font-size:0.75rem;">✅ {s}</span>'
            for s in (strategies or ["No measures"])
        )
        active_note = (
            "<div style='color:#0d9488;font-size:0.75rem;'>← Currently Active</div>"
            if is_active else ""
        )

        st.markdown(
            f"""
            <div style="border:{border};background:{bg};border-radius:10px;
                        padding:12px 14px;margin-bottom:12px;">
              <div style="font-size:1.1rem;font-weight:700;color:#1a6b6b;">
                {RANK_BADGES[i]} {RANK_LABELS[i]}
              </div>
              <div style="margin:6px 0;">{chips}</div>
              <div style="color:#374151;font-size:0.85rem;">{desc}</div>
              <div style="color:#0d9488;font-weight:600;margin-top:6px;">
                Resilience Score: {item["score"]}%
              </div>
              {active_note}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_radar_chart(results: dict, baseline: dict) -> None:
    """Render the resilience radar chart (with vs. without adaptations)."""
    st.subheader("📈 Resilience Profile — With vs. Without Adaptations")

    categories = [
        "Crop Yield Stability", "Water Security", "Groundwater Health",
        "Low Agricultural Risk", "Regional Resilience",
    ]
    with_vals = [
        results["crop_yield_stability"],
        results["water_reservoir_security"],
        100 - results["groundwater_stress"],
        100 - (results["agricultural_risk"] / 5 * 100),
        results["regional_resilience"],
    ]
    without_vals = [
        baseline["crop_yield_stability"],
        baseline["water_reservoir_security"],
        100 - baseline["groundwater_stress"],
        100 - (baseline["agricultural_risk"] / 5 * 100),
        baseline["regional_resilience"],
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=without_vals + [without_vals[0]],
        theta=categories + [categories[0]],
        fill="toself", name="Without Adaptations",
        line={"color": "#f97316", "dash": "dash"},
        fillcolor="rgba(249,115,22,0.1)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=with_vals + [with_vals[0]],
        theta=categories + [categories[0]],
        fill="toself", name="With Current Adaptations",
        line={"color": "#0d9488"},
        fillcolor="rgba(13,148,136,0.15)",
    ))
    fig.update_layout(
        polar={
            "radialaxis": {"visible": True, "range": [0, 100], "ticksuffix": "%"},
            "angularaxis": {"rotation": 90},
        },
        showlegend=True, paper_bgcolor="rgba(0,0,0,0)", height=380,
        margin={"l": 40, "r": 40, "t": 40, "b": 40},
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_zscore_panel(results: dict, hist_stats: dict, backend: dict, policy_rules: dict) -> None:
    """Render the Z-score table + dynamic Active Directive card (Step 3.6).

    A Z-score is computed per indicator from the live results and historical
    stats (via the adapter's zscore callable), mapped to a directive band
    (reversed for Groundwater Stress and Agricultural Risk), and shown in a
    colour-styled table. The single worst indicator drives the "Active Directive"
    card, which quotes guidance text pulled from backend["policy_rules"].
    """
    st.subheader("📐 Z-Score Risk Thresholds & Directive Routing")

    zscore_fn = backend["zscore"]
    rows = []
    worst = None  # (effective_z, display_name, directive_label)

    for result_key, (display_name, unit, reversed_dir, stats_key) in ZSCORE_META.items():
        value = results[result_key]
        stats = hist_stats.get(stats_key, MOCK_STATS.get(stats_key, {"mean": 0, "std": 1}))
        z = round(float(zscore_fn(value, stats["mean"], stats["std"])), 2)
        directive = classify_directive(z, reversed_dir)
        eff = _effective_z(z, reversed_dir)

        rows.append({
            "Indicator":     display_name,
            "Current Value": f"{value}{unit}",
            "Hist. Mean":    f"{round(stats['mean'], 2)}{unit}",
            "Z-Score":       z,
            "Directive":     directive,
        })
        if worst is None or eff < worst[0]:
            worst = (eff, display_name, directive)

    df = pd.DataFrame(rows)
    styled = df.style.map(_directive_cell_style, subset=["Directive"])
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.caption(
        "Z-score = standard deviations from the 1991–2017 historical mean. "
        "Bands: z < −2.0 Critical · −2.0…−1.0 Elevated · −1.0…0 Monitoring · "
        "z ≥ 0 Normal (sign reversed for Groundwater Stress & Agricultural Risk)."
    )

    # ── 📌 Active Directive (the user-visible result of directive routing) ────
    if worst is not None:
        _eff, worst_name, worst_directive = worst
        color = DIRECTIVE_COLORS.get(worst_directive, "#1a6b6b")
        category = DIRECTIVE_POLICY_KEY.get(worst_directive, "normal")
        guidance = _policy_guidance(policy_rules, category)
        st.markdown(
            f"""
            <div style="border-left:6px solid {color};background:#f8fafc;
                        border-radius:8px;padding:12px 16px;margin-top:10px;">
              <div style="font-size:0.8rem;font-weight:700;color:#64748b;
                          text-transform:uppercase;letter-spacing:0.05em;">
                📌 Active Directive
              </div>
              <div style="font-size:1.05rem;font-weight:700;color:{color};margin:4px 0;">
                {worst_directive}
              </div>
              <div style="font-size:0.85rem;color:#475569;margin-bottom:6px;">
                Triggered by worst indicator: <strong>{worst_name}</strong>
              </div>
              <div style="font-size:0.9rem;color:#1f2937;">{guidance}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_transparency(results: dict, ood_warnings: list, hist_stats: dict) -> None:
    """Render the transparency & model-confidence expander (Step 3.7)."""
    confidence = results.get("confidence") if isinstance(results.get("confidence"), dict) else None

    with st.expander("🔍 Transparency & Model Confidence", expanded=False):
        t1, t2, t3 = st.columns(3)

        with t1:
            st.markdown("**📊 Confidence Ranges**")
            INDICATOR_RANGES = {
                "Crop Yield Stability":     ("crop_yield_stability",     "crop_yield_stability_ui",     12.0),
                "Water Reservoir Security": ("water_reservoir_security", "water_reservoir_security_ui", 10.0),
                "Groundwater Stress":       ("groundwater_stress",       "groundwater_stress_ui",       15.0),
                "Agricultural Risk":        ("agricultural_risk",        "agricultural_risk_ui",         0.6),
                "Regional Resilience":      ("regional_resilience",      "regional_resilience",         11.0),
            }
            conf_rows = []
            for name, (result_key, stats_key, default_range) in INDICATOR_RANGES.items():
                stats = hist_stats.get(stats_key, {"mean": 0, "std": 1})
                # Prefer a live per-prediction interval when the engine exposes one.
                rng = default_range
                if confidence and result_key in confidence:
                    rng = confidence[result_key]
                conf_rows.append({
                    "Indicator":  name,
                    "Estimate":   results[result_key],
                    "± Range":    rng,
                    "Hist. Mean": round(stats["mean"], 2),
                })
            st.dataframe(pd.DataFrame(conf_rows), use_container_width=True, hide_index=True)

            if confidence:
                st.caption("Live model prediction intervals.")
            else:
                st.caption("Estimated ranges (model intervals pending).")

            if hist_stats.get("_using_mock", False):
                st.caption("⚠️ Some statistics using fallback values "
                           "(check data/historical_statistics.json).")
            else:
                st.caption("✅ Statistics loaded from real dataset (1991–2017).")

        with t2:
            st.markdown("**🤖 Model Information**")
            st.markdown(
                """
                - **Model:** Random Forest Regressor (Surrogate Emulator)
                - **Training period:** 1991–2017
                - **Features:** ONI (DJF), Rainfall JJAS, Rainfall OND,
                  Reservoir Storage %, Year, Groundnut Yield
                - **Targets:** Crop Yield, Water Security
                - **Limitations:** Cannot account for sudden geopolitical
                  or infrastructure shocks.
                """
            )

        with t3:
            st.markdown("**⚠️ Distribution Check**")
            if ood_warnings:
                for w in ood_warnings:
                    st.error(w)
            else:
                st.success("✅ All parameters within historical training range (1991–2017).")
            st.markdown(
                """
                **Data Coverage:**
                Region: Tamil Nadu, India
                Source: NOAA ONI, IMD Rainfall, ICRISAT Crop Data, CWC Reservoir Data
                Boundary: 1991–2017 annual observations
                """
            )


def render_export(scenario_data: dict, results: dict, all_combos: list, ood_warnings: list) -> None:
    """Render the export action-briefing section (Step 3.7 warnings parity)."""
    st.divider()
    st.subheader("📄 Export Action Briefing")

    top = all_combos[0]
    top_strategy = {"strategies": top["strategies"], "score": top["score"]}

    txt_content = generate_txt_report(scenario_data, results, top_strategy, ood_warnings)

    csv_df = pd.DataFrame([{
        "SST_Anomaly_C":        scenario_data["sst"],
        "Rainfall_Deficit_pct": scenario_data["rainfall"],
        "Drought_Months":       scenario_data["drought"],
        "Heat_Stress":          scenario_data["heat"],
        "Crop_Yield_Stability": results["crop_yield_stability"],
        "Water_Security":       results["water_reservoir_security"],
        "Groundwater_Stress":   results["groundwater_stress"],
        "Agricultural_Risk":    results["agricultural_risk"],
        "Regional_Resilience":  results["regional_resilience"],
        "OOD_Warning":          "; ".join(ood_warnings) or "None",
    }])

    ec1, ec2, ec3 = st.columns(3)
    with ec1:
        st.download_button(
            "⬇ Download TXT Report", data=txt_content,
            file_name="climate_resilience_briefing.txt", mime="text/plain",
            use_container_width=True,
        )
    with ec2:
        try:
            pdf_bytes = generate_pdf_report(scenario_data, results, top_strategy, ood_warnings)
            st.download_button(
                "⬇ Download PDF Report", data=pdf_bytes,
                file_name="climate_resilience_briefing.pdf", mime="application/pdf",
                use_container_width=True,
            )
        except ImportError:
            st.info("PDF export requires: pip install fpdf2")
    with ec3:
        st.download_button(
            "⬇ Download CSV Data", data=csv_df.to_csv(index=False),
            file_name="climate_scenario_data.csv", mime="text/csv",
            use_container_width=True,
        )


def render_dev_panel(backend: dict, timings: dict) -> None:
    """Render the hidden Dev/QA sidebar panel (Step 3.4 integration aid)."""
    with st.sidebar:
        with st.expander("🛠 Dev / QA", expanded=False):
            st.caption("Engine wiring")
            st.write({
                "source":          backend.get("source"),
                "compare_is_live": backend.get("compare_is_live"),
                **backend.get("detail", {}),
            })
            st.caption("Last computation")
            total = timings.get("total", 0.0)
            st.write({
                "emulator_s": round(timings.get("emulator", 0.0), 4),
                "compare_s":  round(timings.get("compare", 0.0), 4),
                "total_s":    round(total, 4),
                "compare_used": timings.get("compare_used", "n/a"),
                "cache":      "likely hit" if total < 0.05 else "fresh compute",
            })
            if st.button("Clear caches", use_container_width=True):
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()


def render_footer() -> None:
    """Render the bottom footer."""
    st.divider()
    st.markdown(
        """
        <div style="text-align:center;color:#9ca3af;font-size:0.78rem;
                    padding:16px 0 8px 0;">
          🌍 <strong>AI Climate Resilience Sandbox</strong> &nbsp;·&nbsp;
          SDG 13: Climate Action &nbsp;·&nbsp;
          Built for Hackathon MVP &nbsp;·&nbsp;
          <em>Not an operational forecast tool</em>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Single top-level orchestrator. All st.* calls live in render_*() functions."""
    st.set_page_config(
        page_title="AI Climate Resilience Sandbox",
        page_icon="🌍", layout="wide", initial_sidebar_state="auto",
    )
    _init_session_state()
    _inject_css()

    backend    = get_backend()
    hist_stats = load_historical_stats()

    sst, rainfall, drought, heat, adapt_dict = render_sidebar(backend)
    scenario_data = {
        "sst": sst, "rainfall": rainfall, "drought": drought, "heat": heat,
        "adapt_dict": adapt_dict,
    }

    # ── Core computations (P3-5: timed + cached hot path) ─────────────────────
    timings: dict = {}
    t0 = time.perf_counter()
    results  = _emulate_cached(sst, rainfall, drought, heat,
                               adapt_dict["crops"], adapt_dict["gw"],
                               adapt_dict["irr"], adapt_dict["water"])
    baseline = _emulate_cached(sst, rainfall, drought, heat, 0, 0, 0, 0)
    timings["emulator"] = time.perf_counter() - t0
    adj = results["adjusted_risk_score"]

    # Recommendations — live engine through the adapter, with a guarded fallback.
    t1 = time.perf_counter()
    if backend.get("compare_is_live"):
        try:
            all_combos = _live_compare_cached(sst, rainfall, drought, heat)
            if not all_combos:
                raise ValueError("live comparison returned no rows")
            timings["compare_used"] = "live"
        except Exception as exc:  # noqa: BLE001 — never traceback on a live failure
            logging.warning("Live policy engine failed — falling back: %s", exc)
            st.warning("Live policy engine error — showing local comparison.")
            all_combos = _fallback_compare_combos(sst, rainfall, drought, heat)
            timings["compare_used"] = "fallback (after live error)"
    else:
        all_combos = _fallback_compare_combos(sst, rainfall, drought, heat)
        timings["compare_used"] = "fallback (no live engine)"
    timings["compare"] = time.perf_counter() - t1
    timings["total"] = timings["emulator"] + timings["compare"]
    st.session_state["_timings"] = timings

    policy_rules = backend["policy_rules"]()
    ood_warnings = get_ood_warnings(sst, rainfall, drought, heat, hist_stats)

    # ── Main area ─────────────────────────────────────────────────────────────
    render_header()
    render_scenario_summary(sst, rainfall, drought, heat, adapt_dict)
    if timings["total"] > SLOW_RESPONSE_SECS:
        st.caption("⏱ Slow response — results cached for speed")
    render_status_alerts(adj, policy_rules, ood_warnings)   # above the cards
    render_metric_cards(results, baseline)

    left_col, right_col = st.columns([1.4, 1], gap="large")
    with left_col:
        render_risk_map(adj)
    with right_col:
        render_recommendations(adapt_dict, all_combos)

    render_radar_chart(results, baseline)
    render_zscore_panel(results, hist_stats, backend, policy_rules)
    render_transparency(results, ood_warnings, hist_stats)
    render_export(scenario_data, results, all_combos, ood_warnings)
    render_dev_panel(backend, timings)
    render_footer()


main()


# ═════════════════════════════════════════════════════════════════════════════
# MANUAL QA SCRIPT — Step 3.12 integration-testing artifact (12-step checklist)
# ═════════════════════════════════════════════════════════════════════════════
#
# Run `streamlit run frontend/app.py` from the project root, then walk through:
#
#  1. STARTUP / ENGINE BADGE — With src/utils.py, models/*.joblib and data/*.json
#     present, the sidebar badge reads "🟢 Live models". Open Dev/QA: source=live,
#     compare_is_live=True, compare="live (utils.evaluate_all_policies)".
#  2. MODERATE PRESET — Click "🌤 Moderate". Status alert is 🟢/🟡; map is mostly
#     green/amber; metric deltas vs. baseline appear; no OOD bar.
#  3. SEVERE PRESET — Click "🔥 Severe". Alert escalates (🟠/🔴); districts shift
#     toward orange/red; Water Reservoir Security drops sharply.
#  4. EXTREME ENSO PRESET — Click "⚡ Extreme". Alert = 🔴 Critical Resource Alert;
#     red districts dominate the map; OOD bar appears above the cards.
#  5. TOP-3 RERANK — As presets change, the 🥇🥈🥉 cards reorder; under Extreme the
#     "All four measures" combo ranks #1 (highest projected resilience).
#  6. CURRENTLY ACTIVE — Tick the four adaptation checkboxes to match the #1 combo;
#     that card gains the 3px teal border + mint background + "← Currently Active".
#  7. DIRECTIVE CARD — Under the Z-score table the "📌 Active Directive" card matches
#     the worst indicator's Z-band (🔴/🟠/🟡/🟢) and quotes policy-rules guidance.
#  8. DIRECTIVE REVERSAL — Confirm Groundwater Stress & Agricultural Risk use the
#     reversed scale (a HIGH value yields a Critical/Elevated directive, not Normal).
#  9. OOD TRIO — Under Extreme, verify the SAME exact warning copy appears in (a) the
#     red alert bar, (b) Transparency → Distribution Check, and (c) the downloaded
#     TXT/PDF "WARNINGS" section — identical wording in all three.
# 10. EXPORTS MATCH SCREEN — Download TXT/PDF/CSV; indicator values, top strategy,
#     and OOD status match what is shown on screen.
# 11. SPEED — First Extreme rerun warms caches; a repeat interaction completes well
#     under ~1.5s (no "⏱ Slow response" caption on the cached path). Dev/QA shows the
#     timing; "Clear caches" forces a cold recompute.
# 12. GRACEFUL DEGRADATION — Temporarily rename models/crop_model.joblib: badge drops
#     to Partial/Stub, a non-blocking notice shows, recommendations fall back to the
#     local 16-combo loop, and the app stays fully usable with NO traceback. Restore
#     the file and confirm Live returns. Repeat by renaming src/utils.py (utils
#     features degrade only) and data/policy_rules.json (guidance text falls back).
