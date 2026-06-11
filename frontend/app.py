"""
AI Climate Resilience Sandbox — Frontend Phase 2

CHANGELOG (Phase 2 fixes applied):
  FIX-1  RISK_NORMAL / RISK_ELEVATED / RISK_SEVERE defined once as module
         constants; all threshold comparisons reference them — not literals.
  FIX-2  Every section moved into a named render_*() function;
         main() is the sole top-level caller — zero loose UI code.
  FIX-3  All Plotly charts now carry config={"displayModeBar": False}
         (radar previously had no config; map now propagates the flag).
  FIX-4  Groundwater Stress confirmed: delta_color="inverse" + reversed
         status-pill scale (_status_gw_stress).
  FIX-5  get_backend() adapter with @st.cache_resource; sidebar engine
         badge reports Stub / Partial / Live mode.
  FIX-6  load_historical_stats() replaces all hardcoded MOCK_STATS usage
         in the Z-score table and transparency expander.
  FIX-7  _is_ood() uses hist_stats min/max for input bounds when the JSON
         file provides them; hardcoded OOD_DEFAULTS remain as fallback.
"""

# ── Stdlib imports ────────────────────────────────────────────────────────────
import inspect
import json
import logging
import os
import sys

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
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)          # project root

# FIX-1: thresholds defined exactly once as module constants.
RISK_NORMAL   = 0.35
RISK_ELEVATED = 0.50
RISK_SEVERE   = 0.70

# OOD boundary fallbacks (historical training range 1982–2022).
# heat_stress >4 means ==5 (top of the slider), matching the original check.
OOD_DEFAULTS: dict = {
    "sst_anomaly":      3.0,
    "rainfall_deficit": 80.0,
    "drought_months":   10.0,
    "heat_stress":       4.0,
}

# Fallback stats — used when historical_statistics.json is absent or partial.
MOCK_STATS: dict = {
    "crop_yield_stability":     {"mean": 72.0, "std": 12.0, "min": 30.0, "max": 100.0},
    "water_reservoir_security": {"mean": 68.0, "std": 10.0, "min": 20.0, "max": 100.0},
    "groundwater_stress":       {"mean": 35.0, "std": 15.0, "min":  5.0, "max":  95.0},
    "agricultural_risk":        {"mean":  2.1, "std":  0.8, "min":  0.0, "max":   5.0},
    "regional_resilience":      {"mean": 65.0, "std": 11.0, "min": 20.0, "max": 100.0},
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

_HIST_STATS_PATH   = os.path.join(_ROOT, "historical_statistics.json")
_REQUIRED_STAT_KEYS = {"mean", "std", "min", "max"}

STRATEGY_NAMES: dict = {
    "crops": "Drought-Resistant Crops",
    "gw":    "Groundwater Rationing",
    "irr":   "Supplemental Irrigation",
    "water": "Water Conservation",
}

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
    """
    Stub emulator — Teammate 1 replaces this via the backend adapter.
    Returns dict with 6 float indicators.
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
    """Stub — Teammate 3 replaces with utils.calculate_zscore."""
    return round((value - mean) / std if std > 0 else 0.0, 2)


@st.cache_data(ttl=3600)
def load_policy_rules() -> dict:
    """Stub — Teammate 3 replaces with utils.load_policy_rules."""
    return {
        "Normal":         "Maintain standard monitoring. No immediate action required.",
        "Elevated Risk":  "Activate early-warning protocols. Review irrigation schedules.",
        "Severe Stress":  "Implement water rationing. Distribute drought-resistant seeds.",
        "Critical Alert": "Emergency response required. Activate all adaptation measures.",
    }


def load_risk_thresholds() -> dict:
    """Stub — Teammate 3 replaces with utils.load_thresholds."""
    return {
        "normal":   RISK_NORMAL,
        "elevated": RISK_ELEVATED,
        "severe":   RISK_SEVERE,
    }


# ═════════════════════════════════════════════════════════════════════════════
# BACKEND ADAPTER  (FIX-5)
# ═════════════════════════════════════════════════════════════════════════════

def make_model_emulator(crop_model, water_model):
    """
    Factory that returns a drop-in replacement for run_climate_emulator.
    Feature order MUST match Teammate 1's train_model.py:
        [sst, rainfall, drought, heat, crops, gw, irr, water]
    Remaining indicators are derived with the same stub formulas for
    consistency until Teammate 1 provides dedicated target columns.
    """
    def _emulator(
        sst: float,
        rainfall: float,
        drought: float,
        heat: float,
        adapt_dict: dict,
    ) -> dict:
        features = [[
            sst, rainfall, drought, heat,
            adapt_dict.get("crops", 0),
            adapt_dict.get("gw",    0),
            adapt_dict.get("irr",   0),
            adapt_dict.get("water", 0),
        ]]
        crop_yield = float(crop_model.predict(features)[0])
        water_sec  = float(water_model.predict(features)[0])
        # Derived indicators (same formulas as stub for backward compatibility).
        base  = sst / 3.5 * 0.4 + rainfall / 100 * 0.4 + drought / 12 * 0.2
        bonus = (
            adapt_dict.get("crops", 0) * 0.08
            + adapt_dict.get("gw",    0) * 0.06
            + adapt_dict.get("irr",   0) * 0.07
            + adapt_dict.get("water", 0) * 0.05
        )
        adj = max(0.0, min(1.0, base - bonus))
        return {
            "crop_yield_stability":     round(max(0.0, min(100.0, crop_yield)), 1),
            "water_reservoir_security": round(max(0.0, min(100.0, water_sec)),  1),
            "groundwater_stress":       round(adj * 100, 1),
            "agricultural_risk":        round(adj * 5,   2),
            "regional_resilience":      round((1 - adj) * 100, 1),
            "adjusted_risk_score":      adj,
        }

    return _emulator


def _utils_fn_ok(fn, min_params: int) -> bool:
    """Return True if fn is callable with at least min_params positional args."""
    try:
        return len(inspect.signature(fn).parameters) >= min_params
    except (TypeError, ValueError):
        return False


@st.cache_resource
def get_backend() -> dict:
    """
    Returns a dict of callables. Prefers real teammate modules, falls back
    to local stubs. Logged, never raises.

    Keys:
      source        — "stub" | "partial" | "live"
      emulator      — callable(sst, rainfall, drought, heat, adapt_dict) -> dict
      zscore        — callable(value, mean, std) -> float
      policy_rules  — callable() -> dict
      thresholds    — callable() -> dict
    """
    backend: dict = {
        "source":       "stub",
        "emulator":     run_climate_emulator,
        "zscore":       calculate_zscore,
        "policy_rules": load_policy_rules,
        "thresholds":   load_risk_thresholds,
    }

    # ── Teammate 3: utils.py ─────────────────────────────────────────────────
    try:
        if _ROOT not in sys.path:
            sys.path.insert(0, _ROOT)
        import utils  # noqa: PLC0415

        z_ok  = _utils_fn_ok(getattr(utils, "calculate_zscore",   None), 3)
        pr_ok = _utils_fn_ok(getattr(utils, "load_policy_rules",  None), 0)
        th_ok = _utils_fn_ok(getattr(utils, "load_thresholds",    None), 0)

        if z_ok and pr_ok and th_ok:
            backend.update(
                zscore=utils.calculate_zscore,
                policy_rules=utils.load_policy_rules,
                thresholds=utils.load_thresholds,
                source="partial",
            )
            logging.info("utils.py loaded — teammate zscore/policy/thresholds active")
        else:
            logging.info("utils.py found but functions not yet fully implemented")
    except ImportError:
        logging.info("utils.py not present — using stubs")

    # ── Teammate 1: joblib models ────────────────────────────────────────────
    try:
        import joblib  # lazy import — app works without scikit-learn

        crop_path  = os.path.join(_ROOT, "models", "crop_model.joblib")
        water_path = os.path.join(_ROOT, "models", "water_model.joblib")

        if (
            os.path.exists(crop_path)
            and os.path.getsize(crop_path) > 0
            and os.path.exists(water_path)
            and os.path.getsize(water_path) > 0
        ):
            crop_model  = joblib.load(crop_path)
            water_model = joblib.load(water_path)
            backend["emulator"] = make_model_emulator(crop_model, water_model)
            backend["source"]   = "live"
            logging.info("Live models loaded from %s", crop_path)
        else:
            logging.info("Model files absent or empty — using stub emulator")
    except Exception as exc:  # noqa: BLE001
        logging.warning("Model load failed — using stub emulator: %s", exc)

    return backend


# ═════════════════════════════════════════════════════════════════════════════
# HISTORICAL STATISTICS LOADER  (FIX-6 / FIX-7)
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_historical_stats() -> dict:
    """
    Loads historical_statistics.json (Teammate 1 deliverable).
    Falls back to MOCK_STATS per-indicator if the file is absent, unreadable,
    or missing required keys.  Adds a ``_using_mock`` bool to the result so
    callers can show a notice to the team.
    Also absorbs input-parameter bounds (sst_anomaly, rainfall_deficit,
    drought_months, heat_stress) when Teammate 1 includes them — those drive
    the OOD check in _is_ood().
    """
    result: dict = {k: dict(v) for k, v in MOCK_STATS.items()}
    result["_using_mock"] = True

    try:
        with open(_HIST_STATS_PATH, encoding="utf-8") as fh:
            raw: dict = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return result  # all mock

    any_mock = False
    for indicator in MOCK_STATS:
        entry = raw.get(indicator, {})
        if isinstance(entry, dict) and _REQUIRED_STAT_KEYS.issubset(entry):
            result[indicator] = {k: entry[k] for k in _REQUIRED_STAT_KEYS}
        else:
            any_mock = True   # keep the MOCK_STATS value already in result

    # Absorb input-feature bounds when Teammate 1 provides them (used for OOD).
    for input_key in ("sst_anomaly", "rainfall_deficit", "drought_months", "heat_stress"):
        if input_key in raw and isinstance(raw[input_key], dict):
            result[input_key] = raw[input_key]

    result["_using_mock"] = any_mock
    return result


def _is_ood(
    sst: float,
    rainfall: float,
    drought: float,
    heat: float,
    hist_stats: dict,
) -> bool:
    """True if any input exceeds the historical training max (FIX-7)."""
    def _max(key: str, fallback: float) -> float:
        return hist_stats.get(key, {}).get("max", fallback)

    return (
        sst      > _max("sst_anomaly",      OOD_DEFAULTS["sst_anomaly"])
        or rainfall > _max("rainfall_deficit", OOD_DEFAULTS["rainfall_deficit"])
        or drought  > _max("drought_months",   OOD_DEFAULTS["drought_months"])
        or heat     > _max("heat_stress",      OOD_DEFAULTS["heat_stress"])
    )


# ═════════════════════════════════════════════════════════════════════════════
# REPORT GENERATORS
# ═════════════════════════════════════════════════════════════════════════════

def generate_txt_report(
    scenario: dict,
    results: dict,
    top_strategy: dict,
    ood: bool,
) -> str:
    """Generates plain-text action briefing for download."""
    from datetime import datetime

    active = [k for k, v in scenario["adapt_dict"].items() if v] or ["None selected"]
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

TRANSPARENCY NOTE
─────────────────
{"⚠️  OUT-OF-DISTRIBUTION: Scenario exceeds training range (1982–2022)." if ood else "✅  Scenario within historical training range (1982–2022)."}
Model: Random Forest Regressor | Training period: 1982–2022

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
    ood: bool,
) -> bytes:
    """Generates PDF action briefing for download. Returns bytes."""
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

    if ood:
        pdf.set_fill_color(254, 226, 226)
        pdf.set_draw_color(220, 38, 38)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(153, 27, 27)
        pdf.multi_cell(
            0, 7,
            "OUT-OF-DISTRIBUTION WARNING: Parameters exceed training range. "
            "Interpret results as exploratory estimates only.",
            border=1, fill=True,
        )
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
# RANKING HELPER
# ═════════════════════════════════════════════════════════════════════════════

def _rank_all_combos(
    emulator,
    sst: float,
    rainfall: float,
    drought: float,
    heat: float,
) -> list:
    """Evaluate all 16 adaptation combos; return list sorted by resilience desc."""
    combos = []
    for c in range(16):
        combo = {
            "crops": bool(c & 1),
            "gw":    bool(c & 2),
            "irr":   bool(c & 4),
            "water": bool(c & 8),
        }
        r = emulator(sst, rainfall, drought, heat, {k: int(v) for k, v in combo.items()})
        combos.append({
            "combo": combo,
            "score": r["regional_resilience"],
            "adj":   r["adjusted_risk_score"],
        })
    return sorted(combos, key=lambda x: x["score"], reverse=True)


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
    """
    Renders the sidebar controls and engine-source badge.
    Returns (sst, rainfall, drought, heat, adapt_dict).
    """
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
        st.checkbox(
            "Drought-Resistant Crop Program",
            key="adapt_crops",
            help="Short-cycle, drought-tolerant seed deployment.",
        )
        st.checkbox(
            "Emergency Groundwater Rationing",
            key="adapt_groundwater",
            help="Volumetric water-use limits across irrigation zones.",
        )
        st.checkbox(
            "Supplemental Irrigation Support",
            key="adapt_irrigation",
            help="Activate supplemental micro-irrigation infrastructure.",
        )
        st.checkbox(
            "Water Conservation Initiative",
            key="adapt_water",
            help="Community rainwater harvesting and conservation.",
        )

        st.subheader("🚀 Quick Presets")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.button(
                "🌤 Moderate", use_container_width=True,
                on_click=_apply_preset, args=(MODERATE_EL_NINO, "moderate_el_nino"),
            )
        with col2:
            st.button(
                "🔥 Severe", use_container_width=True,
                on_click=_apply_preset, args=(SEVERE_DROUGHT, "severe_drought"),
            )
        with col3:
            st.button(
                "⚡ Extreme", use_container_width=True,
                on_click=_apply_preset, args=(EXTREME_ENSO, "extreme_enso"),
            )

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
    """Renders the teal header banner and amber disclaimer bar."""
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
    sst: float,
    rainfall: float,
    drought: float,
    heat: float,
    adapt_dict: dict,
) -> None:
    """Renders the scenario summary bar below the header."""
    climate_desc = (
        f"SST +{sst}°C  |  "
        f"{rainfall}% Rainfall Deficit  |  "
        f"{drought} Month Drought  |  "
        f"Heat Level {heat}/5"
    )
    active_adaptations = [
        name
        for name, key in {
            "Drought-Resistant Crops": "crops",
            "Groundwater Rationing":   "gw",
            "Supplemental Irrigation": "irr",
            "Water Conservation":      "water",
        }.items()
        if adapt_dict.get(key)
    ]
    adapt_desc = (
        "Adaptations Active: " + ", ".join(active_adaptations)
        if active_adaptations
        else "⚠️ No adaptation measures active"
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


def render_status_alerts(adj: float, policy_rules: dict, ood: bool) -> None:
    """
    Renders OOD warning (if triggered) then the risk-level status alert.
    Uses RISK_NORMAL / RISK_ELEVATED / RISK_SEVERE module constants (FIX-1).
    Always rendered ABOVE the metric cards.
    """
    if ood:
        st.error(
            "⚠️ Out-of-Distribution Warning: One or more parameters exceed "
            "the model's historical training range (1982–2022). Results are "
            "exploratory estimates only — interpret with caution."
        )
    if adj >= RISK_SEVERE:
        st.error("🔴 Critical Resource Alert — " + policy_rules.get("Critical Alert", ""))
    elif adj >= RISK_ELEVATED:
        st.warning("🟠 Severe Stress — " + policy_rules.get("Severe Stress", ""))
    elif adj >= RISK_NORMAL:
        st.warning("🟡 Elevated Risk — " + policy_rules.get("Elevated Risk", ""))
    else:
        st.success("🟢 Normal Conditions — " + policy_rules.get("Normal", ""))


def render_metric_cards(results: dict, baseline: dict) -> None:
    """Renders the five st.metric() cards inside a bordered container."""
    st.subheader("📊 Key Resilience Indicators")

    # Spec-required metrics (FIX-4: Groundwater Stress uses delta_color="inverse").
    metric_defs = [
        {
            "label":      "🌾 Crop Yield Stability",
            "value":      results["crop_yield_stability"],
            "base":       baseline["crop_yield_stability"],
            "unit":       "%",
            "delta_color": "normal",
            "status_fn":  _status_higher_better,
        },
        {
            "label":      "💧 Water Reservoir Security",
            "value":      results["water_reservoir_security"],
            "base":       baseline["water_reservoir_security"],
            "unit":       "%",
            "delta_color": "normal",
            "status_fn":  _status_higher_better,
        },
        {
            "label":      "🏜 Groundwater Stress",
            "value":      results["groundwater_stress"],
            "base":       baseline["groundwater_stress"],
            "unit":       "%",
            "delta_color": "inverse",   # FIX-4
            "status_fn":  _status_gw_stress,
        },
        {
            "label":      "⚠️ Agricultural Risk Exposure",
            "value":      results["agricultural_risk"],
            "base":       baseline["agricultural_risk"],
            "unit":       "/ 5.0",
            "delta_color": "inverse",
            "status_fn":  _status_ag_risk,
        },
        {
            "label":      "🛡 Regional Resilience Score",
            "value":      results["regional_resilience"],
            "base":       baseline["regional_resilience"],
            "unit":       "%",
            "delta_color": "normal",
            "status_fn":  _status_higher_better,
        },
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
    """Renders the interactive district risk map (or treemap fallback)."""
    st.subheader("📍 Regional Risk Distribution")

    DISTRICTS = [
        {"name": "Northern Plains",    "lat": 22.5, "lon": 78.0},
        {"name": "Central Valley",     "lat": 21.0, "lon": 77.5},
        {"name": "Eastern Hills",      "lat": 21.5, "lon": 80.0},
        {"name": "Southern Delta",     "lat": 19.0, "lon": 77.0},
        {"name": "Western Plateau",    "lat": 20.5, "lon": 75.5},
        {"name": "Coastal Zone",       "lat": 18.0, "lon": 73.5},
        {"name": "Mountain Foothills", "lat": 23.0, "lon": 79.5},
        {"name": "Inland Basin",       "lat": 20.0, "lon": 76.0},
    ]
    OFFSETS  = [0.10, -0.12, 0.08, -0.15, 0.05, -0.09, 0.13, -0.07]
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
                "Risk Level": True,
                "Resilience Score (%)": True,
                "lat": False, "lon": False, "size": False,
            },
            zoom=5,
            center={"lat": 21, "lon": 77.5},
            height=420,
        )
        fig_map.update_layout(
            map_style="carto-positron",
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            legend_title_text="Risk Level",
        )
        # FIX-3: displayModeBar: False on all charts
        st.plotly_chart(
            fig_map,
            use_container_width=True,
            config={"scrollZoom": True, "displayModeBar": False, "displaylogo": False},
            key="risk_map",
        )
    except Exception:  # noqa: BLE001
        fig_tree = px.treemap(
            map_df,
            path=["District"],
            values=[1] * len(map_df),
            color="Resilience Score (%)",
            color_continuous_scale=["#dc2626", "#ca8a04", "#16a34a"],
            hover_data={"Risk Level": True, "Resilience Score (%)": True},
            height=420,
        )
        fig_tree.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0})
        st.plotly_chart(
            fig_tree,
            use_container_width=True,
            config={"displayModeBar": False, "displaylogo": False},  # FIX-3
            key="risk_map",
        )

    st.markdown(
        "🟢 Normal&nbsp;&nbsp; 🟡 Elevated Risk&nbsp;&nbsp; "
        "🟠 Severe Stress&nbsp;&nbsp; 🔴 Critical Alert"
    )


def render_recommendations(adapt_dict: dict, all_combos: list) -> None:
    """Renders the top-3 ranked strategy recommendation cards."""
    st.subheader("🏆 Recommended Strategies")

    STRATEGY_DESCRIPTIONS = {
        frozenset(["crops", "water"]):
            "Reduces crop failure while preserving groundwater. "
            "Effective for moderate droughts.",
        frozenset(["crops", "gw", "irr", "water"]):
            "Maximum resilience. Requires significant coordination "
            "and resources across all sectors.",
        frozenset(["gw", "irr"]):
            "Balances short-term water access with long-term aquifer health.",
        frozenset(["crops", "gw"]):
            "Cost-effective pairing for early-stage drought response.",
        frozenset(["irr", "water"]):
            "Strong for water security; moderate crop protection.",
    }
    DEFAULT_DESC = (
        "Moderate improvement. Consider pairing with additional measures "
        "for stronger outcomes."
    )
    RANK_BADGES = ["🥇", "🥈", "🥉"]
    RANK_LABELS = ["Best Strategy", "Runner-Up", "Third Option"]

    active_keys = frozenset(k for k, v in adapt_dict.items() if v)
    top3 = all_combos[:3]

    for i, item in enumerate(top3):
        active_in_combo = [STRATEGY_NAMES[k] for k, v in item["combo"].items() if v]
        combo_keys = frozenset(k for k, v in item["combo"].items() if v)
        desc = STRATEGY_DESCRIPTIONS.get(combo_keys, DEFAULT_DESC)
        is_active = combo_keys == active_keys

        border = "3px solid #0d9488" if is_active else "1px solid #ccfbf1"
        bg     = "#f0fdfa"          if is_active else "#ffffff"

        chips = "  ".join(
            f'<span style="background:#dcfce7;color:#166534;'
            f'padding:2px 8px;border-radius:10px;font-size:0.75rem;">'
            f'✅ {s}</span>'
            for s in (active_in_combo or ["No measures"])
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
              <div style="color:#1a6b6b;font-weight:600;margin-top:6px;">
                Resilience Score: {item["score"]}%
              </div>
              {active_note}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_radar_chart(results: dict, baseline: dict) -> None:
    """Renders the resilience radar chart (with vs. without adaptations)."""
    st.subheader("📈 Resilience Profile — With vs. Without Adaptations")

    categories = [
        "Crop Yield Stability",
        "Water Security",
        "Groundwater Health",
        "Low Agricultural Risk",
        "Regional Resilience",
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
        fill="toself",
        name="Without Adaptations",
        line={"color": "#f97316", "dash": "dash"},
        fillcolor="rgba(249,115,22,0.1)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=with_vals + [with_vals[0]],
        theta=categories + [categories[0]],
        fill="toself",
        name="With Current Adaptations",
        line={"color": "#0d9488"},
        fillcolor="rgba(13,148,136,0.15)",
    ))
    fig.update_layout(
        polar={
            "radialaxis": {"visible": True, "range": [0, 100], "ticksuffix": "%"},
            "angularaxis": {"rotation": 90},
        },
        showlegend=True,
        paper_bgcolor="rgba(0,0,0,0)",
        height=380,
        margin={"l": 40, "r": 40, "t": 40, "b": 40},
    )
    # FIX-3: displayModeBar: False
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_zscore_panel(results: dict, hist_stats: dict, backend: dict) -> None:
    """
    Renders Z-score guidance table.
    Uses hist_stats (Teammate 1 JSON or MOCK_STATS fallback) for mean/std.
    Uses backend["zscore"] callable (utils or stub).
    """
    st.subheader("📐 Z-Score Risk Thresholds")

    ZSCORE_META = {
        "crop_yield_stability":     ("🌾 Crop Yield Stability",     "%",     False),
        "water_reservoir_security": ("💧 Water Reservoir Security",  "%",     False),
        "groundwater_stress":       ("🏜 Groundwater Stress",        "%",     True),
        "agricultural_risk":        ("⚠️ Agricultural Risk",         "/ 5.0", True),
        "regional_resilience":      ("🛡 Regional Resilience",       "%",     False),
    }

    def _directive_for_z(z: float, reversed_dir: bool) -> str:
        effective = -z if reversed_dir else z
        if effective < -2.0:
            return "🔴 Critical Drought Guidance — Emergency Response"
        if effective < -1.0:
            return "🟠 Elevated Stress Protocol"
        if effective < 0:
            return "🟡 Enhanced Monitoring Required"
        return "🟢 Normal Operating Range"

    zscore_fn = backend["zscore"]
    rows = []
    for key, (display_name, unit, reversed_dir) in ZSCORE_META.items():
        value = results[key]
        stats = hist_stats.get(key, MOCK_STATS[key])
        z = zscore_fn(value, stats["mean"], stats["std"])
        rows.append({
            "Indicator":     display_name,
            "Current Value": f"{value}{unit}",
            "Hist. Mean":    f"{stats['mean']}{unit}",
            "Z-Score":       z,
            "Directive":     _directive_for_z(z, reversed_dir),
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(
        "Z-score indicates standard deviations from the 40-year historical mean. "
        "Values below −2.0 trigger emergency-level guidance from the policy "
        "knowledge base."
    )


def render_transparency(results: dict, ood: bool, hist_stats: dict) -> None:
    """Renders the transparency & model confidence expander."""
    with st.expander("🔍 Transparency & Model Confidence", expanded=False):
        t1, t2, t3 = st.columns(3)

        with t1:
            st.markdown("**📊 Confidence Ranges**")
            INDICATOR_RANGES = {
                "Crop Yield Stability":     ("crop_yield_stability",     12.0),
                "Water Reservoir Security": ("water_reservoir_security", 10.0),
                "Groundwater Stress":       ("groundwater_stress",       15.0),
                "Agricultural Risk":        ("agricultural_risk",         0.6),
                "Regional Resilience":      ("regional_resilience",      11.0),
            }
            conf_rows = []
            for name, (key, default_range) in INDICATOR_RANGES.items():
                stats = hist_stats.get(key, MOCK_STATS[key])
                conf_rows.append({
                    "Indicator":  name,
                    "Estimate":   results[key],
                    "± Range":    default_range,
                    "Hist. Mean": stats["mean"],
                })
            st.dataframe(pd.DataFrame(conf_rows), use_container_width=True, hide_index=True)

            # FIX-6: notice when mock stats are in use
            if hist_stats.get("_using_mock", True):
                st.caption(
                    "Historical statistics: mock values "
                    "(awaiting Teammate 1 dataset)."
                )

        with t2:
            st.markdown("**🤖 Model Information**")
            st.markdown(
                """
                - **Model:** Random Forest Regressor (Surrogate Emulator)
                - **Training period:** 1982–2022 (40 years)
                - **Features:** SST anomaly, Rainfall deficit,
                  Drought duration, Heat stress, Policy flags
                - **Targets:** Crop Yield, Water Security
                - **Last updated:** June 2025
                - **Limitations:** Cannot account for sudden
                  geopolitical or infrastructure shocks.
                """
            )

        with t3:
            st.markdown("**⚠️ Distribution Check**")
            if ood:
                st.error(
                    "**Out-of-Distribution Scenario**  \n"
                    "Parameters exceed historical training range.  \n"
                    "Results are exploratory estimates only."
                )
            else:
                st.success(
                    "✅ All parameters within historical training range (1982–2022)."
                )
            st.markdown(
                """
                **Data Coverage:**
                Region: South/Central Asia pilot zone
                Source: Historical climate + agricultural records
                Boundary: 1982–2022 daily observations
                """
            )


def render_export(
    scenario_data: dict,
    results: dict,
    all_combos: list,
    ood: bool,
) -> None:
    """Renders the export action-briefing section."""
    st.divider()
    st.subheader("📄 Export Action Briefing")

    top = all_combos[0]
    top_strategy = {
        "strategies": [STRATEGY_NAMES[k] for k, v in top["combo"].items() if v],
        "score":      top["score"],
    }

    txt_content = generate_txt_report(scenario_data, results, top_strategy, ood)

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
    }])

    ec1, ec2, ec3 = st.columns(3)

    with ec1:
        st.download_button(
            label="⬇ Download TXT Report",
            data=txt_content,
            file_name="climate_resilience_briefing.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with ec2:
        try:
            pdf_bytes = generate_pdf_report(scenario_data, results, top_strategy, ood)
            st.download_button(
                label="⬇ Download PDF Report",
                data=pdf_bytes,
                file_name="climate_resilience_briefing.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except ImportError:
            st.info("PDF export requires: pip install fpdf2")

    with ec3:
        st.download_button(
            label="⬇ Download CSV Data",
            data=csv_df.to_csv(index=False),
            file_name="climate_scenario_data.csv",
            mime="text/csv",
            use_container_width=True,
        )


def render_footer() -> None:
    """Renders the bottom footer."""
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
    """
    Single top-level orchestrator.  All st.* calls happen inside the
    render_*() functions called from here — no loose UI code at module level.
    """
    # Must be the first Streamlit command.
    st.set_page_config(
        page_title="AI Climate Resilience Sandbox",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="auto",
    )
    _init_session_state()
    _inject_css()

    backend    = get_backend()
    hist_stats = load_historical_stats()

    # Sidebar renders widgets; returns current slider/checkbox values.
    sst, rainfall, drought, heat, adapt_dict = render_sidebar(backend)

    # Core computations — done once and shared across all render functions.
    emulator = backend["emulator"]
    results  = emulator(sst, rainfall, drought, heat, adapt_dict)
    baseline = emulator(sst, rainfall, drought, heat, {"crops": 0, "gw": 0, "irr": 0, "water": 0})
    adj      = results["adjusted_risk_score"]

    policy_rules = backend["policy_rules"]()
    ood          = _is_ood(sst, rainfall, drought, heat, hist_stats)
    all_combos   = _rank_all_combos(emulator, sst, rainfall, drought, heat)

    scenario_data = {
        "sst":        sst,
        "rainfall":   rainfall,
        "drought":    drought,
        "heat":       heat,
        "adapt_dict": adapt_dict,
    }

    # ── Main area ─────────────────────────────────────────────────────────────
    render_header()
    render_scenario_summary(sst, rainfall, drought, heat, adapt_dict)
    render_status_alerts(adj, policy_rules, ood)      # above cards — spec §A-3
    render_metric_cards(results, baseline)

    left_col, right_col = st.columns([1.4, 1], gap="large")
    with left_col:
        render_risk_map(adj)
    with right_col:
        render_recommendations(adapt_dict, all_combos)

    render_radar_chart(results, baseline)
    render_zscore_panel(results, hist_stats, backend)
    render_transparency(results, ood, hist_stats)
    render_export(scenario_data, results, all_combos, ood)
    render_footer()


main()
