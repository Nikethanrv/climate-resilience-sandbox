# AI Climate Resilience Sandbox

An interactive Streamlit scenario-planning tool for exploring how ENSO-driven climate shocks may affect crop yield stability, reservoir security, groundwater stress, agricultural risk, and regional resilience in Tamil Nadu.

The app is designed for hackathon-scale decision support: users can adjust climate stressors, toggle adaptation strategies, compare outcomes, view regional risk patterns, and export an action briefing. It is not an operational forecast system or official drought declaration tool.

## Live Demo

Try the deployed app here: [AI Climate Resilience Sandbox](https://nikethanrv-climate-resilience-sandbox-app-tzjhn4.streamlit.app/)

## What It Does

- Simulates climate stress scenarios using sea surface temperature anomaly, monsoon rainfall deficit, drought duration, and heat stress.
- Compares outcomes with and without adaptation measures.
- Scores resilience across food, water, groundwater, and agricultural-risk indicators.
- Routes scenarios into planning categories: Normal, Elevated, Severe, and Critical.
- Displays regional risk distribution for representative Tamil Nadu districts.
- Recommends policy actions from a curated knowledge base.
- Exports scenario briefings as TXT, CSV, and PDF when `fpdf2` is available.

## Live App Workflow

1. Set climate inputs in the sidebar.
2. Toggle adaptation measures:
   - Drought-resistant crop program
   - Emergency groundwater rationing
   - Supplemental irrigation support
   - Water conservation initiative
3. Review key resilience indicators.
4. Compare the radar chart:
   - With Current Adaptations
   - Without Adaptations
5. Inspect risk maps, z-score routing, recommendation cards, transparency notes, and exportable reports.

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- scikit-learn
- joblib
- fpdf2

## Repository Structure

```text
.
├── app.py                         # Streamlit frontend and backend adapter
├── requirements.txt               # Runtime dependencies
├── src/
│   └── utils.py                   # Risk, resilience, and policy helper functions
├── scripts/
│   ├── generate_dataset.py        # Dataset generation/preparation script
│   ├── generate_hist_stats.py     # Historical indicator statistic helper
│   ├── train_model.py             # Model training script
│   └── validate_data.py           # Dataset validation script
├── data/
│   ├── historical_climate_data.csv
│   ├── historical_statistics.json
│   ├── metadata.json
│   ├── policy_rules.json
│   ├── risk_thresholds.json
│   └── raw/
├── models/
│   ├── crop_model.joblib
│   └── water_model.joblib
├── docs/
└── reports/
```

## Data

The working dataset is `data/historical_climate_data.csv`.

It contains 27 yearly records from 1991 to 2017 with these columns:

- `year`
- `oni_djf`
- `rainfall_jjas`
- `rainfall_ond`
- `rice_yield`
- `groundnut_yield`
- `reservoir_storage_pct`

The project metadata references climate, agricultural, and water sources including IMD, NOAA CPC, India-WRIS/CWC, ICRISAT District Level Data, FAO, NDMA, and Ministry of Agriculture guidance.

## Model Overview

The app uses trained surrogate models stored in `models/`:

- `crop_model.joblib`: predicts crop yield stability from climate and agricultural features.
- `water_model.joblib`: predicts reservoir security from climate and water features.

The frontend normalizes model outputs to user-facing indicator scales. Adaptation measures are applied in two ways:

- By adjusting model input features such as rainfall proxies, reservoir storage, and groundnut-yield proxy.
- By applying a visible planning impact layer to the displayed indicators so the adaptation comparison remains interpretable in the app.

If the trained model dependencies or artifacts are unavailable, the app can fall back to a lightweight stub emulator so the interface remains usable for demos.

## Indicators

The main dashboard reports:

- Crop Yield Stability: higher is better.
- Water Reservoir Security: higher is better.
- Groundwater Stress: lower is better.
- Agricultural Risk Exposure: lower is better, displayed on a `/ 5.0` scale.
- Regional Resilience Score: higher is better.

The regional resilience score combines food security, water security, groundwater health, and low agricultural risk into one planning-oriented score.

## Risk Classification

The simulator uses project-defined planning categories:

- Normal
- Elevated
- Severe
- Critical

Z-score routing is based on historical indicator distributions from 1991-2017 and IMD SPI-inspired threshold bands:

- Normal: greater than -0.5 standard deviations
- Elevated: -1.0 to -0.5
- Severe: -1.5 to -1.0
- Critical: less than or equal to -1.5

For stress/risk indicators where lower values are better, the sign is reversed for routing.

## Policy Knowledge Base

Policy guidance lives in `data/policy_rules.json`.

Each category contains:

- title
- description
- recommendations
- source attribution
- recommendation category

The knowledge base is documented by `data/metadata.json`, which tracks source documents and reference materials.

## Local Setup

Clone the repository and create a fresh virtual environment.

```bash
python -m venv .venv
```

Activate it.

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

Run the app from the repository root.

```bash
streamlit run app.py
```

## Streamlit Deployment

For Streamlit Community Cloud:

1. Push this repository to GitHub.
2. Create a new Streamlit app.
3. Select the repository and branch.
4. Set the main file path to:

```text
app.py
```

5. Deploy.

After future pushes to the deployed branch, Streamlit usually rebuilds automatically. If a change does not appear, reboot or redeploy the app from the Streamlit Cloud app management page.

## Validation And Training

Validate the dataset:

```bash
python scripts/validate_data.py
```

Train the surrogate models:

```bash
python scripts/train_model.py
```

Generate historical statistics:

```bash
python scripts/generate_hist_stats.py
```

Note for Windows users: if console output fails because of emoji or Unicode symbols, run commands with UTF-8 output enabled, or replace the symbols in script output with plain ASCII.

## Outputs

The app can generate:

- TXT action briefing
- CSV scenario data
- PDF action briefing, when `fpdf2` is installed

Generated reports are intended for scenario communication and hackathon demo workflows.

## Limitations

- This is a scenario-planning sandbox, not a certified forecast.
- The dataset is small, covering 27 annual observations.
- The models are surrogate estimators for exploratory use.
- Adaptation impacts are simplified and should be interpreted as planning assumptions.
- Some historical reservoir values include imputation, as documented in the data notes.
- Recommendations are guidance-oriented and do not replace official government advisories.

## Suggested Demo Script

1. Start with default settings and show the normal dashboard.
2. Select a severe or extreme climate preset.
3. Show how risk indicators and policy directives change.
4. Toggle adaptation measures and compare the radar chart.
5. Open the transparency panel to explain assumptions.
6. Export an action briefing.

## Contributing

When updating the project:

- Keep data files and model assumptions documented.
- Update `data/metadata.json` when adding or changing sources.
- Re-run validation after dataset changes.
- Re-train models after feature or dataset changes.
- Keep Streamlit UI changes readable on both desktop and mobile.

## Disclaimer

This tool is generated for educational, hackathon, and scenario-planning purposes only. It does not constitute an operational forecast, official drought declaration, or government guidance.
