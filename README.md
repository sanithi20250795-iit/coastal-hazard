# Coastal Hazard Forecasting for Southern Sri Lanka (ModelX 2.0, Team Binary Belles)

Two-stage model validated at Matara only:
1. **Stage 1** learns how far the free Open-Meteo / ERA5-Ocean forecast is from what the Matara buoy measured (wave height, wave period, wind).
2. **Stage 2** predicts a LOW / MODERATE / HIGH hazard level per skill level from the corrected values, tide and season, and names the main drivers.
3. A **dashboard** (Streamlit) and an **API** (FastAPI) show the next 48 hours.

## Folder layout (put files exactly here)

```
coastal-hazard/
  config.py
  requirements.txt
  data/raw/            <- put The_wind_wave_dataset_at_Matara.xlsx here
  src/                 matara.py tide.py openmeteo.py dataset.py cv.py hazard.py
                       train_stage1.py train_stage2.py live.py daily_pull.py check_timezone.py
  app/                 api.py dashboard.py
  tests/               smoke_test_synthetic.py
  .github/workflows/   daily.yml
```

## One-time setup (Windows PowerShell shown; on Mac/Linux use `source .venv/bin/activate`)

```
cd coastal-hazard
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Copy your Matara Excel file into `data/raw/` (any `.xlsx` name works; there must be only one).

## Run order (always from the `coastal-hazard` folder)

| Step | Command | What to check |
|---|---|---|
| 0 | `python tests/smoke_test_synthetic.py` | Ends with `SMOKE TEST PASSED`. Uses fake data in a temp folder, proves the install works. Ignore its numbers. |
| 1 | `python -m src.matara` | Prints real row counts and the Hs percentiles. Copy these into Sections 5.1 and 6 of the proposal. |
| 2 | `python -m src.openmeteo` | Proves API access. **Screenshot this for Section 5.1.** It also tests how far back the marine history goes. |
| 3 | `python -m src.check_timezone` | Reports the best clock shift. If it is not 0, edit `BUOY_UTC_OFFSET_H` in `config.py` as the message says. |
| 4 | `python -m src.tide` | Fits the tide model and prints the hold-out r and RMSE. |
| 5 | `python -m src.dataset` | Builds `data/processed/train.parquet` and prints the raw model-vs-buoy error (your Stage 1 baseline). |
| 6 | `python -m src.train_stage1` | Prints RMSE for raw, bias-only, persistence, climatology, ridge, LightGBM, per fold. |
| 7 | `python -m src.train_stage2` | Writes `data/processed/thresholds.json` and prints recall and precision per skill. |
| 8 | `streamlit run app/dashboard.py` | Opens the dashboard in your browser. |
| 9 | `uvicorn app.api:app --reload` | API at http://127.0.0.1:8000/forecast?skill=beginner (docs at /docs). |

## Hazard thresholds (important)

`data/processed/thresholds.json` is created with **initial percentile cut-offs**. Show it to at least two surf instructors or
lifeguards, edit the numbers they correct, then rerun `python -m src.train_stage2`. Until then the hazard levels are only a proxy.

## Daily forecast pull

`python -m src.daily_pull` stores today's forecast in `data/forecast_log/`. `.github/workflows/daily.yml` runs it every day on
GitHub (commit `models/` and `data/processed/thresholds.json` so the workflow and dashboard can load them).

## Known limits (keep these in the proposal)

- Trained on ERA5-Ocean (reanalysis) but served on forecasts, so real errors may be larger. The stored daily forecasts let you measure accuracy by lead time later.
- Buoys sit offshore (10 m and 20 m depth), not at a surf break. No rip-current measurements exist in the data.
- Validated at Matara only. Other coasts need their own validation.
- Time zone of the buoy clock is assumed (UTC+5:30) until `check_timezone` confirms it.
