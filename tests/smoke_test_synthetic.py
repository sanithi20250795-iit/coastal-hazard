"""Offline smoke test. Checks that the WHOLE pipeline runs end to end on your machine.

It copies the project to a temporary folder and replaces Open-Meteo with SYNTHETIC data made from the
buoy measurements, so the numbers it prints are meaningless. Your real data/ and models/ are not touched.

Run:  python tests/smoke_test_synthetic.py
"""
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1]
TMP = Path(tempfile.mkdtemp()) / "proj"
shutil.copytree(SRC, TMP, ignore=shutil.ignore_patterns("__pycache__", "cache", "processed", "forecast_log", "*.joblib", "tide.json", ".git"))
sys.path.insert(0, str(TMP))

from src import live, matara, openmeteo, tide, dataset, train_stage1, train_stage2  # noqa: E402

rng = np.random.default_rng(0)
OBS = matara.load_buoy2()


def _marine_history(start, end):
    idx = pd.date_range(start, pd.Timestamp(end) + pd.Timedelta(hours=23), freq="h")
    o = OBS.reindex(idx).interpolate(limit_direction="both")
    hs = 1.25 * o["obs_hs"] + 0.35 + rng.normal(0, 0.12, len(idx))
    return pd.DataFrame({"f_hs": hs, "f_tm": o["obs_tz"] + 0.8 + rng.normal(0, 0.4, len(idx)),
                         "f_wavedir": 190 + rng.normal(0, 8, len(idx)), "f_swh": 0.85 * hs,
                         "f_swp": o["obs_tz"] + 1.5}, index=idx)


def _wind_history(start, end):
    idx = pd.date_range(start, pd.Timestamp(end) + pd.Timedelta(hours=23), freq="h")
    w = 3 + 1.5 * np.sin(np.arange(len(idx)) / 12) + rng.normal(0, 0.5, len(idx))
    return pd.DataFrame({"f_wind": w, "f_winddir": 200 + rng.normal(0, 25, len(idx)), "f_gust": w * 1.4}, index=idx)


def _future(days):
    idx = pd.date_range(pd.Timestamp.now(tz="UTC").floor("h").tz_localize(None), periods=24 * days, freq="h")
    hs = 1.8 + 0.5 * np.sin(np.arange(len(idx)) / 9)
    return idx, hs


def _marine_forecast(days=3):
    idx, hs = _future(days)
    return pd.DataFrame({"f_hs": hs, "f_tm": 10 + np.sin(np.arange(len(idx)) / 20), "f_wavedir": 190.0,
                         "f_swh": 0.85 * hs, "f_swp": 11.0, "f_sea_level": 0.4 * np.sin(np.arange(len(idx)) / 2)}, index=idx)


def _wind_forecast(days=3):
    idx, _ = _future(days)
    return pd.DataFrame({"f_wind": 3.5, "f_winddir": 200.0, "f_gust": 5.0}, index=idx)


openmeteo.marine_history, openmeteo.wind_history = _marine_history, _wind_history
openmeteo.marine_forecast, openmeteo.wind_forecast = _marine_forecast, _wind_forecast

print("1. loaders        ", len(OBS), "buoy rows")
print("2. tide model     ", tide.evaluate()); tide.fit_and_save()
print("3. training table ", dataset.build_training_table().shape)
print("4. stage 1"); train_stage1.main()
print("5. stage 2"); train_stage2.main()
res = live.run()
print("6. live forecast  ", res["meta"]["mode"], len(res["df"]), "rows")
print("   summary:", live.summary_text(res["df"], "beginner", res["meta"]))
res2 = live.predict(openmeteo.marine_forecast().join(openmeteo.wind_forecast()), "stale")
print("7. stale fallback ", res2.filter(like="level_").max().to_dict())
openmeteo.marine_forecast = lambda days=3: (_ for _ in ()).throw(RuntimeError("simulated outage"))
print("8. simulated outage ->", live.run()["meta"]["mode"])
print("\nSMOKE TEST PASSED (synthetic data: ignore the metrics above).")
shutil.rmtree(TMP.parent, ignore_errors=True)
