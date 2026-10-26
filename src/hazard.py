"""Hazard levels (0 LOW, 1 MODERATE, 2 HIGH) per skill level.

Initial cut-offs are percentiles of the measured Buoy 2 wave height. They are written to
data/processed/thresholds.json so surf instructors/lifeguards can review and edit them;
the models use whatever is in that file.
"""
import json

import numpy as np

from config import HAZARD_PERCENTILES, LONG_PERIOD_PERCENTILE, PROC

PATH = PROC / "thresholds.json"


def make_thresholds(obs_hs, obs_tz):
    hs = np.asarray(obs_hs, float)
    tz = np.asarray(obs_tz, float)
    hs, tz = hs[np.isfinite(hs)], tz[np.isfinite(tz)]
    thr = {"status": "INITIAL: percentile-based, awaiting instructor review",
           "long_period_tz": float(np.percentile(tz, LONG_PERIOD_PERCENTILE)), "hs": {}}
    for skill, (mod, high) in HAZARD_PERCENTILES.items():
        thr["hs"][skill] = {"moderate": float(np.percentile(hs, mod)), "high": float(np.percentile(hs, high))}
    PATH.write_text(json.dumps(thr, indent=2))
    return thr


def load_thresholds():
    if not PATH.exists():
        raise FileNotFoundError("Thresholds missing. Run `python -m src.train_stage2` first.")
    return json.loads(PATH.read_text())


def hazard_level(hs, tz, skill, thr):
    """Level from wave height; long-period swell raises it one step once wave height is already moderate."""
    hs, tz = np.asarray(hs, float), np.asarray(tz, float)
    t = thr["hs"][skill]
    level = (hs >= t["moderate"]).astype(float) + (hs >= t["high"]).astype(float)
    bump = (tz >= thr["long_period_tz"]) & (hs >= t["moderate"])
    level = np.minimum(level + bump, 2)
    level[~np.isfinite(hs) | ~np.isfinite(tz)] = np.nan
    return level
