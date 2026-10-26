"""Harmonic tide model fitted to the Buoy 1 water-level record.

Run:  python -m src.tide      (fits, reports the hold-out score, saves models/tide.json)
"""
import json

import numpy as np
import pandas as pd

from config import MODELS
from src.matara import load_buoy1_level

# Hours. Only constituents that a ~5-month record can separate.
CONSTITUENTS = {"M2": 12.4206012, "S2": 12.0, "N2": 12.65834751, "K1": 23.93447213, "O1": 25.81933871}
EPOCH = pd.Timestamp("2013-01-01")
PATH = MODELS / "tide.json"


def _hours(index):
    return ((pd.DatetimeIndex(index) - EPOCH) / pd.Timedelta(hours=1)).to_numpy(float)


def _design(t):
    cols = [np.ones_like(t)]
    for period in CONSTITUENTS.values():
        w = 2 * np.pi / period
        cols += [np.cos(w * t), np.sin(w * t)]
    return np.column_stack(cols)


def _fit(level: pd.Series):
    coef, *_ = np.linalg.lstsq(_design(_hours(level.index)), level.to_numpy(float), rcond=None)
    return coef


def evaluate(train_days=120):
    lvl = load_buoy1_level()
    cut = lvl.index[0] + pd.Timedelta(days=train_days)
    train, test = lvl[lvl.index < cut], lvl[lvl.index >= cut]
    coef = _fit(train)
    pred = _design(_hours(test.index)) @ coef
    r = float(np.corrcoef(test.to_numpy(), pred)[0, 1])
    rmse = float(np.sqrt(np.mean((test.to_numpy() - pred) ** 2)))
    return {"train_hours": len(train), "test_hours": len(test), "r": round(r, 3), "rmse_m": round(rmse, 3)}


def fit_and_save():
    lvl = load_buoy1_level()
    coef = _fit(lvl)
    PATH.write_text(json.dumps({"coef": coef.tolist(), "n_hours": len(lvl),
                                "start": str(lvl.index.min()), "end": str(lvl.index.max())}))
    return coef


def predict(index) -> pd.DataFrame:
    """Tide height (m, relative to the mean level) and its rate of change (m/h)."""
    if not PATH.exists():
        raise FileNotFoundError("Run `python -m src.tide` first to fit the tide model.")
    coef = np.array(json.loads(PATH.read_text())["coef"])
    t = _hours(index)
    h = _design(t) @ coef
    rate = (_design(t + 1) @ coef - _design(t - 1) @ coef) / 2
    return pd.DataFrame({"tide_h": h - coef[0], "tide_rate": rate}, index=pd.DatetimeIndex(index))


if __name__ == "__main__":
    print("Hold-out check (fit on first 120 days, test on the rest):", evaluate())
    fit_and_save()
    print("Saved", PATH)
