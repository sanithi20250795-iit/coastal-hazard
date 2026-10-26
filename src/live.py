"""Live path: pull the latest forecast, correct it (Stage 1), classify hazard (Stage 2), explain it.

Fallbacks (Section 5.3):
  live    - the pull worked.
  cached  - the pull failed; the last stored forecast is younger than 24 h (age is shown).
  stale   - the pull failed and the stored forecast is older than 24 h: uncorrected forecast + warning.
"""
import sqlite3

import joblib
import numpy as np
import pandas as pd

from config import FORECAST_DAYS, LEVEL_NAMES, LOCAL_TZ, LOG, MODELS, SHOW_HOURS, SKILLS, STALE_AFTER_H
from src import hazard, openmeteo
from src.dataset import add_features

FEATURE_TEXT = {
    "hs_corr": "wave height above the usual level", "tm_corr": "long wave period (powerful swell)",
    "wind_corr": "wind speed", "f_wavedir_sin": "swell direction", "f_wavedir_cos": "swell direction",
    "f_swell_frac": "a high share of swell in the waves", "tide_h": "tide height", "tide_rate": "tide rising or falling",
    "hour_sin": "time of day", "hour_cos": "time of day", "season": "the monsoon season",
}


# ----------------------------------------------------------------------------- forecast pull / storage
def _store(df, fetched):
    stamp = fetched.strftime("%Y%m%dT%H%M%SZ")
    df.to_parquet(LOG / f"forecast_{stamp}.parquet")
    try:                                                   # convenience log; the parquet files are the archive
        con = sqlite3.connect(LOG / "forecast_log.db")
        x = df.copy()
        x["fetched_at"] = fetched.isoformat()
        x.index.name = "valid_time"
        x.to_sql("forecast", con, if_exists="append")
        con.close()
    except Exception:
        pass


def _latest_stored():
    files = sorted(LOG.glob("forecast_*.parquet"))
    if not files:
        return None
    f = files[-1]
    fetched = pd.to_datetime(f.stem.split("_")[1], format="%Y%m%dT%H%M%SZ", utc=True)
    return pd.read_parquet(f), fetched


def get_forecast():
    now = pd.Timestamp.now(tz="UTC")
    try:
        df = openmeteo.marine_forecast(FORECAST_DAYS).join(openmeteo.wind_forecast(FORECAST_DAYS), how="left")
        _store(df, now)
        return df, now, "live", None
    except Exception as exc:
        stored = _latest_stored()
        if stored is None:
            raise RuntimeError(f"Forecast source unavailable and nothing is stored yet: {exc}")
        df, fetched = stored
        age_h = (now - fetched).total_seconds() / 3600
        return df, fetched, ("cached" if age_h <= STALE_AFTER_H else "stale"), str(exc)


# ----------------------------------------------------------------------------- prediction
def _top_drivers(model, Xf, feats, pred_idx, k=3):
    contrib = model.booster_.predict(Xf, pred_contrib=True)
    n, nf = len(Xf), len(feats)
    if len(model.classes_) > 2:
        c = contrib.reshape(n, len(model.classes_), nf + 1)[np.arange(n), pred_idx, :nf]
    else:
        c = contrib[:, :nf] * np.where(pred_idx == 1, 1, -1)[:, None]
    out = []
    for row in c:
        names = []
        for j in np.argsort(-row):
            if row[j] <= 0 or len(names) == k:
                break
            t = FEATURE_TEXT.get(feats[j], feats[j])
            if t not in names:
                names.append(t)
        out.append("; ".join(names))
    return out


def predict(df, mode):
    thr = hazard.load_thresholds()
    X = add_features(df.copy())
    if mode == "stale":
        for tgt, raw in (("hs", "f_hs"), ("tm", "f_tm"), ("wind", "f_wind")):
            X[f"{tgt}_corr"] = X[raw]
    else:
        stage1 = joblib.load(MODELS / "stage1.joblib")
        for tgt, b in stage1.items():
            resid = b["model"].predict(X.reindex(columns=b["features"]))
            X[f"{tgt}_corr"] = X[b["raw_col"]] + resid
    stage2 = joblib.load(MODELS / "stage2.joblib") if mode != "stale" else {}
    out = pd.DataFrame(index=X.index)
    for c in ("f_hs", "hs_corr", "f_tm", "tm_corr", "f_wind", "wind_corr", "tide_h", "f_wavedir"):
        if c in X:
            out[c] = X[c]
    for skill in SKILLS:
        rule = np.nan_to_num(hazard.hazard_level(X["hs_corr"], X["tm_corr"], skill, thr), nan=0).astype(int)
        out[f"rawrule_{skill}"] = np.nan_to_num(hazard.hazard_level(X["f_hs"], X["f_tm"], skill, thr), nan=0).astype(int)
        if mode == "stale":
            out[f"level_{skill}"] = out[f"rawrule_{skill}"]
            out[f"p_high_{skill}"] = np.nan
            out[f"drivers_{skill}"] = "uncorrected forecast (rule-based)"
            continue
        b = stage2[skill]
        Xf = X.reindex(columns=b["features"])
        proba = b["model"].predict_proba(Xf)
        full = np.zeros((len(Xf), 3))
        for i, c in enumerate(b["model"].classes_):
            full[:, int(c)] = proba[:, i]
        pred_idx = proba.argmax(1)
        out[f"level_{skill}"] = np.array(b["model"].classes_)[pred_idx].astype(int)
        out[f"p_high_{skill}"] = full[:, 2] if not b["merged"] else full[:, 1]
        out[f"drivers_{skill}"] = _top_drivers(b["model"], Xf, b["features"], pred_idx)
    now = pd.Timestamp.now(tz="UTC").floor("h").tz_localize(None)
    out = out[out.index >= now].head(SHOW_HOURS).copy()
    out["time_local"] = out.index.tz_localize("UTC").tz_convert(LOCAL_TZ)
    return out


def run():
    df, fetched, mode, error = get_forecast()
    out = predict(df, mode)
    age_h = (pd.Timestamp.now(tz="UTC") - fetched).total_seconds() / 3600
    meta = {"mode": mode, "fetched_at_utc": fetched.isoformat(), "age_hours": round(age_h, 1), "error": error,
            "validated_site": "Matara only", "note": "Typical local bias correction, not a guarantee. "
            "Hazard levels are a proxy for risk built from measured wave conditions."}
    return {"meta": meta, "df": out}


# ----------------------------------------------------------------------------- summaries
def safest_window(df, skill, hours=3):
    lvl = df[f"level_{skill}"]
    if len(lvl) < hours:
        return None
    roll = lvl.rolling(hours).max()
    score = roll * 10 + df[f"p_high_{skill}"].fillna(0).rolling(hours).mean()
    end = score.idxmin()
    pos = df.index.get_loc(end)
    return df["time_local"].iloc[pos - hours + 1], df["time_local"].iloc[pos]


def summary_text(df, skill, meta):
    lvl = df[f"level_{skill}"]
    top = int(lvl.max())
    hrs = df.loc[lvl == top, "time_local"]
    if top == 0:
        head = f"LOW risk for {skill}s throughout the next {len(df)} hours."
    else:
        head = (f"{LEVEL_NAMES[top]} risk for {skill}s between {hrs.iloc[0]:%H:%M} and "
                f"{(hrs.iloc[-1] + pd.Timedelta(hours=1)):%H:%M} ({hrs.iloc[0]:%a %d %b}).")
    w = safest_window(df, skill)
    tail = f" Safest 3-hour window: {w[0]:%a %H:%M} to {(w[1] + pd.Timedelta(hours=1)):%H:%M}." if w else ""
    warn = " Uncorrected forecast: data is old." if meta["mode"] == "stale" else ""
    return head + tail + warn + " Conditions at Matara; not a guarantee."


def to_json(res, skill):
    df = res["df"]
    rows = [{"time_local": r.time_local.isoformat(), "wave_height_raw_m": _f(r.f_hs), "wave_height_corrected_m": _f(r.hs_corr),
             "level": LEVEL_NAMES[int(getattr(r, f"level_{skill}"))], "drivers": getattr(r, f"drivers_{skill}")}
            for r in df.itertuples()]
    return {"skill": skill, "meta": res["meta"], "summary": summary_text(df, skill, res["meta"]), "hours": rows}


def _f(x):
    return None if pd.isna(x) else round(float(x), 2)
