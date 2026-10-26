"""Open-Meteo access (free, no API key). Every raw response is cached under data/cache/."""
import hashlib
import json
import time

import pandas as pd
import requests

from config import CACHE, LAT, LON

MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

MARINE_VARS = ["wave_height", "wave_period", "wave_direction", "swell_wave_height",
               "swell_wave_period", "swell_wave_direction", "wind_wave_height"]
MARINE_RENAME = {"wave_height": "f_hs", "wave_period": "f_tm", "wave_direction": "f_wavedir",
                 "swell_wave_height": "f_swh", "swell_wave_period": "f_swp",
                 "swell_wave_direction": "f_swdir", "wind_wave_height": "f_wwh",
                 "sea_level_height_msl": "f_sea_level"}
WIND_VARS = ["wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"]
WIND_RENAME = {"wind_speed_10m": "f_wind", "wind_direction_10m": "f_winddir", "wind_gusts_10m": "f_gust"}


def _get(url, params, use_cache=True, retries=3):
    key = hashlib.md5((url + json.dumps(params, sort_keys=True)).encode()).hexdigest()
    path = CACHE / f"{key}.json"
    if use_cache and path.exists():
        return json.loads(path.read_text())
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            js = r.json()
            path.write_text(json.dumps(js))
            return js
        except Exception as exc:                       # network or API error
            last = exc
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"Open-Meteo request failed: {last}")


def _frame(js, rename):
    h = js["hourly"]
    df = pd.DataFrame(h).set_index("time")
    df.index = pd.to_datetime(df.index)               # requested in UTC, so naive UTC
    df.index.name = None
    df = df.rename(columns=rename)
    return df.dropna(axis=1, how="all")


def marine_history(start, end):
    """ERA5-Ocean waves for Matara (hourly, ~0.5 degree grid)."""
    base = dict(latitude=LAT, longitude=LON, start_date=start, end_date=end,
                models="era5_ocean", timezone="UTC")
    try:
        js = _get(MARINE_URL, {**base, "hourly": ",".join(MARINE_VARS)})
    except RuntimeError:                               # some variables may not exist for this model
        js = _get(MARINE_URL, {**base, "hourly": "wave_height,wave_period,wave_direction"})
    return _frame(js, MARINE_RENAME)


def wind_history(start, end):
    """ERA5 reanalysis wind at 10 m (m/s) from the Open-Meteo archive."""
    js = _get(ARCHIVE_URL, dict(latitude=LAT, longitude=LON, start_date=start, end_date=end,
                                hourly=",".join(WIND_VARS), wind_speed_unit="ms", timezone="UTC"))
    return _frame(js, WIND_RENAME)


def marine_forecast(days=3):
    js = _get(MARINE_URL, dict(latitude=LAT, longitude=LON, forecast_days=days, timezone="UTC",
                               hourly=",".join(MARINE_VARS + ["sea_level_height_msl"])), use_cache=False)
    return _frame(js, MARINE_RENAME)


def wind_forecast(days=3):
    js = _get(FORECAST_URL, dict(latitude=LAT, longitude=LON, forecast_days=days, timezone="UTC",
                                 hourly=",".join(WIND_VARS), wind_speed_unit="ms"), use_cache=False)
    return _frame(js, WIND_RENAME)


if __name__ == "__main__":
    print("Testing the Open-Meteo connection for Matara ...")
    m = marine_history("2013-05-20", "2013-05-27")
    w = wind_history("2013-05-20", "2013-05-27")
    print(m.head(), "\n", w.head(), "\n", len(m), "marine rows,", len(w), "wind rows")
    print("Earliest marine date check: request 1990-01-01 ->", end=" ")
    try:
        print(len(marine_history("1990-01-01", "1990-01-03")), "rows")
    except Exception as exc:
        print("failed:", exc)
