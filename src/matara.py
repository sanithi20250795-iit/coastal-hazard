"""Load and clean the Matara wind and wave dataset (Luo et al., 2017/2018).

All timestamps are converted to UTC (naive) so they can be joined with Open-Meteo data.
"""
import numpy as np
import pandas as pd

from config import RAW, BUOY_UTC_OFFSET_H


def _xlsx():
    files = sorted(p for p in RAW.glob("*.xlsx") if not p.name.startswith("~$"))
    if not files:
        raise FileNotFoundError(f"Put the Matara .xlsx file in {RAW}")
    return files[0]


def _to_utc(index):
    return index - pd.Timedelta(hours=BUOY_UTC_OFFSET_H)


def _clean_dates(df, col="Date"):
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors="coerce")      # the unit row becomes NaT
    df = df.dropna(subset=[col]).set_index(col).sort_index()
    return df[~df.index.duplicated(keep="first")]


def _circular_mean(deg):
    rad = np.deg2rad(deg.dropna())
    if rad.empty:
        return np.nan
    return (np.rad2deg(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) + 360) % 360


def load_buoy2() -> pd.DataFrame:
    """Hourly waves from Buoy 2 (10 m depth), Apr 2013 - Apr 2014."""
    df = pd.read_excel(_xlsx(), sheet_name="Buoy2_Wave")
    df = df.drop(columns=[df.columns[1]])                    # unnamed MATLAB date-number column
    df = _clean_dates(df)
    cols = ["Tp", "Tz", "Ts", "pdir", "avedir", "Hs"]
    df = df[cols].apply(pd.to_numeric, errors="coerce")
    df = df[df["Hs"] > 0]
    df.index = _to_utc(df.index)
    df = df.resample("1h").mean()
    return df.rename(columns={"Hs": "obs_hs", "Tz": "obs_tz", "Tp": "obs_tp", "Ts": "obs_ts",
                              "pdir": "obs_pdir", "avedir": "obs_dir"})


def load_wind() -> pd.DataFrame:
    """Hourly wind from weather station AMS1 (already corrected to 10 m in the file)."""
    df = pd.read_excel(_xlsx(), sheet_name="AMS1")
    df = _clean_dates(df)
    need = ["wind speed 1 min at 10m", "wind speed 3 sec at 10m", "wind dir 1 min"]
    df = df[need].apply(pd.to_numeric, errors="coerce")
    df.index = _to_utc(df.index)
    g = df.resample("1h")
    out = pd.DataFrame({
        "obs_wind10": g["wind speed 1 min at 10m"].mean(),
        "obs_gust10": g["wind speed 3 sec at 10m"].max(),
        "obs_wdir": g["wind dir 1 min"].apply(_circular_mean),
    })
    return out.dropna(subset=["obs_wind10"])


def load_buoy1_level() -> pd.Series:
    """Hourly measured water level from Buoy 1 (used to fit the tide model)."""
    df = pd.read_excel(_xlsx(), sheet_name="Buoy1")
    df = _clean_dates(df)
    lvl = pd.to_numeric(df["Water Level"], errors="coerce").dropna()
    lvl.index = _to_utc(lvl.index)
    return lvl.resample("1h").mean().dropna().rename("level")


if __name__ == "__main__":
    b2, w, lv = load_buoy2(), load_wind(), load_buoy1_level()
    print("Buoy 2 hourly rows :", len(b2), b2.index.min(), "->", b2.index.max())
    print("   rows with Hs     :", int(b2["obs_hs"].notna().sum()))
    print("Wind hourly rows    :", len(w), w.index.min(), "->", w.index.max())
    both = b2.join(w, how="inner").dropna(subset=["obs_hs", "obs_wind10"])
    print("Waves + wind rows   :", len(both))
    print("Buoy 1 level hours  :", len(lv), lv.index.min(), "->", lv.index.max())
    print(b2["obs_hs"].describe().round(2).to_string())
    print("Hs percentiles 60/85/90:", b2["obs_hs"].quantile([.6, .85, .9]).round(2).tolist())
