"""Build the hourly training table: Matara measurements + Open-Meteo model data + features."""
import numpy as np
import pandas as pd

from config import PROC
from src import matara, openmeteo, tide

TRAIN_PATH = PROC / "train.parquet"


# target name -> (measured column, raw model column)
TARGETS = {"hs": ("obs_hs", "f_hs"), "tm": ("obs_tz", "f_tm"), "wind": ("obs_wind10", "f_wind")}

S1_FEATURES = ["f_hs", "f_tm", "f_wavedir_sin", "f_wavedir_cos", "f_swh", "f_swp", "f_swell_frac",
               "f_wind", "f_winddir_sin", "f_winddir_cos", "f_gust",
               "hour_sin", "hour_cos", "month", "season", "tide_h", "tide_rate"]


def season_of(month):
    """0 = south-west monsoon (May-Sep), 1 = north-east monsoon (Dec-Feb), 2 = inter-monsoon."""
    return np.select([(month >= 5) & (month <= 9), (month == 12) | (month <= 2)], [0, 1], default=2)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    hour = out.index.hour
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["month"] = out.index.month
    out["season"] = season_of(out.index.month)
    for col in ("f_wavedir", "f_swdir", "f_winddir"):
        if col in out:
            rad = np.deg2rad(out[col])
            out[col + "_sin"], out[col + "_cos"] = np.sin(rad), np.cos(rad)
    if "f_swh" in out and "f_hs" in out:
        out["f_swell_frac"] = (out["f_swh"] / out["f_hs"].where(out["f_hs"] > 0.05)).clip(0, 1.5)
    out = out.join(tide.predict(out.index))
    return out


def build_training_table() -> pd.DataFrame:
    waves = matara.load_buoy2()
    wind = matara.load_wind()
    obs = waves.join(wind, how="left")
    start, end = str(obs.index.min().date()), str(obs.index.max().date())
    model = openmeteo.marine_history(start, end).join(openmeteo.wind_history(start, end), how="outer")
    df = obs.join(model, how="left")
    df = add_features(df)
    df.to_parquet(TRAIN_PATH)
    return df


def load_training_table() -> pd.DataFrame:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError("Run `python -m src.dataset` first.")
    return pd.read_parquet(TRAIN_PATH)


if __name__ == "__main__":
    df = build_training_table()
    print("Saved", TRAIN_PATH, df.shape)
    print("Rows with measured Hs and model Hs :", int(df[["obs_hs", "f_hs"]].notna().all(axis=1).sum()))
    print("Rows with measured wind and model wind:", int(df[["obs_wind10", "f_wind"]].notna().all(axis=1).sum()))
    d = df[["obs_hs", "f_hs"]].dropna()
    err = d["f_hs"] - d["obs_hs"]
    print(f"Raw model Hs vs buoy: r={d.corr().iloc[0,1]:.3f}  bias={err.mean():+.2f} m  "
          f"MAE={err.abs().mean():.2f} m  RMSE={np.sqrt((err**2).mean()):.2f} m")
