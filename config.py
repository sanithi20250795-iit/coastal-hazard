"""Central settings. Edit values here, not inside the other files."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"              # put the Matara .xlsx file here
CACHE = DATA / "cache"          # raw API responses are cached here
PROC = DATA / "processed"       # cleaned tables, metrics, thresholds
LOG = DATA / "forecast_log"     # every daily forecast pull is stored here
MODELS = ROOT / "models"        # trained models
for _p in (RAW, CACHE, PROC, LOG, MODELS):
    _p.mkdir(parents=True, exist_ok=True)

# Matara buoy location used for the Open-Meteo requests (the API snaps to the nearest grid cell)
LAT, LON = 5.94, 80.55
LOCAL_TZ = "Asia/Colombo"

# ASSUMPTION: the buoy timestamps are Sri Lanka local time (UTC+5.5).
# Run `python -m src.check_timezone` and change this if it reports a different best shift.
BUOY_UTC_OFFSET_H = 5.5

SKILLS = ("beginner", "intermediate", "advanced")
LEVEL_NAMES = {0: "LOW", 1: "MODERATE", 2: "HIGH"}

# Initial hazard cut-offs (percentiles of the measured Buoy 2 wave height) per skill level.
# These are a STARTING POINT. Instructors/lifeguards review data/processed/thresholds.json
# and the values there are what the models use.
HAZARD_PERCENTILES = {
    "beginner": (60, 85),
    "intermediate": (80, 93),
    "advanced": (90, 98),
}
LONG_PERIOD_PERCENTILE = 80     # long-period swell bumps the level up by one step

# Live forecast behaviour
FORECAST_DAYS = 3
SHOW_HOURS = 48
STALE_AFTER_H = 24
