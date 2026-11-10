"""Clock check for the buoy timestamps, using tides.

Tides repeat about every 12.4 hours, so a clock error shows up as a clear shift (waves change too slowly to show it).
This compares the fitted tide model with Open-Meteo's tide-driven sea level, scanning in 15-minute steps.

Run after `python -m src.tide`: python -m src.check_tide_clock
"""
import numpy as np
import pandas as pd

from config import BUOY_UTC_OFFSET_H
from src import openmeteo, tide

om = openmeteo.marine_forecast(3)
if "f_sea_level" not in om:
    raise SystemExit("Open-Meteo returned no sea level, so this check cannot run.")
om = om["f_sea_level"].dropna()

rows = []
for s in np.arange(-6, 6.01, 0.25):
    pred = tide.predict(om.index - pd.Timedelta(hours=float(s)))["tide_h"].to_numpy()
    rows.append((float(s), float(np.corrcoef(pred, om.to_numpy())[0, 1])))
table = pd.DataFrame(rows, columns=["shift_h", "correlation"])
print(table[table.shift_h.mod(1) == 0].round(3).to_string(index=False))
best = table.loc[table.correlation.idxmax()]
s, r = float(best.shift_h), float(best.correlation)
print(f"\nBest shift: {s:+.2f} h (r={r:.3f}).")

if r < 0.9:
    print("The match is not clear enough to act on. Send this output to the team.")
else:
    new = BUOY_UTC_OFFSET_H - s
    new_rounded = round(new * 2) / 2
    print(f"Current BUOY_UTC_OFFSET_H = {BUOY_UTC_OFFSET_H}. Implied value = {new:.2f}, nearest half hour = {new_rounded}.")
    if abs(new_rounded - BUOY_UTC_OFFSET_H) < 0.25:
        print("The tide model agrees with Open-Meteo. No change needed.")
    else:
        print(f"Set BUOY_UTC_OFFSET_H = {new_rounded} in config.py, save, then rerun in this order:\n"
              " python -m src.tide\n python -m src.check_tide_clock (it should now say no change needed)\n"
              " python -m src.check_timezone\n python -m src.dataset\n python -m src.train_stage1\n python -m src.train_stage2")