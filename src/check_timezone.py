"""Check the buoy clock: finds the shift (hours) at which buoy Hs best matches ERA5-Ocean Hs."""
import pandas as pd

from src import matara, openmeteo

obs = matara.load_buoy2()["obs_hs"]
model = openmeteo.marine_history(str(obs.index.min().date()), str(obs.index.max().date()))["f_hs"]
rows = []
for shift in range(-12, 13):
    r = obs.shift(shift, freq="h").to_frame("o").join(model, how="inner").dropna().corr().iloc[0, 1]
    rows.append((shift, round(float(r), 4)))
best = max(rows, key=lambda x: x[1])
print(pd.DataFrame(rows, columns=["shift_h", "correlation"]).to_string(index=False))
print(f"\nBest shift: {best[0]} h (r={best[1]}).")
print("If the best shift is not 0, change BUOY_UTC_OFFSET_H in config.py by that many hours "
      "(a shift of +s means the buoy times need to move s hours later, so SUBTRACT s from the offset) "
      "and rerun src.tide and src.dataset.")
