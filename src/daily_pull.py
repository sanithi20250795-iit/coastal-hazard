"""Scheduled job: pull and store today's forecast (retries are built into the request helper)."""
from src import live

if __name__ == "__main__":
    res = live.run()
    m = res["meta"]
    print(f"mode={m['mode']} fetched_at={m['fetched_at_utc']} age_h={m['age_hours']} rows={len(res['df'])}")
    if m["mode"] != "live":
        print("WARNING: using stored data:", m["error"])
