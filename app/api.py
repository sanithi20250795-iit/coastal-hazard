"""FastAPI service.  Run from the project root:  uvicorn app.api:app --reload"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException, Query  # noqa: E402

from src import live  # noqa: E402

app = FastAPI(title="Coastal Hazard Forecast (Matara)", version="1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/forecast")
def forecast(skill: str = Query("beginner", pattern="^(beginner|intermediate|advanced)$")):
    try:
        return live.to_json(live.run(), skill)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))
