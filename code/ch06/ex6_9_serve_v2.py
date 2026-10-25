"""Exercise 6-9: serve.py plus a /model endpoint and a far-future date guard.

Run:  uvicorn ex6_9_serve_v2:app --host 127.0.0.1 --port 8006
Only listens on localhost; nothing leaves the machine.
"""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

HERE = Path(__file__).parent
MODEL_PATH = HERE / "model/price_linear_v1.joblib"
META = json.loads((HERE / "model/price_linear_v1.json").read_text())

# refuse to load a model file whose hash does not match its metadata
if hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() != META["sha256"]:
    raise RuntimeError("model file hash mismatch: refusing to load")
MODEL = joblib.load(MODEL_PATH)
EPOCH = pd.Timestamp("2024-01-01")

app = FastAPI(title="price-model (example)", version=META["version"])


class PriceRequest(BaseModel):
    ref_price_eur: float = Field(gt=0, le=10_000, description="catalogue price")
    quantity: int = Field(gt=0, le=10_000_000)
    quote_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")


class PriceResponse(BaseModel):
    unit_price_eur: float
    model: str
    version: str


@app.get("/health")
def health():
    return {"status": "ok", "model": META["model"], "version": META["version"]}


@app.get("/model")
def model_info():
    # expose metadata, but never file paths or internals
    return {k: META[k] for k in ("model", "version", "features", "train_until")}


MAX_DATE = pd.Timestamp(META["train_until"]) + pd.Timedelta(days=365)


@app.post("/predict", response_model=PriceResponse)
def predict(req: PriceRequest):
    try:
        day = (pd.Timestamp(req.quote_date) - EPOCH).days
    except ValueError:
        raise HTTPException(status_code=422, detail="invalid date")
    if pd.Timestamp(req.quote_date) > MAX_DATE:
        raise HTTPException(status_code=422, detail="date too far beyond training data")
    row = pd.DataFrame([{"log_ref": np.log(req.ref_price_eur),
                         "log10_qty": np.log10(req.quantity), "days": day}])
    price = float(np.exp(MODEL.predict(row[META["features"]])[0]))
    return PriceResponse(unit_price_eur=round(price, 4), model=META["model"],
                         version=META["version"])


if __name__ == "__main__":
    from fastapi.testclient import TestClient
    client = TestClient(app)
    print(client.get("/model").json())
    ok = {"ref_price_eur": 2.0, "quantity": 500, "quote_date": "2025-09-01"}
    far = {**ok, "quote_date": "2031-01-01"}
    for body in (ok, far):
        r = client.post("/predict", json=body)
        print(r.status_code, r.json())
