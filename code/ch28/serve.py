"""serve.py - price prediction API (teaching). Run: uvicorn serve:app --port 8077
POST /v1/predict/price  {"part_id": 0, "qty": 2500, "asof": "2026-09-30"}
The service returns numbers + model version + inputs used; the LLM only
narrates these numbers later (never invents them).
"""
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
from predict_one import features_for, bundle

app = FastAPI(title="price-demo")
CONFORMAL_WIDEN = 0.02           # from calibrate.py (log scale), versioned

class PriceRequest(BaseModel):
    part_id: int
    qty: int = Field(gt=0, le=1_000_000)
    asof: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")

@app.post("/v1/predict/price")
def predict_price(req: PriceRequest):
    row = features_for(req.part_id, req.qty, req.asof)
    if row.empty:
        raise HTTPException(404, "no price history before asof (cold start)")
    X = row[bundle["features"]]
    ref = float(row.ref_price.iloc[0])
    p = {k: float(m.predict(X)[0]) for k, m in bundle["models"].items()}
    out = {"p10": ref * np.exp(p["p10"] - CONFORMAL_WIDEN),
           "p50": ref * np.exp(p["p50"]),
           "p90": ref * np.exp(p["p90"] + CONFORMAL_WIDEN)}
    return {"part_id": req.part_id, "mpn": row.mpn.iloc[0], "qty": req.qty,
            "asof": req.asof, "horizon_days": 30, "currency": "EUR",
            **{k: round(v, 4) for k, v in out.items()},
            "reference_price": round(ref, 4),
            "n_offers_90d": int(row.n_offers_90.iloc[0]),
            "model_version": bundle["version"], "data": "synthetic"}

if __name__ == "__main__":
    from fastapi.testclient import TestClient
    c = TestClient(app)
    r = c.post("/v1/predict/price",
               json={"part_id": 0, "qty": 2500, "asof": "2026-09-30"})
    print(r.status_code, r.json())
    r = c.post("/v1/predict/price",
               json={"part_id": 0, "qty": -5, "asof": "2026-09-30"})
    print(r.status_code, r.json()["detail"][0]["msg"])
