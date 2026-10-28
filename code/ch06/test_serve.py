"""Smoke tests for serve.py using FastAPI's in-process test client."""
from fastapi.testclient import TestClient

from serve import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_predict_volume_discount():
    base = {"ref_price_eur": 2.0, "quote_date": "2025-09-01"}
    small = client.post("/predict", json={**base, "quantity": 10}).json()
    large = client.post("/predict", json={**base, "quantity": 10000}).json()
    assert large["unit_price_eur"] < small["unit_price_eur"]


def test_rejects_bad_input():
    bad = {"ref_price_eur": -1, "quantity": 10, "quote_date": "2025-09-01"}
    assert client.post("/predict", json=bad).status_code == 422
