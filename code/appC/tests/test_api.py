from fastapi.testclient import TestClient
from refproj.api import app

client = TestClient(app)
H = lambda tok: {"Authorization": f"Bearer {tok}"}


def test_health_and_auth(built):
    assert client.get("/health").json() == {"status": "ok"}
    r = client.post("/v1/predict/price", json={"mpn": "NE555DR", "qty": 100})
    assert r.status_code == 401


def test_end_to_end_quote_flow(built):
    r = client.post("/v1/assist/quote", headers=H("tok-sales"), json=dict(
        customer_id="C-ALPEN-01", mpn="TPS5430DDAR", qty=1000))
    body = r.json()
    assert r.status_code == 200 and body["guard_problems"] == []
    qid = body["draft"]["quote_id"]
    url = f"/v1/quotes/{qid}/approve"
    assert client.post(url, headers=H("tok-sales")).status_code == 403
    assert client.post(url, headers=H("tok-ns")).status_code == 404
    assert client.post(url, headers=H("tok-sarah")).json()["status"] == "approved"


def test_extra_fields_rejected(built):
    r = client.post("/v1/predict/price", headers=H("tok-sales"),
                    json={"mpn": "NE555DR", "qty": 100, "tenant": "tenant-d"})
    assert r.status_code == 422
