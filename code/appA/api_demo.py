"""Appendix A: calling a JSON API with requests, against a tiny local test server.

The server returns fixed example numbers; it only exists so the client code runs.
"""
import json
import os
import threading
from pprint import pprint
from http.server import BaseHTTPRequestHandler, HTTPServer

import requests


class FakePriceAPI(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/v1/predict/price":
            self.send_error(404, "unknown path")
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.headers.get("Authorization") != "Bearer demo-token":
            self.send_error(401, "missing or wrong token")
            return
        answer = {"mpn": body["mpn"], "qty": body["qty"],          # example numbers
                  "p10": 1.96, "p50": 2.02, "p90": 2.30,
                  "model_version": "price-lgbm-demo"}
        data = json.dumps(answer).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):        # keep the demo output quiet
        pass


def predict_price(base_url, mpn, qty, token):
    resp = requests.post(
        f"{base_url}/v1/predict/price",
        json={"mpn": mpn, "qty": qty},               # requests sets the JSON header
        headers={"Authorization": f"Bearer {token}"},
        timeout=5,                                   # never wait forever
    )
    resp.raise_for_status()                          # 4xx/5xx -> exception
    return resp.json()


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", 0), FakePriceAPI)     # port 0 = any free port
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}"

    os.environ.setdefault("PRICE_API_TOKEN", "demo-token")  # normally set outside code
    token = os.environ["PRICE_API_TOKEN"]
    pprint(predict_price(url, "EXS-BUCK36-SO8", 1000, token), width=60, sort_dicts=False)
    try:
        predict_price(url, "EXS-BUCK36-SO8", 1000, "wrong")
    except requests.HTTPError as err:
        print("error:", err.response.status_code, err.response.reason)
    try:
        requests.get("http://10.255.255.1", timeout=1)      # unreachable address
    except requests.exceptions.RequestException as err:
        print("network problem:", type(err).__name__)
    server.shutdown()
