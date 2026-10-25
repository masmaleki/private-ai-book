"""HTTP API for the ERP and the UI. Run: uvicorn refproj.api:app --port 8100"""

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from . import db as dbmod
from .orchestrator import assist_quote
from .tools import SESSIONS, Tools, approve_quote

app = FastAPI(title="refproj (example)", version="0.1")
STATE = {}


def tools():
    if "tools" not in STATE:  # lazy: tests can build the data first
        STATE["db"] = dbmod.connect()
        STATE["tools"] = Tools(STATE["db"])
    return STATE["tools"]


def session(authorization: str = Header(default="")):
    s = SESSIONS.get(authorization.removeprefix("Bearer "))
    if s is None:
        raise HTTPException(401, "unauthenticated")
    return s


class AssistReq(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customer_id: str = Field(min_length=2, max_length=40)
    mpn: str
    qty: int = Field(ge=1, le=1_000_000)
    margin_pct: float = Field(default=18.0, ge=0, le=60)


def run(s, name, args):
    status, out = tools().call(s, name, args)
    if status != "ok":
        code = 422 if status == "invalid_args" else 403
        raise HTTPException(code, {"status": status, "detail": out})
    return out


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/predict/price")
def predict(body: dict, s=Depends(session)):
    return run(s, "predict_price", body)


@app.post("/v1/search")
def search(body: dict, s=Depends(session)):
    return run(s, "search_docs", body)


@app.post("/v1/assist/quote")
def assist(body: AssistReq, s=Depends(session)):
    return assist_quote(tools(), s, **body.model_dump())


@app.post("/v1/quotes/{quote_id}/approve")
def approve(quote_id: str, s=Depends(session)):
    tools()
    status = approve_quote(STATE["db"], s, quote_id)
    if status != "ok":
        raise HTTPException(404 if status == "not_found" else 403, status)
    return {"quote_id": quote_id, "status": "approved"}
