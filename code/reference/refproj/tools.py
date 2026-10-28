"""Safe tool layer: session-bound tenant, strict args, roles, audit, approval.

Read tools return data. The only write tool an AI may call creates a DRAFT
with status pending_approval. Approval is a separate, human-only action."""

import hashlib
import json
import time
from datetime import date, timedelta
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from . import price_model
from .config import (AS_OF, AUDIT_LOG, MIN_MARGIN_PCT, QUOTE_VALID_DAYS,
                     TENANTS)
from .rag import Index

# Example token table. In reality: verify a signed token from the IdP.
SESSIONS = {
    "tok-sales": dict(user="sales-01", tenant="alpenboard", role="sales"),
    "tok-sarah": dict(user="sarah", tenant="alpenboard", role="sales_lead"),
    "tok-ns": dict(user="ns-sales", tenant="nordsensor", role="sales"),
    "tok-d": dict(user="d-sales", tenant="tenant-d", role="sales"),
    "tok-reza": dict(user="reza", tenant="alpenboard", role="developer"),
}


SALES = {"sales", "sales_lead"}


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")  # reject unexpected arguments


class MpnArgs(Strict):
    mpn: str = Field(min_length=3, max_length=40, pattern=r"^[A-Za-z0-9,.\-]+$")


class OffersArgs(MpnArgs):
    qty: int = Field(default=1, ge=1, le=1_000_000)


class PriceArgs(OffersArgs):
    supplier_id: int = Field(default=1, ge=1, le=99)


class SearchArgs(Strict):
    query: str = Field(min_length=2, max_length=300)
    k: int = Field(default=3, ge=1, le=8)


class DraftArgs(PriceArgs):
    customer_id: str = Field(min_length=2, max_length=40)
    unit_price: float = Field(gt=0, lt=100_000)
    cost_price: float = Field(gt=0, lt=100_000)
    explanation: str = Field(default="", max_length=2000)


class Tools:
    def __init__(self, db, index=None, model=None):
        self.db = db
        self.index = index or Index().add_folder()
        self.model, self.feat, self.card = model or price_model.load()

    # ---------- read tools ----------
    def get_part_offers(self, s, a: OffersArgs):
        sql = """SELECT o.supplier_id, su.name, su.authorized, o.qty,
                        o.unit_price, o.lead_days, o.observed_at
                 FROM demo_offers o JOIN demo_parts p USING(part_id)
                 JOIN demo_suppliers su USING(supplier_id)
                 WHERE o.tenant_id = ? AND p.mpn_norm = ? AND o.qty <= ?
                 ORDER BY o.observed_at DESC LIMIT 5"""
        # qty <= requested: only price breaks that apply to this order size
        args = (s["tenant"], a.mpn.upper(), a.qty)
        rows = self.db.execute(sql, args).fetchall()
        offers = [dict(r) for r in rows]
        if s["tenant"] == "tenant-d":  # tenant policy: no broker offers
            offers = [o for o in offers if o["authorized"]]
        return offers

    def predict_price(self, s, a: PriceArgs):
        known = a.mpn.upper() in self.feat.part_level
        X = self.feat.row(a.mpn.upper(), a.qty, a.supplier_id, AS_OF)
        p10, p50, p90 = (round(float(v[0]), 4) for v in self.model.predict(X))
        return dict(p10=p10, p50=p50, p90=p90, currency="EUR",
                    known_part=known, model_version=self.card["version"])

    def search_docs(self, s, a: SearchArgs):
        return self.index.search(a.query, s, k=a.k)

    # ---------- write tool (draft only) ----------
    def create_quote_draft(self, s, a: DraftArgs):
        margin = round(100 * (a.unit_price - a.cost_price) / a.unit_price, 2)
        warnings = []
        if margin < MIN_MARGIN_PCT:
            warnings.append(f"margin {margin}% below {MIN_MARGIN_PCT}%")
        pred = self.predict_price(s, PriceArgs(mpn=a.mpn, qty=a.qty,
                                               supplier_id=a.supplier_id))
        if not pred["known_part"]:
            warnings.append("part unknown to price model")
        elif a.cost_price > pred["p90"]:
            warnings.append("cost above model P90: negotiate or recheck")
        elif a.cost_price < pred["p10"]:
            warnings.append("cost below model P10: check offer validity")
        sup = self.db.execute("SELECT authorized FROM demo_suppliers "
                              "WHERE supplier_id=?", (a.supplier_id,)).fetchone()
        if sup is None or not sup[0]:
            warnings.append("broker offer: needs customer consent")
        n = self.db.execute("SELECT COUNT(*) FROM demo_quote_lines").fetchone()
        qid = f"Q-2026-{n[0] + 1:05d}"
        valid = (date.today() + timedelta(days=QUOTE_VALID_DAYS)).isoformat()
        part = self.db.execute("SELECT part_id FROM demo_parts WHERE mpn_norm=?",
                               (a.mpn.upper(),)).fetchone()
        self.db.execute(
            "INSERT INTO demo_quote_lines VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (qid, s["tenant"], a.customer_id, part[0] if part else None, a.qty,
             a.unit_price, margin, valid, "pending_approval", s["user"], None,
             a.explanation, json.dumps(warnings), None))
        self.db.commit()
        return dict(quote_id=qid, status="pending_approval", margin_pct=margin,
                    warnings=warnings)

    REGISTRY = {  # name -> (method, args model, roles allowed)
        "get_part_offers": ("get_part_offers", OffersArgs, SALES),
        "predict_price": ("predict_price", PriceArgs, SALES),
        "search_docs": ("search_docs", SearchArgs, SALES),
        "create_quote_draft": ("create_quote_draft", DraftArgs, SALES),
    }

    def call(self, s, name, args):
        t0 = time.perf_counter()
        out = None
        if name not in self.REGISTRY:
            status = "denied_unknown_tool"
        elif s["role"] not in self.REGISTRY[name][2]:
            status = "denied_role"
        else:
            method, schema, _ = self.REGISTRY[name]
            try:
                out = getattr(self, method)(s, schema.model_validate(args))
                status = "ok"
            except ValidationError as e:
                status = "invalid_args"
                out = [f"{'.'.join(map(str, x['loc']))}: {x['msg']}"
                       for x in e.errors()]
        audit(s, name, args, status, time.perf_counter() - t0)
        if status not in ("ok", "invalid_args"):
            out = {"error": status}
        return status, out


def approve_quote(db, s, quote_id):
    """Human-only action. NOT in the tool registry, so no model can call it."""
    row = db.execute("SELECT * FROM demo_quote_lines WHERE quote_id=?",
                     (quote_id,)).fetchone()
    if row is None or row["tenant_id"] != s["tenant"]:
        status = "not_found"  # never reveal other tenants' quotes
    elif s["role"] != "sales_lead":
        status = "denied_role"
    elif row["created_by"] == s["user"]:
        status = "denied_self_approval"
    elif row["status"] != "pending_approval":
        status = "denied_state"
    else:
        db.execute("UPDATE demo_quote_lines SET status='approved', "
                   "approved_by=?, issued_at=? WHERE quote_id=?",
                   (s["user"], date.today().isoformat(), quote_id))
        db.commit()
        status = "ok"
    audit(s, "approve_quote", {"quote_id": quote_id}, status, 0.0)
    return status


def audit(s, tool, args, status, seconds):
    digest = hashlib.sha256(json.dumps(args, sort_keys=True).encode())
    rec = dict(ts=time.strftime("%Y-%m-%dT%H:%M:%S"), user=s.get("user"),
               tenant=s.get("tenant"), tool=tool, status=status,
               args_sha256=digest.hexdigest()[:16], ms=round(seconds * 1e3, 1))
    AUDIT_LOG.parent.mkdir(exist_ok=True)
    with open(AUDIT_LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")


assert set(TENANTS) >= {s["tenant"] for s in SESSIONS.values()}
