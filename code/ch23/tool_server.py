"""tool_server.py - minimal MCP-style tool server over stdio (JSON-RPC 2.0).

Teaching version of the MCP ideas: initialize, tools/list, tools/call.
For production use the official MCP SDK and the current specification.
Security properties shown here:
  * read-only tools only
  * tenant and role come from the authenticated SESSION, never from arguments
  * every argument is validated against a schema; unknown fields are rejected
  * policy check before execution; every call is written to an audit log
"""

import json, os, sys, time, hashlib
from pydantic import BaseModel, Field, ValidationError, ConfigDict
from demo_db import build

DB = build()
AUDIT = os.environ.get("AUDIT_LOG", "audit.jsonl")
# Example token table. In reality: verify a signed token from the
# identity provider.
SESSIONS = {
    "tok-sarah": {"user": "sarah", "tenant": "alpenboard", "role": "sales"},
    "tok-reza": {"user": "reza", "tenant": "alpenboard", "role": "developer"},
    "tok-ns": {"user": "ns-buyer", "tenant": "nordsensor", "role": "sales"},
}


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")  # reject unexpected arguments


class PartOffersArgs(Strict):
    mpn: str = Field(min_length=3, max_length=40)
    min_qty: int = Field(default=1, ge=1, le=1_000_000)


class InventoryArgs(Strict):
    mpn: str = Field(min_length=3, max_length=40)


class PredictArgs(Strict):
    mpn: str = Field(min_length=3, max_length=40)
    qty: int = Field(ge=1, le=1_000_000)


def get_part_offers(s, a: PartOffersArgs):
    rows = DB.execute(
        """
        SELECT su.name, su.authorized, o.qty, o.unit_price, o.currency,
               o.lead_days
        FROM demo_offers o JOIN demo_parts p USING(part_id)
        JOIN demo_suppliers su USING(supplier_id)
        WHERE o.tenant_id = ? AND p.mpn_norm = ? AND o.qty >= ?
        ORDER BY o.unit_price""",
        (s["tenant"], a.mpn.upper(), a.min_qty),
    ).fetchall()
    return [
        dict(
            zip(
                [
                    "supplier",
                    "authorized",
                    "qty",
                    "unit_price",
                    "currency",
                    "lead_days",
                ],
                r,
            )
        )
        for r in rows
    ]


def get_inventory(s, a: InventoryArgs):
    r = DB.execute(
        """SELECT i.on_hand, i.reserved, i.snapshot_at
        FROM demo_inventory i JOIN demo_parts p USING(part_id)
        WHERE i.tenant_id = ? AND p.mpn_norm = ?""",
        (s["tenant"], a.mpn.upper()),
    ).fetchone()
    return (
        None
        if r is None
        else {
            "on_hand": r[0],
            "reserved": r[1],
            "available": r[0] - r[1],
            "snapshot_at": r[2],
        }
    )


def predict_price(s, a: PredictArgs):
    # Stub: a real version calls the price model service (chapter 27).
    offers = get_part_offers(s, PartOffersArgs(mpn=a.mpn, min_qty=1))
    if not offers:
        return None
    prices = sorted(o["unit_price"] for o in offers)
    mid = prices[len(prices) // 2]
    return {
        "p50": round(mid, 4),
        "p10": round(prices[0], 4),
        "p90": round(prices[-1], 4),
        "model_version": "stub-0",
        "currency": "EUR",
    }


TOOLS = {  # name -> (function, args model, description, roles allowed)
    "get_part_offers": (
        get_part_offers,
        PartOffersArgs,
        "Read current supplier offers for one MPN (own tenant only). "
        "Returns supplier, qty, unit_price EUR, lead_days.",
        {"sales"},
    ),
    "get_inventory": (
        get_inventory,
        InventoryArgs,
        "Read on-hand and reserved stock for one MPN in our warehouse.",
        {"sales"},
    ),
    "predict_price": (
        predict_price,
        PredictArgs,
        "Get the price model's P10/P50/P90 unit price for an MPN and quantity. "
        "Numbers come from the model service, never invent them.",
        {"sales"},
    ),
}


def audit(session, tool, args, status, ms):
    rec = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "user": session.get("user"),
        "tenant": session.get("tenant"),
        "tool": tool,
        "status": status,
        "args_sha256": hashlib.sha256(
            json.dumps(args, sort_keys=True).encode()
        ).hexdigest()[:16],
        "ms": ms,
    }
    with open(AUDIT, "a") as f:
        f.write(json.dumps(rec) + "\n")


def call_tool(session, name, args):
    t0 = time.perf_counter()
    if name not in TOOLS:
        status, out = "denied_unknown_tool", {"error": f"unknown tool {name}"}
    elif session["role"] not in TOOLS[name][3]:
        status, out = "denied_role", {"error": "not permitted for your role"}
    else:
        fn, model = TOOLS[name][0], TOOLS[name][1]
        try:
            out, status = fn(session, model.model_validate(args)), "ok"
        except ValidationError as e:
            status = "invalid_args"
            out = {
                "error": [
                    f"{'.'.join(map(str, x['loc']))}: {x['msg']}"
                    for x in e.errors()
                ]
            }
    audit(
        session,
        name,
        args,
        status,
        round((time.perf_counter() - t0) * 1000, 1),
    )
    return status, out


def handle(session, req):
    m, rid = req.get("method"), req.get("id")
    if m == "initialize":
        res = {
            "serverInfo": {"name": "demo-erp-tools", "version": "0.1"},
            "capabilities": {"tools": {}},
        }
    elif m == "tools/list":
        res = {
            "tools": [
                {
                    "name": n,
                    "description": t[2],
                    "inputSchema": t[1].model_json_schema(),
                }
                for n, t in TOOLS.items()
                if session["role"] in t[3]
            ]
        }
    elif m == "tools/call":
        p = req.get("params", {})
        status, out = call_tool(session, p.get("name"), p.get("arguments", {}))
        res = {
            "isError": status != "ok",
            "content": [{"type": "text", "text": json.dumps(out)}],
        }
    else:
        return {
            "jsonrpc": "2.0",
            "id": rid,
            "error": {"code": -32601, "message": "method not found"},
        }
    return {"jsonrpc": "2.0", "id": rid, "result": res}


if __name__ == "__main__":
    # The session is bound when the connection is opened (here: env var set by
    # the gateway after authenticating the user), NOT chosen by the model.
    session = SESSIONS.get(os.environ.get("SESSION_TOKEN", ""))
    if session is None:
        sys.exit("unauthenticated")
    for line in sys.stdin:
        if line.strip():
            print(json.dumps(handle(session, json.loads(line))), flush=True)
