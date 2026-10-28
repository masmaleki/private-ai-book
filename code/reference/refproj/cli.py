"""Command line: python -m refproj.cli {setup,train,eval,ask,demo}"""

import argparse
import json
import sys
from . import data_gen, db, evaluate, price_model
from .orchestrator import assist_quote
from .tools import SESSIONS, Tools, approve_quote


def cmd_setup(_):
    n_parts, n_sup, n_off = data_gen.build_db()
    print(f"demo.db: {n_parts} parts, {n_sup} suppliers, {n_off} offers")


def cmd_train(_):
    model, feat, tr, te = price_model.train(db.offers_frame(db.connect()))
    metrics = price_model.evaluate(model, feat, te, tr)
    price_model.save(model, feat, metrics)
    print(json.dumps(metrics))


def cmd_eval(_):
    card = price_model.load()[2]
    report = evaluate.run(card["metrics"])
    for name, ok in report["checks"].items():
        print(("PASS " if ok else "FAIL ") + name)
    print("rag:", {k: report["rag"][k] for k in ("hit_at_3", "leaks")})
    sys.exit(0 if report["release_ok"] else 1)


def cmd_ask(a):
    s = SESSIONS[a.token]
    tools = Tools(db.connect())
    out = assist_quote(tools, s, a.customer, a.mpn, a.qty, a.margin)
    print(json.dumps(out, indent=1))


def cmd_demo(_):
    conn = db.connect()
    tools = Tools(conn)
    out = assist_quote(tools, SESSIONS["tok-sales"], "C-ALPEN-01",
                       "TPS5430DDAR", 1000)
    print("explanation:", out["explanation"])
    print("draft:", out["draft"])
    qid = out["draft"]["quote_id"]
    for who in ("tok-sales", "tok-ns", "tok-sarah"):
        print(f"approve by {SESSIONS[who]['user']}:",
              approve_quote(conn, SESSIONS[who], qid))


def main():
    p = argparse.ArgumentParser(prog="refproj")
    sub = p.add_subparsers(required=True)
    for name in ("setup", "train", "eval", "demo"):
        sub.add_parser(name).set_defaults(fn=globals()[f"cmd_{name}"])
    ask = sub.add_parser("ask")
    ask.add_argument("mpn")
    ask.add_argument("qty", type=int)
    ask.add_argument("--customer", default="C-ALPEN-01")
    ask.add_argument("--token", default="tok-sales")
    ask.add_argument("--margin", type=float, default=18.0)
    ask.set_defaults(fn=cmd_ask)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
