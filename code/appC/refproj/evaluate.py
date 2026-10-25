"""Release gate: one report for all four building blocks. Fails loudly."""

import json
from .config import ARTIFACTS
from .llm import check_answer
from .rag import Index

GOLD = [  # (tenant, question, chunk that must be in the top 3)
    ("alpenboard", "what matters more for AlpenBoard lead time or price",
     "DOC-004#0"),
    ("nordsensor", "is NordSensor price sensitive", "DOC-005#0"),
    ("alpenboard", "minimum margin for a quote line", "DOC-003#0"),
    ("alpenboard", "who may approve and send a quote", "DOC-003#2"),
    ("nordsensor", "EXS-BUCK36-SO8 maximum output current", "DOC-001#0"),
    ("nordsensor", "EXS-LDO500-33 dropout voltage", "DOC-002#1"),
    ("tenant-d", "are broker offers allowed", "DOC-006#0"),
    ("alpenboard", "STM32F407VGT6 temperature range suffix", "DOC-008#0"),
]
LEAKS = [  # (tenant, question, doc that must NOT appear)
    ("nordsensor", "AlpenBoard Markus lead time production line", "DOC-004"),
    ("alpenboard", "NordSensor price sensitive volumes", "DOC-005"),
    ("alpenboard", "Tenant-D air-gapped broker", "DOC-006"),
]
GUARD = [  # (answer, should pass)
    ("Fastest offer costs 2.5 EUR [F1]", True),
    ("Fastest offer costs 2.4 EUR [F1]", False),   # number not in facts
    ("The offer is good.", False),                 # no citation
    ("Fastest offer costs 2.5 EUR [F9]", False),   # unknown citation
]


def rag_report(index):
    hits = []
    for tenant, q, want in GOLD:
        got = [h["id"] for h in index.search(q, dict(tenant=tenant,
                                                     role="sales"))]
        hits.append(want in got)
    leaks = 0
    for tenant, q, bad in LEAKS:
        got = index.search(q, dict(tenant=tenant, role="sales"), k=8)
        leaks += any(h["id"].startswith(bad) for h in got)
    return dict(hit_at_3=round(sum(hits) / len(hits), 3), leaks=leaks,
                quarantined=index.quarantine)


def guard_report():
    facts = [("F1", "Fastest offer costs 2.5 EUR.")]
    ok = [(not check_answer(a, facts)) == want for a, want in GUARD]
    return dict(guard_cases=len(ok), guard_correct=sum(ok))


def gate(price, rag, guard):
    checks = {
        "model beats baseline": price["mape_model"] < price["mape_baseline"],
        "interval coverage 0.70-0.90":
            0.70 <= price["coverage_p10_p90"] <= 0.90,
        "retrieval hit@3 >= 0.85": rag["hit_at_3"] >= 0.85,
        "zero cross-tenant leaks": rag["leaks"] == 0,
        "injection quarantined": len(rag["quarantined"]) >= 1,
        "guard catches all cases": guard["guard_correct"]
            == guard["guard_cases"],
    }
    return checks, all(checks.values())


def run(price_metrics):
    rag, guard = rag_report(Index().add_folder()), guard_report()
    checks, passed = gate(price_metrics, rag, guard)
    report = dict(price=price_metrics, rag=rag, guard=guard,
                  checks=checks, release_ok=passed)
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "eval_report.json").write_text(json.dumps(report, indent=1))
    return report
