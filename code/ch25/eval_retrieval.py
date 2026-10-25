"""eval_retrieval.py - Recall@5, MRR and nDCG@5 for four retrieval setups,
plus a cross-tenant leak test. Gold labels are hand-made example data."""

import math, json
from secure_rag import SecureRAG, allowed_set, rerank, rewrite

GOLD = [  # query -> {chunk_id: grade}  (2 = answers it, 1 = useful context)
    (
        "EXS-BUCK36-SO8 maximum input voltage",
        {"DS-0001#2": 2, "DS-0001#4": 2, "DS-0001#0": 1, "PCN-2026-014#0": 1},
    ),
    (
        "output current at 105 °C EXS-BUCK36-SO8",
        {"DS-0001#2": 2, "DS-0001#3": 2},
    ),
    ("EXSLDO500 dropout", {"DS-0002#1": 2}),
    ("thermal vias under the exposed pad", {"AN-0101#1": 2, "DS-0001#3": 1}),
    ("returns of opened reels", {"SUP-A-03#0": 2}),
    ("Lieferzeit Werksbestellung", {"FAQ-DE-01#0": 2}),
    ("which MCU variant for 105 °C", {"DS-0003#2": 2, "DS-0003#0": 2}),
    ("minimum margin for broker stock", {"POL-0007#0": 2}),
    ("Markus lead time preference", {"CUST-AB-01#0": 2}),
    ("last time buy rev B", {"PCN-2026-014#0": 2}),
    # harder ones: abbreviation, paraphrase, other language, bare MPN
    ("VIN abs max EXS-BUCK36-SO8", {"DS-0001#4": 2, "DS-0001#2": 1}),
    ("how hot can the microcontroller run", {"DS-0003#0": 2}),
    ("Rücksendung geöffnete Rolle", {"SUP-A-03#0": 2}),
    ("EXSLDO500", {"DS-0002#0": 2, "DS-0002#1": 1}),
    (
        "regulator for 12 V to 3.3 V at 400 mA",
        {"DS-0002#2": 2, "DS-0001#0": 1},
    ),
]


def recall_at(ranked, rel, k=5):
    return len([c for c in ranked[:k] if c in rel]) / len(rel)


def mrr(ranked, rel):
    return next((1 / i for i, c in enumerate(ranked, 1) if c in rel), 0.0)


def ndcg_at(ranked, rel, k=5):
    dcg = sum(
        (2 ** rel.get(c, 0) - 1) / math.log2(i + 1)
        for i, c in enumerate(ranked[:k], 1)
    )
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2**g - 1) / math.log2(i + 1) for i, g in enumerate(ideal, 1))
    return dcg / idcg


def run(rag, session, setup):
    allowed = allowed_set(rag.index, session)
    allowed -= {
        i
        for i, c in enumerate(rag.index.chunks)
        if c.chunk_id in rag.quarantine
    }
    rows = []
    for q, rel in GOLD:
        if setup == "hybrid+rerank":
            hits = rerank(
                rewrite(q),
                rag.index.search(rewrite(q), k=12, allowed=allowed),
                top=10,
            )
        else:
            hits = rag.index.search(q, k=10, mode=setup, allowed=allowed)
        ranked = [c.chunk_id for c, _ in hits]
        rows.append(
            (recall_at(ranked, rel), mrr(ranked, rel), ndcg_at(ranked, rel))
        )
    return [sum(col) / len(rows) for col in zip(*rows)]


def leak_test(rag):
    """Each tenant asks probing questions; no other tenant's chunk may show."""
    probes = [
        "target price account notes",
        "NordSensor payment terms",
        "AlpenBoard approved converter",
        "customer notes EXS-BUCK36-SO8",
    ]
    sessions = [
        {"user": "u1", "tenant": "alpenboard", "role": "sales", "staff": True},
        {"user": "u2", "tenant": "nordsensor", "role": "sales", "staff": True},
    ]
    leaks_secure = leaks_open = 0
    for s in sessions:
        for p in probes:
            for secure in (True, False):
                hits = (
                    rag.retrieve(p, s, k=10)
                    if secure
                    else rag.index.search(p, k=10)
                )
                bad = [
                    c
                    for c, _ in hits
                    if c.meta.get("visibility") == "tenant"
                    and c.meta.get("tenant") != s["tenant"]
                ]
                if secure:
                    leaks_secure += len(bad)
                else:
                    leaks_open += len(bad)
    return leaks_open, leaks_secure


if __name__ == "__main__":
    rag = SecureRAG()
    staff = {
        "user": "sarah",
        "tenant": "alpenboard",
        "role": "sales",
        "staff": True,
    }
    print(f"{'setup':<15}{'Recall@5':>9}{'MRR':>7}{'nDCG@5':>8}")
    results = {}
    for setup in ("bm25", "vector", "hybrid", "hybrid+rerank"):
        r, m, n = run(rag, staff, setup)
        results[setup] = [round(r, 3), round(m, 3), round(n, 3)]
        print(f"{setup:<15}{r:>9.3f}{m:>7.3f}{n:>8.3f}")
    open_, secure = leak_test(rag)
    print(
        f"\nleak test: other-tenant chunks in top-10 -> without ACL: {open_}, "
        f"with ACL: {secure}"
    )
    json.dump(results, open("eval_results.json", "w"))
