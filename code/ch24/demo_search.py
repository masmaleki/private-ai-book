"""demo_search.py - BM25 vs n-gram vectors vs hybrid (RRF) on example queries.
Prints the rank of the chunk that answers each query (lower is better)."""

import json
from rag import load_docs, make_chunks, Index, NgramVectorizer

idx = Index(make_chunks(load_docs(), "structural"))
print(f"{len(idx.chunks)} chunks from 10 fictional documents\n")
QUERIES = [  # (query, prefix of the chunk that answers it) - example data
    ("EXS-BUCK36-SO8 maximum input voltage", "DS-0001#4"),
    ("EXSLDO500", "DS-0002"),  # MPN without dashes
    ("exs-buk36 derating", "DS-0001#3"),  # typo in the MPN
    ("2.1 A", "DS-0001#2"),  # a bare number
    ("which converter for a 24 V rail", "DS-0001#1"),
    ("how hot can the microcontroller run", "DS-0003#0"),  # paraphrase
    ("Rücksendung geöffnete Rolle", "SUP-A-03"),  # German, doc is English
]
results = []
print(f"{'query':<38}{'bm25':>6}{'vector':>8}{'hybrid':>8}")
for q, want in QUERIES:
    row = []
    for mode in ("bm25", "vector", "hybrid"):
        ids = [c.chunk_id for c, _ in idx.search(q, k=20, mode=mode)]
        r = next(
            (i + 1 for i, h in enumerate(ids) if h.startswith(want)), None
        )
        row.append(r)
    results.append({"query": q, "ranks": row})
    print(
        f"{q:<38}"
        + "".join(f"{(r or '-'):>{w}}" for r, w in zip(row, (6, 8, 8)))
    )

# Vectors blur codes: two different real-looking MPNs look "similar".
v = NgramVectorizer(["STM32F407VGT6", "STM32F407VET6", "GRM188R71H104KA93D"])
m = v.matrix
print("\ncosine(STM32F407VGT6, STM32F407VET6) =", round(float(m[0] @ m[1]), 2))
print(
    "cosine(STM32F407VGT6, GRM188R71H104KA93D) =", round(float(m[0] @ m[2]), 2)
)
json.dump(results, open("search_ranks.json", "w"), indent=1)
