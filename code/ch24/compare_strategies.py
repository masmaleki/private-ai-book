"""compare_strategies.py - rank of the 105 degC row per chunking strategy."""

from rag import load_docs, make_chunks, Index, CHUNKERS

docs = load_docs()
q = "EXS-BUCK36-SO8 output current at 105 °C"
print(f"query: {q}")
for strategy in CHUNKERS:
    idx = Index(make_chunks(docs, strategy))
    ranks = []
    for mode in ("bm25", "vector", "hybrid"):
        hits = idx.search(q, k=10, mode=mode)
        ranks.append(
            next(
                (
                    i + 1
                    for i, (c, _) in enumerate(hits)
                    if "TA = 105" in c.text
                ),
                None,
            )
        )
    hit = [c for c in idx.chunks if "TA = 105" in c.text][0]
    print(
        f"{strategy:<10} chunks={len(idx.chunks):2d} rank bm25/vec/hyb={ranks} "
        f"mpn_in_chunk={'EXS-BUCK36' in hit.text}"
    )
