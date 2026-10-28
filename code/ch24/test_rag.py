"""test_rag.py - small checks for the RAG toolkit (run: python3 test_rag.py)."""
from rag import load_docs, make_chunks, Index, tokenize, rrf, chunk_structural

DOCS = load_docs()
IDX = Index(make_chunks(DOCS, "structural"))

def test_ten_docs_with_metadata():
    assert len(DOCS) == 10 and all("doc_id" in d.meta for d in DOCS)

def test_tokenizer_keeps_mpn_variants():
    t = tokenize("EXS-BUCK36-SO8")
    assert {"exs-buck36-so8", "buck36", "exsbuck36so8"} <= set(t)

def test_exact_mpn_query_hits_right_doc():
    top = IDX.search("EXS-LDO500-SOT23 dropout", k=1, mode="bm25")[0][0]
    assert top.chunk_id.startswith("DS-0002")

def test_rrf_math():
    order, score = rrf([[1, 2, 3], [3, 1]], k=60)
    assert order[0] == 1 and abs(score[1] - (1/61 + 1/62)) < 1e-12

def test_structural_chunk_keeps_table_with_header():
    ds = [d for d in DOCS if d.meta["doc_id"] == "DS-0001"][0]
    c = [x for x in chunk_structural(ds) if "TA = 105" in x][0]
    assert "| Parameter |" in c and "EXS-BUCK36-SO8" in c

def test_allowed_filter_restricts_results():
    allowed = {i for i, c in enumerate(IDX.chunks) if c.meta["visibility"] == "public"}
    hits = IDX.search("target price", k=5, allowed=allowed)
    assert all(c.meta["visibility"] == "public" for c, _ in hits)

if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
