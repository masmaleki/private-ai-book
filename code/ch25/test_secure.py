"""test_secure.py - security and evaluation checks for chapter 24."""

from secure_rag import SecureRAG, can_read, check_answer, scan_chunks
from eval_retrieval import recall_at, mrr, ndcg_at, leak_test

RAG = SecureRAG()
AB = {"user": "u", "tenant": "alpenboard", "role": "sales", "staff": True}
EXT = {"user": "x", "tenant": "alpenboard", "role": "customer", "staff": False}


def test_acl_matrix():
    assert can_read({"visibility": "public"}, EXT)
    assert not can_read({"visibility": "internal", "roles": ["sales"]}, EXT)
    assert can_read({"visibility": "internal", "roles": ["sales"]}, AB)
    assert not can_read({"visibility": "tenant", "tenant": "nordsensor"}, AB)
    assert not can_read({"visibility": "secret"}, AB)  # unknown -> deny


def test_injection_quarantined():
    assert "SUP-C-07#1" in RAG.quarantine
    assert all(
        c.chunk_id != "SUP-C-07#1"
        for c, _ in RAG.retrieve("Is EXS-BUCK36-SO8 end-of-life?", AB, k=10)
    )


def test_no_cross_tenant_leak():
    open_leaks, secure_leaks = leak_test(RAG)
    assert open_leaks > 0 and secure_leaks == 0


def test_metrics_by_hand():
    ranked = ["a", "x", "b", "y", "z"]
    rel = {"a": 2, "b": 1, "c": 2}
    assert abs(recall_at(ranked, rel) - 2 / 3) < 1e-9
    assert mrr(["x", "a"], rel) == 0.5
    assert 0 < ndcg_at(ranked, rel) < 1


def test_answer_check():
    hits = RAG.retrieve("EXS-BUCK36-SO8 input voltage", AB)
    assert check_answer("Input voltage 4.5 V to 36 V [DS-0001#2]", hits) == []
    assert check_answer("Input voltage up to 99 V [DS-0001#2]", hits)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
