import pytest
from refproj import evaluate
from refproj.llm import HttpLLM, MockLLM, check_answer, facts_prompt
from refproj.rag import Index


def test_injection_is_quarantined_and_acl_holds():
    ix = Index().add_folder()
    assert "DOC-007#1" in ix.quarantine
    rag = evaluate.rag_report(ix)
    assert rag["leaks"] == 0 and rag["hit_at_3"] >= 0.85


def test_guard_rejects_invented_numbers():
    facts = [("F1", "Offer is 2.5 EUR with 12 days lead time.")]
    assert check_answer("Offer is 2.5 EUR [F1]", facts) == []
    assert check_answer("Offer is 2.1 EUR [F1]", facts)
    assert check_answer("Offer is 2.5 EUR", facts)


def test_mock_llm_answer_passes_guard():
    facts = [("F1", "Cheapest offer is 1.7 EUR."), ("F2", "Margin is 18.")]
    answer = MockLLM().chat(facts_prompt("why?", facts))
    assert check_answer(answer, facts) == []


def test_public_llm_hosts_are_refused():
    with pytest.raises(ValueError):
        HttpLLM("https://api.public-llm.example.com/v1")


def test_acl_is_applied_before_ranking():
    # The best-scoring chunks belong to other tenants. With a post-filter,
    # Tenant-D would get fewer than k results; with a pre-filter it gets k.
    ix = Index().add_folder()
    s = dict(tenant="tenant-d", role="sales")
    hits = ix.search("AlpenBoard NordSensor lead time price", s, k=3)
    assert len(hits) == 3
    assert not any(h["id"][:7] in ("DOC-004", "DOC-005") for h in hits)
