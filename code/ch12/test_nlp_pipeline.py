from nlp_pipeline import (
    normalize_persian_and_tech,
    extract_entities_rule_based,
    NaiveBayesIntentClassifier,
    TRAINING_SET,
    EVAL_EMAILS,
    route_email,
    levenshtein_distance,
    char_ngram_cosine,
)


def test_persian_normalization():
    raw = "سلام؛ قيمت هاي پيشنهادي براي ۲۵۰۰ عدد STM32F407VGT6–TR را می فرستيد؟"
    clean = normalize_persian_and_tech(raw)
    assert "ي" not in clean and "ك" not in clean
    assert "قیمت‌های" in clean
    assert "می‌فرستید" in clean
    assert "2500" in clean
    assert "STM32F407VGT6-TR" in clean


def test_entity_extraction():
    txt = "Need quote for 4,000 pcs TPS5430DDAR @ 1.38 EUR max, delivery 2 weeks."
    ent = extract_entities_rule_based(txt)
    assert ent.mpns == ["TPS5430DDAR"]
    assert ent.quantities == [4000]
    assert ent.target_prices_eur == [1.38]
    assert ent.lead_times == ["2 weeks"]


def test_hybrid_router_decisions():
    clf = NaiveBayesIntentClassifier(alpha=1.0)
    clf.fit(TRAINING_SET)
    for item in EVAL_EMAILS:
        res = route_email(item["text"], clf)
        assert res["route"] == item["expected_route"]


def test_similarity_metrics():
    assert levenshtein_distance("STM32F407VGT6", "STM32F4O7VGT6") == 1
    assert char_ngram_cosine("STM32F407VGT6", "STM32F4O7VGT6") > 0.70
