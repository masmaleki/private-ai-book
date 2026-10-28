"""تست‌های واحد برای معماری Agentic AI (مهارت‌ها، ریزعامل‌ها، بازبینی خودکار، تحقیق وب امن و حافظه‌ی ۴ لایه) — فصل ۳۴."""
from pathlib import Path
from agentic_system import (
    SkillRegistry,
    normalize_mpn,
    tool_compute_quote_line,
    SubagentRunner,
    OutboundQuerySanitizer,
    MultiTierAgentMemory,
    WebResearchSubagent,
    LeadAgentCoordinator,
    SIMULATED_WEB_INDEX,
)


def test_progressive_disclosure_saves_tokens():
    reg = SkillRegistry()
    active = reg.select_and_load_skills("لطفاً لیست قطعات BOM و قیمت RFQ را بررسی کن")
    names = {s.name for s in active}
    assert "bom-mpn-normalizer" in names
    assert "rfq-margin-governance" in names
    metrics = reg.compute_token_savings(active, total_enterprise_skills=30)
    assert metrics["progressive_tokens"] < metrics["monolithic_tokens"]
    assert metrics["saved_percent"] >= 65.0


def test_mpn_normalization_and_packaging_split():
    mpn, pkg = normalize_mpn("  stm32f407vgt6-tr ")
    assert mpn == "STM32F407VGT6"
    assert pkg == "TR"


def test_pricing_tool_margin_floor_detection():
    # تخفیف ۱۲٪ همراه با تخفیف حجمی ۵٪ باعث افت حاشیه‌ی سود زیر کف ۱۸٪ برای TPS5430DDAR می‌شود
    res_bad = tool_compute_quote_line("STM32F407VGT6", qty=1200, requested_discount_pct=12.0, customer_tier="Tier-A")
    assert res_bad["margin_ok"] is False

    # بدون تخفیف اضافی، حاشیه‌ی سود بالای ۱۸٪ است
    res_good = tool_compute_quote_line("STM32F407VGT6", qty=1200, requested_discount_pct=0.0, customer_tier="Tier-A")
    assert res_good["margin_ok"] is True


def test_outbound_query_sanitizer_blocks_private_leakage():
    raw_prompt = (
        "مشتری NordSensor در استعلام RFQ-2026-9941 با ایمیل purchasing@nordsensor.eu "
        "قیمت هدف 4.10 EUR را برای قطعه STM32N657X0H6Q-TR در سطح Tier-A درخواست کرده است."
    )
    sanitized = OutboundQuerySanitizer.sanitize(raw_prompt, "STM32N657X0H6Q-TR")
    assert sanitized.zero_leakage_verified is True
    assert "nordsensor" not in sanitized.sanitized_outbound_query.lower()
    assert "rfq-2026-9941" not in sanitized.sanitized_outbound_query.lower()
    assert "4.10" not in sanitized.sanitized_outbound_query
    assert sanitized.sanitized_outbound_query.startswith("STM32N657X0H6Q ")
    assert len(sanitized.redacted_entities) >= 4


def test_web_research_subagent_and_four_tier_memory_persistence(tmp_path: Path):
    memory = MultiTierAgentMemory(dossier_dir=tmp_path / "dossiers")
    raw_ctx = "NordSensor RFQ-2026-9941 target 8.50 EUR for STM32N657X0H6Q"
    sub_res = WebResearchSubagent.run_for_unknown_parts(
        missing_items=[{"mpn": "STM32N657X0H6Q", "requested_qty": 300}],
        raw_internal_rfq_context=raw_ctx,
        memory_store=memory,
        session_id="test-sess-42",
    )
    assert sub_res.status == "SUCCESS"
    receipt = sub_res.structured_output["memory_persistence_receipts"][0]

    # بررسی مسدودسازی منبع آلوده به Prompt Injection و تأیید ۲ منبع معتبر (st.com و mouser.com)
    assert receipt["cross_verified_sources"] == 2
    assert receipt["blocked_untrusted_sources"] == 1
    assert receipt["composite_trust_score"] >= 0.94

    # بررسی لایه‌ی ۱ (Working Memory)
    t1_key = receipt["tier1_working_memory_key"]
    assert t1_key in memory.tier1_working_session
    assert memory.tier1_working_session[t1_key]["specs"]["npu_performance"] == "0.6 TOPS"

    # بررسی لایه‌ی ۲ (Markdown Research Dossier روی دیسک)
    dossier_file = tmp_path / "dossiers" / "STM32N657X0H6Q_research_dossier.md"
    assert dossier_file.exists()
    dossier_text = dossier_file.read_text(encoding="utf-8")
    assert "STMicroelectronics" in dossier_text
    assert "st.com" in dossier_text

    # بررسی لایه‌ی ۳ (Vector / Hybrid RAG Index با متادیتای منشأ و TTL)
    assert len(memory.tier3_vector_index) == 2
    assert memory.tier3_vector_index[0]["metadata"]["ttl_days"] == 30
    assert memory.tier3_vector_index[0]["metadata"]["prompt_injection_scanned"] is True

    # بررسی لایه‌ی ۴ (SQL Staging Table در دیتابیس عملیاتی)
    assert len(memory.tier4_sql_staging_table) == 1
    row = memory.tier4_sql_staging_table[0]
    assert row["mpn"] == "STM32N657X0H6Q"
    assert row["promotion_status"] == "STAGED_READY_FOR_CATALOG_SYNC"


def test_coordinator_reflection_and_web_enrichment_end_to_end():
    coord = LeadAgentCoordinator()
    report = coord.execute_rfq_workflow(
        {
            "session_id": "rfq-2026-9941",
            "goal": (
                "لطفاً برای مشتری NordSensor لیست قطعات BOM و قیمت RFQ را با تخفیف ۱۲٪ "
                "و تحقیق وب برای قطعات جدید STM32N6 به فارسی آماده کن"
            ),
            "customer_tier": "Tier-A",
            "requested_discount_pct": 12.0,
            "bom_lines": [
                {"raw_mpn": "stm32f407vgt6-tr", "qty": 1200},
                {"raw_mpn": "TPS5430DDAR", "qty": 500},
                {"raw_mpn": "STM32N657X0H6Q-TR", "qty": 300},
            ],
        }
    )
    assert report.reflection_iterations >= 1
    assert report.final_quote["all_margins_ok"] is True
    assert report.coordinator_context_tokens < report.total_subagent_isolated_tokens
    assert report.web_research_and_memory_summary is not None
    assert report.web_research_and_memory_summary["researched_count"] == 1
