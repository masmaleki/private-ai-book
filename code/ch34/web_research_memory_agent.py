"""
فصل ۳۴ — ماژول مستقل ایجنت جست‌وجوی وب و معماری چهارلایه‌ی ماندگاری دانش (Web Research & 4-Tier Memory Agent).
این فایل کلاس‌های پاک‌سازی کوئری برون‌مرزی، ریزعامل تحقیق وب، و ذخیره‌سازی چهارلایه (Working Memory، پرونده‌ی Markdown،
ایندکس برداری RAG با متادیتای منشأ/TTL، و جدول Staging دیتابیس SQL) را صادر و اجرا می‌کند.
"""
from __future__ import annotations

from agentic_system import (
    SIMULATED_WEB_INDEX,
    SanitizedSearchQuery,
    OutboundQuerySanitizer,
    MultiTierPersistenceReceipt,
    MultiTierAgentMemory,
    WebResearchSubagent,
)

__all__ = [
    "SIMULATED_WEB_INDEX",
    "SanitizedSearchQuery",
    "OutboundQuerySanitizer",
    "MultiTierPersistenceReceipt",
    "MultiTierAgentMemory",
    "WebResearchSubagent",
]

if __name__ == "__main__":
    memory = MultiTierAgentMemory()
    raw_ctx = (
        "Customer NordSensor (RFQ-2026-9941, Tier-A, target price 4.10 EUR, contact buyer@nordsensor.de) "
        "needs 300 pcs of STM32N657X0H6Q-TR with RoHS3 and PCN check."
    )
    res = WebResearchSubagent.run_for_unknown_parts(
        missing_items=[{"mpn": "STM32N657X0H6Q", "requested_qty": 300}],
        raw_internal_rfq_context=raw_ctx,
        memory_store=memory,
        session_id="demo-web-session-01",
    )
    receipt = res.structured_output["memory_persistence_receipts"][0]
    print("=== Web Research & 4-Tier Memory Persistence Demo ===")
    print("Sanitized Outbound Query:", receipt["sanitized_query"])
    print("Redacted Internal Entities:", receipt["redacted_entities"])
    print("Tier 1 (Working Memory Key):", receipt["tier1_working_memory_key"])
    print("Tier 2 (Markdown Dossier):", receipt["tier2_markdown_dossier_path"])
    print("Tier 3 (Vector RAG Chunk IDs):", receipt["tier3_vector_chunk_ids"])
    print("Tier 4 (SQL Staging Row & Status):", receipt["tier4_sql_staging_row_id"], receipt["tier4_promotion_status"])
