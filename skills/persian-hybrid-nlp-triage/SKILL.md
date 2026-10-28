---
name: persian-hybrid-nlp-triage
description: Three-layer hybrid NLP pipeline for Persian and bilingual (Persian/English) commercial emails and tickets: Regex/normalization + Fast Embedding Classifier + Selective LLM escalation.
---
# Persian & Bilingual Hybrid NLP Triage Skill

## Architecture (The 3-Layer Funnel — Chapter 12)
1. **Layer 1 (Sub-millisecond CPU Rules & Hazm Normalization)**:
   - Normalize Persian Yeh/Kaf (`ی`/`ک`), preserve ZWNJ (`نیم‌فاصله`), convert Persian/Arabic digits to ASCII, and extract MPNs, quantities, and PO numbers via deterministic Regex.
2. **Layer 2 (Fast Encoder / Embedding Router — 5 to 15 ms)**:
   - Classify intent (`RFQ_QUOTE`, `ORDER_STATUS`, `INVOICE_PAYMENT`, `SPAM_OTHER`) using TF-IDF / ParsBERT / BGE-M3 embeddings + Logistic Regression.
   - If confidence `>= 0.78` and no complex ambiguity is detected, route directly without calling the LLM (`80%+` of traffic).
3. **Layer 3 (Private LLM Escalation — 15 to 20% of traffic)**:
   - Escalate only low-confidence, multi-intent, or high-value negotiation messages to the local LLM with schema-constrained JSON output.
