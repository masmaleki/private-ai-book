---
name: rfq-pricing-audit
description: Standard Operating Procedure (SOP) for auditing multi-line RFQ quotes, checking gross margin floors, volume tiers, and customer tier rules. Load only when pricing or quoting RFQs.
---
# RFQ Pricing & Margin Audit Skill

## When to Use
Load this skill whenever an agent or subagent computes, reviews, or approves unit pricing for a Request for Quotation (RFQ).

## Mandatory Rules
1. **Strict Organization Isolation**: Query only the calling organization's own product/offer tables (`organization_id = current_org_id`). Never mix prices across tenants.
2. **Minimum Gross Margin Floor**:
   - Tier A (Strategic Customer): minimum `14.0%` gross margin.
   - Tier B (Standard Customer): minimum `18.0%` gross margin.
   - Tier C (Spot / New Customer): minimum `22.0%` gross margin.
3. **Anomaly Flagging**: If the proposed unit price deviates by more than `25%` from the 90-day median winning quote (`p50`), flag the line item for human review.
4. **Structured Output**: Always emit JSON matching the `PricingAuditResult` schema with `mpn`, `landed_cost_eur`, `proposed_unit_price_eur`, `margin_pct`, and `requires_human_approval`.
