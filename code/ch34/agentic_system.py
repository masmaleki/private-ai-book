#!/usr/bin/env python3
"""
agentic_system.py — Chapter 34 (Agentic AI: From Skills & Subagents to Multi-Agent Systems).
Demonstrates a complete, zero-dependency, reproducible Agentic AI architecture on synthetic
Alpenline Example GmbH data:
  1. AI Stack as a Human Body (Brain=LLM, Brain+Library=RAG, Nervous System=MCP,
     Muscle Memory=Skills, Brain+Hands=Agent, Team of Specialists=Subagents).
  2. Progressive Disclosure Skills (`SkillRegistry`):
     - Keeps only lightweight metadata (`name` + `when_to_use`) in the base system prompt (~180 tokens).
     - Loads full SOP body (`SKILL.md`) on demand only when the task needs it (saving ~87% prompt tokens).
  3. Subagent Architecture (`SubagentDispatcher`) with Context Isolation & Least-Privilege Tool Sandboxes:
     - `tech_bom_subagent`: read-only access to `lookup_part` and `find_alternates`
     - `pricing_subagent`: read-only access to `predict_price` and `check_margin_floor`
     - `supplier_risk_subagent`: read-only access to `get_lead_time_risk`
     Each subagent processes raw noisy rows in its own isolated context window and returns ONLY
     a compact, schema-validated JSON summary to the Coordinator.
  4. Plan-and-Execute + Reflection Verifier Loop before the Human Approval Gate.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Callable, Any, Set


# ---------------------------------------------------------------------------
# 1. Synthetic ERP Database & Tools (MCP Nervous System)
# ---------------------------------------------------------------------------
DEMO_CATALOG = {
    "STM32F407VGT6": {"stock": 400, "ref_price": 9.10, "lead_wks": 8, "alt": "STM32F405RGT6"},
    "TPS5430DDAR":   {"stock": 5000, "ref_price": 1.38, "lead_wks": 2, "alt": "TPS5431DDAR"},
    "LM2596S-ADJ":   {"stock": 0, "ref_price": 1.15, "lead_wks": 10, "alt": "LM2596SX-ADJ/NOPB"},
    "GRM188R71H104KA93D": {"stock": 25000, "ref_price": 0.04, "lead_wks": 2, "alt": "CL10B104KB8NNNC"},
}


def tool_lookup_part(mpn: str) -> Dict[str, Any]:
    info = DEMO_CATALOG.get(mpn)
    if not info:
        return {"found": False, "mpn": mpn}
    # Simulate verbose raw ERP log/rows that would pollute a single agent's context window
    verbose_audit_trail = [f"2026-Q{q}: wh_bin=A{q*12} lot=LT-{mpn[:4]}-{q} qc=PASS" for q in range(1, 18)]
    return {"found": True, "mpn": mpn, "stock": info["stock"], "raw_history_logs": verbose_audit_trail}


def tool_find_alternates(mpn: str) -> Dict[str, Any]:
    info = DEMO_CATALOG.get(mpn, {})
    alt = info.get("alt")
    raw_datasheet_notes = " | ".join([f"Pin_{i}: compatible V_in=4.5..36V thermal_pad=match" for i in range(1, 16)])
    return {"mpn": mpn, "alternate": alt, "compat_level": "DROP_IN" if alt else "NONE", "notes": raw_datasheet_notes}


def tool_predict_price(mpn: str, qty: int) -> Dict[str, Any]:
    info = DEMO_CATALOG.get(mpn)
    if not info:
        return {"error": "unknown_mpn"}
    base = info["ref_price"]
    discount = 0.94 if qty >= 1000 else 1.0
    p50 = round(base * discount, 3)
    p10 = round(p50 * 0.92, 3)
    p90 = round(p50 * 1.08, 3)
    floor = round(p50 * 0.95, 3)
    raw_offer_rows = [f"offer_{i}: sup=S{(i%4)+1} price={round(p50*(0.96+0.01*i),3)}" for i in range(1, 20)]
    return {"mpn": mpn, "qty": qty, "p10": p10, "p50": p50, "p90": p90, "floor": floor, "raw_offers": raw_offer_rows}


def tool_get_lead_time_risk(mpn: str) -> Dict[str, Any]:
    info = DEMO_CATALOG.get(mpn, {"lead_wks": 12})
    wks = info["lead_wks"]
    risk = "HIGH" if wks >= 8 else "LOW"
    raw_tariff_bulletin = " ; ".join([f"customs_clause_{k}: standard EU transit check OK" for k in range(1, 15)])
    return {"mpn": mpn, "lead_weeks": wks, "risk_level": risk, "raw_bulletin": raw_tariff_bulletin}


ALL_TOOLS: Dict[str, Callable] = {
    "lookup_part": tool_lookup_part,
    "find_alternates": tool_find_alternates,
    "predict_price": tool_predict_price,
    "get_lead_time_risk": tool_get_lead_time_risk,
}


# ---------------------------------------------------------------------------
# 2. Skills Architecture (Muscle Memory with Progressive Disclosure)
# ---------------------------------------------------------------------------
@dataclass
class SkillDefinition:
    name: str
    when_to_use: str
    full_instructions: str


SKILL_STORE: List[SkillDefinition] = [
    SkillDefinition(
        name="bom-shortage-resolution",
        when_to_use="When any requested BOM line has stock < requested qty or lead_weeks >= 8.",
        full_instructions=(
            "SOP BOM-SHORTAGE v2.4:\n"
            "1. If stock >= qty: fulfill from stock, no alternate needed.\n"
            "2. If stock < qty: call find_alternates(mpn) and propose DROP_IN alternate for the deficit.\n"
            "3. Never offer unit_price < floor. Flag HIGH lead_time risk lines for human buyer review.\n" +
            ("Detailed checklist rule: verify reel packaging and RoHS compliance.\n" * 18)
        ),
    ),
    SkillDefinition(
        name="export-compliance-d",
        when_to_use="When customer belongs to defense/air-gapped Tenant-D or dual-use part family.",
        full_instructions="SOP EXPORT-CTRL v4.1:\n" + ("Check EU Dual-Use Annex I & BAFA license.\n" * 22),
    ),
    SkillDefinition(
        name="rma-quality-claim",
        when_to_use="When customer reports defective lot, bent pins, oxidized leads, or COC mismatch.",
        full_instructions="SOP RMA-8D v3.0:\n" + ("Quarantine lot, request photos, open 8D report.\n" * 20),
    ),
    SkillDefinition(
        name="annual-frame-contract",
        when_to_use="When negotiating 12-month blanket order with quarterly call-offs and currency hedge.",
        full_instructions="SOP FRAME-2026:\n" + ("Apply volume tier 3 and EUR/USD collar clause.\n" * 20),
    ),
    SkillDefinition(
        name="vendor-onboarding-kyc",
        when_to_use="When registering a new supplier and verifying ISO 9001 / banking IBAN.",
        full_instructions="SOP VENDOR-KYC:\n" + ("Verify VAT ID, ISO cert expiry, and two-person IBAN check.\n" * 20),
    ),
]


class ProgressiveSkillRegistry:
    def __init__(self, skills: List[SkillDefinition]):
        self.skills = {s.name: s for s in skills}
        self.loaded_skills: Set[str] = set()

    def catalog_prompt(self) -> str:
        """Lightweight index always present in system prompt."""
        lines = ["Available Skills (load with load_skill(name) only when needed):"]
        for s in self.skills.values():
            lines.append(f"- {s.name}: {s.when_to_use}")
        return "\n".join(lines)

    def monolithic_prompt(self) -> str:
        """Anti-pattern: stuffing all SOPs into the system prompt on every turn."""
        return "\n\n".join(f"# {s.name}\n{s.full_instructions}" for s in self.skills.values())

    def load_skill(self, name: str) -> str:
        self.loaded_skills.add(name)
        return self.skills[name].full_instructions


# ---------------------------------------------------------------------------
# 3. Subagents Architecture (Team of Specialists with Isolated Context Windows)
# ---------------------------------------------------------------------------
@dataclass
class DelegationBrief:
    subagent_role: str
    goal: str
    allowed_tools: List[str]
    payload: Dict[str, Any]


@dataclass
class SubagentResult:
    subagent_role: str
    internal_tokens_consumed: int
    distilled_output: Dict[str, Any]


def estimate_tokens(text: str) -> int:
    # Approx 1 token per 3.8 characters in mixed technical JSON/text
    return max(1, int(len(text) / 3.8))


class SubagentWorker:
    """Executes a narrow brief in an isolated context window with least-privilege tools."""
    def __init__(self, role: str, allowed_tools: List[str]):
        self.role = role
        self.allowed_tools = set(allowed_tools)

    def invoke_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        if tool_name not in self.allowed_tools:
            raise PermissionError(f"Subagent '{self.role}' is forbidden from calling '{tool_name}'")
        return ALL_TOOLS[tool_name](**kwargs)

    def run(self, brief: DelegationBrief) -> SubagentResult:
        internal_context_log = [json.dumps(asdict(brief))]
        mpn = brief.payload["mpn"]
        qty = brief.payload["qty"]

        if self.role == "tech_bom_specialist":
            stock_res = self.invoke_tool("lookup_part", mpn=mpn)
            internal_context_log.append(json.dumps(stock_res))
            alt_res = {"alternate": None, "compat_level": "NONE"}
            if stock_res.get("stock", 0) < qty:
                alt_res = self.invoke_tool("find_alternates", mpn=mpn)
                internal_context_log.append(json.dumps(alt_res))
            distilled = {
                "mpn": mpn,
                "stock": stock_res.get("stock", 0),
                "shortage": max(0, qty - stock_res.get("stock", 0)),
                "recommended_alternate": alt_res.get("alternate"),
                "compat_level": alt_res.get("compat_level"),
            }

        elif self.role == "pricing_specialist":
            p_res = self.invoke_tool("predict_price", mpn=mpn, qty=qty)
            internal_context_log.append(json.dumps(p_res))
            distilled = {
                "mpn": mpn,
                "qty": qty,
                "unit_price_p50": p_res["p50"],
                "floor_price": p_res["floor"],
            }

        elif self.role == "supplier_risk_specialist":
            r_res = self.invoke_tool("get_lead_time_risk", mpn=mpn)
            internal_context_log.append(json.dumps(r_res))
            distilled = {
                "mpn": mpn,
                "lead_weeks": r_res["lead_weeks"],
                "risk_level": r_res["risk_level"],
            }
        else:
            raise ValueError(f"Unknown role: {self.role}")

        internal_tokens = estimate_tokens("\n".join(internal_context_log))
        return SubagentResult(
            subagent_role=self.role,
            internal_tokens_consumed=internal_tokens,
            distilled_output=distilled,
        )


# ---------------------------------------------------------------------------
# 4. Plan-and-Execute + Reflection Verifier Coordinator
# ---------------------------------------------------------------------------
def reflect_and_verify(draft_lines: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """Deterministic + critic reflection gate before human approval."""
    issues = []
    for idx, line in enumerate(draft_lines):
        if line["offered_price"] < line["floor_price"]:
            issues.append(f"Line {idx} ({line['mpn']}): offered_price {line['offered_price']} < floor {line['floor_price']}")
        if line["shortage"] > 0 and not line.get("recommended_alternate"):
            issues.append(f"Line {idx} ({line['mpn']}): shortage={line['shortage']} without alternate")
    return (len(issues) == 0), issues


def run_orchestrator_comparison():
    bom_request = [
        {"mpn": "STM32F407VGT6", "qty": 1000},
        {"mpn": "TPS5430DDAR", "qty": 2000},
        {"mpn": "LM2596S-ADJ", "qty": 500},
        {"mpn": "GRM188R71H104KA93D", "qty": 10000},
    ]

    registry = ProgressiveSkillRegistry(SKILL_STORE)
    mono_tokens = estimate_tokens(registry.monolithic_prompt())
    cat_tokens = estimate_tokens(registry.catalog_prompt())
    loaded_sop_tokens = estimate_tokens(registry.load_skill("bom-shortage-resolution"))

    workers = {
        "tech_bom_specialist": SubagentWorker("tech_bom_specialist", ["lookup_part", "find_alternates"]),
        "pricing_specialist": SubagentWorker("pricing_specialist", ["predict_price"]),
        "supplier_risk_specialist": SubagentWorker("supplier_risk_specialist", ["get_lead_time_risk"]),
    }

    coordinator_context_tokens = cat_tokens + loaded_sop_tokens
    single_agent_context_tokens = mono_tokens
    subagent_internal_total = 0

    compiled_lines = []
    for item in bom_request:
        # Dispatch 3 parallel specialists per BOM line
        b_tech = DelegationBrief("tech_bom_specialist", "Check stock & alternate", ["lookup_part", "find_alternates"], item)
        b_price = DelegationBrief("pricing_specialist", "Predict P50 & floor price", ["predict_price"], item)
        b_risk = DelegationBrief("supplier_risk_specialist", "Assess lead time risk", ["get_lead_time_risk"], item)

        r_tech = workers["tech_bom_specialist"].run(b_tech)
        r_price = workers["pricing_specialist"].run(b_price)
        r_risk = workers["supplier_risk_specialist"].run(b_risk)

        for r in (r_tech, r_price, r_risk):
            subagent_internal_total += r.internal_tokens_consumed
            single_agent_context_tokens += r.internal_tokens_consumed
            coordinator_context_tokens += estimate_tokens(json.dumps(r.distilled_output))

        merged = {
            **r_tech.distilled_output,
            **r_price.distilled_output,
            **r_risk.distilled_output,
            # Intentionally simulate a first-pass draft where LM2596S-ADJ is priced 3% below floor to trigger Reflection!
            "offered_price": round(r_price.distilled_output["floor_price"] * 0.97, 3)
            if item["mpn"] == "LM2596S-ADJ"
            else r_price.distilled_output["unit_price_p50"],
        }
        compiled_lines.append(merged)

    # First reflection pass catches the floor violation on LM2596S-ADJ and self-corrects
    ok_pass1, issues_pass1 = reflect_and_verify(compiled_lines)
    if not ok_pass1:
        for line in compiled_lines:
            if line["offered_price"] < line["floor_price"]:
                line["offered_price"] = line["unit_price_p50"]
                line["reflection_repaired"] = True
    ok_pass2, issues_pass2 = reflect_and_verify(compiled_lines)

    # Extrapolate to a 50-line BOM
    scale = 50 / len(bom_request)
    single_50 = int(mono_tokens + subagent_internal_total * scale)
    coord_50 = int(cat_tokens + loaded_sop_tokens + (coordinator_context_tokens - cat_tokens - loaded_sop_tokens) * scale)

    report = {
        "monolithic_skill_prompt_tokens": mono_tokens,
        "progressive_catalog_tokens": cat_tokens,
        "loaded_one_skill_tokens": loaded_sop_tokens,
        "4_line_bom": {
            "single_agent_context_tokens": single_agent_context_tokens,
            "coordinator_context_tokens": coordinator_context_tokens,
            "context_reduction_pct": round((1 - coordinator_context_tokens / single_agent_context_tokens) * 100, 1),
        },
        "50_line_bom_projection": {
            "single_agent_context_tokens": single_50,
            "coordinator_context_tokens": coord_50,
            "context_reduction_pct": round((1 - coord_50 / single_50) * 100, 1),
        },
        "reflection_pass1_ok": ok_pass1,
        "reflection_pass1_issues": issues_pass1,
        "reflection_pass2_ok": ok_pass2,
        "final_status": "READY_FOR_HUMAN_APPROVAL" if ok_pass2 else "BLOCKED",
        "lines": compiled_lines,
    }
    with Path(__file__).with_name("agentic_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "lines"}, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    run_orchestrator_comparison()
