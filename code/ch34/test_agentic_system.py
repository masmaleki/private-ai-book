import pytest
from agentic_system import (
    ProgressiveSkillRegistry,
    SKILL_STORE,
    SubagentWorker,
    DelegationBrief,
    run_orchestrator_comparison,
)


def test_progressive_disclosure_saves_tokens():
    reg = ProgressiveSkillRegistry(SKILL_STORE)
    assert len(reg.catalog_prompt()) < len(reg.monolithic_prompt()) * 0.25


def test_subagent_least_privilege_sandbox():
    tech = SubagentWorker("tech_bom_specialist", ["lookup_part", "find_alternates"])
    with pytest.raises(PermissionError):
        tech.invoke_tool("predict_price", mpn="STM32F407VGT6", qty=100)


def test_context_isolation_and_reflection():
    rep = run_orchestrator_comparison()
    assert rep["50_line_bom_projection"]["context_reduction_pct"] >= 85.0
    assert rep["reflection_pass1_ok"] is False
    assert rep["reflection_pass2_ok"] is True
    assert rep["final_status"] == "READY_FOR_HUMAN_APPROVAL"
