"""test_tools.py - security properties of the tool server (no network)."""

import json, os

os.environ["AUDIT_LOG"] = "audit_test.jsonl"
import tool_server as ts

SARAH, NS, REZA = (ts.SESSIONS[k] for k in ("tok-sarah", "tok-ns", "tok-reza"))


def test_tenant_isolation():
    a = ts.call_tool(SARAH, "get_part_offers", {"mpn": "TPS5430DDAR"})[1]
    b = ts.call_tool(NS, "get_part_offers", {"mpn": "TPS5430DDAR"})[1]
    assert a and b and a != b  # each tenant sees only its own rows


def test_tenant_argument_rejected():
    st, out = ts.call_tool(
        SARAH,
        "get_part_offers",
        {"mpn": "TPS5430DDAR", "tenant_id": "nordsensor"},
    )
    assert st == "invalid_args"


def test_role_filter():
    assert (
        ts.handle(REZA, {"id": 1, "method": "tools/list"})["result"]["tools"]
        == []
    )
    assert (
        ts.call_tool(REZA, "get_inventory", {"mpn": "NE555DR"})[0]
        == "denied_role"
    )


def test_unknown_tool():
    assert ts.call_tool(SARAH, "delete_offer", {})[0] == "denied_unknown_tool"


def test_sql_injection_is_just_data():
    st, out = ts.call_tool(SARAH, "get_part_offers", {"mpn": "x' OR '1'='1"})
    assert st == "ok" and out == []


def test_audit_written():
    n = sum(1 for _ in open("audit_test.jsonl"))
    assert n >= 6


if __name__ == "__main__":
    if os.path.exists("audit_test.jsonl"):
        os.remove("audit_test.jsonl")
    ts.AUDIT = "audit_test.jsonl"
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
    os.remove("audit_test.jsonl")
