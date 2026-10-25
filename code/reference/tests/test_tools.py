from refproj.tools import SESSIONS, approve_quote

SALES, SARAH, NS = (SESSIONS[t] for t in ("tok-sales", "tok-sarah", "tok-ns"))


def test_tenant_comes_from_session_not_arguments(tools):
    status, out = tools.call(SALES, "get_part_offers",
                             {"mpn": "NE555DR", "tenant": "nordsensor"})
    assert status == "invalid_args" and "tenant" in out[0]
    for key in ("Tenant", "tenant ", "tenant_id"):  # look-alike names too
        args = {"mpn": "NE555DR", key: "nordsensor"}
        assert tools.call(SALES, "get_part_offers", args)[0] == "invalid_args"


def test_unknown_and_approve_tools_are_not_callable(tools):
    assert tools.call(SALES, "approve_quote", {})[0] == "denied_unknown_tool"
    assert tools.call(SALES, "delete_part", {})[0] == "denied_unknown_tool"


def test_developer_role_cannot_read_offers(tools):
    status, _ = tools.call(SESSIONS["tok-reza"], "get_part_offers",
                           {"mpn": "NE555DR"})
    assert status == "denied_role"


def test_tenant_d_never_sees_broker_offers(tools):
    _, offers = tools.call(SESSIONS["tok-d"], "get_part_offers",
                           {"mpn": "STM32F407VGT6", "qty": 20000})
    assert offers and all(o["authorized"] for o in offers)


def test_draft_needs_a_different_human_lead(tools):
    args = dict(mpn="NE555DR", qty=1000, customer_id="C-1",
                unit_price=0.30, cost_price=0.20)
    _, d1 = tools.call(SALES, "create_quote_draft", args)
    assert d1["status"] == "pending_approval"
    assert approve_quote(tools.db, SALES, d1["quote_id"]) == "denied_role"
    assert approve_quote(tools.db, NS, d1["quote_id"]) == "not_found"
    assert approve_quote(tools.db, SARAH, d1["quote_id"]) == "ok"
    assert approve_quote(tools.db, SARAH, d1["quote_id"]) == "denied_state"
    _, d2 = tools.call(SARAH, "create_quote_draft", args)
    assert approve_quote(tools.db, SARAH, d2["quote_id"]) == "denied_self_approval"


def test_low_margin_is_flagged(tools):
    _, d = tools.call(SALES, "create_quote_draft", dict(
        mpn="NE555DR", qty=1000, customer_id="C-1",
        unit_price=0.21, cost_price=0.20))
    assert any("margin" in w for w in d["warnings"])
