"""Exercise C-7: an expired draft can no longer be approved."""

from datetime import date
from refproj import db
from refproj.tools import SESSIONS, Tools, approve_quote


def approve_if_valid(conn, s, quote_id, today=None):
    today = (today or date.today()).isoformat()
    row = conn.execute("SELECT valid_until, tenant_id FROM demo_quote_lines "
                       "WHERE quote_id=?", (quote_id,)).fetchone()
    if row and row["tenant_id"] == s["tenant"] and row["valid_until"] < today:
        return "denied_expired"  # checked first: no approval after expiry
    return approve_quote(conn, s, quote_id)


if __name__ == "__main__":
    conn = db.connect()
    t = Tools(conn)
    _, d = t.call(SESSIONS["tok-sales"], "create_quote_draft", dict(
        mpn="NE555DR", qty=1000, customer_id="C-1",
        unit_price=0.30, cost_price=0.20))
    late = date(2027, 1, 1)
    print(approve_if_valid(conn, SESSIONS["tok-sarah"], d["quote_id"], late))
    print(approve_if_valid(conn, SESSIONS["tok-sarah"], d["quote_id"]))
