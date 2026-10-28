"""Exercise C-3: a new read-only tool, registered with schema and roles."""

import statistics
from refproj import db
from refproj.tools import SESSIONS, OffersArgs, Tools


class MyTools(Tools):
    REGISTRY = dict(Tools.REGISTRY,
                    get_lead_time_stats=("get_lead_time_stats", OffersArgs,
                                         {"sales", "sales_lead"}))

    def get_lead_time_stats(self, s, a: OffersArgs):
        rows = self.db.execute(
            """SELECT o.lead_days FROM demo_offers o
               JOIN demo_parts p USING(part_id)
               WHERE o.tenant_id = ? AND p.mpn_norm = ?""",
            (s["tenant"], a.mpn.upper())).fetchall()  # tenant from session
        days = sorted(r[0] for r in rows)
        if not days:
            return None
        return dict(n=len(days), median=statistics.median(days),
                    p90=days[int(0.9 * (len(days) - 1))])


if __name__ == "__main__":
    t = MyTools(db.connect())
    for tok in ("tok-sales", "tok-ns", "tok-reza"):
        print(tok, t.call(SESSIONS[tok], "get_lead_time_stats",
                          {"mpn": "STM32F407VGT6"}))
