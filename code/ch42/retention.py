"""retention.py - propagate a deletion / retention rule to EVERY copy.

A person (data subject) asks for erasure, or a retention period ends.
Deleting the row in the ERP is not enough: copies live in the search index,
embedding store, fine-tuning datasets, eval sets, logs and adapters.
All stores below are in-memory fakes (example data).
"""
import datetime as dt
import json

TODAY = dt.date(2026, 10, 17)

STORES = {  # example data only
    "erp_contacts": [{"id": "c1", "email": "markus@alpenboard-example.at"},
                     {"id": "c2", "email": "purchasing@nordsensor-example.de"}],
    "search_index": [{"doc": "mail-881", "contact": "c1", "text": "..."},
                     {"doc": "mail-882", "contact": "c2", "text": "..."}],
    "vector_store": [{"chunk": "mail-881#0", "contact": "c1"},
                     {"chunk": "mail-881#1", "contact": "c1"},
                     {"chunk": "mail-882#0", "contact": "c2"}],
    "ft_dataset_v4": [{"sample": "s-17", "contact": "c1"},
                      {"sample": "s-18", "contact": "c2"}],
    "call_audio": [{"file": "call-0412.wav", "contact": "c1",
                    "recorded": "2026-06-01"},
                   {"file": "call-0977.wav", "contact": "c2",
                    "recorded": "2026-09-20"}],
}
ADAPTERS = {"sales-tone-v2": {"trained_on": "ft_dataset_v4"}}
RETENTION_DAYS = {"call_audio": 90}  # example policy, set by your DPO


def erase_contact(contact_id):
    report = {}
    for name, rows in STORES.items():
        key = "id" if name == "erp_contacts" else "contact"
        before = len(rows)
        STORES[name] = [r for r in rows if r.get(key) != contact_id]
        report[name] = before - len(STORES[name])
    # weights cannot "forget" one row: flag adapters for retraining
    stale = [a for a, meta in ADAPTERS.items()
             if report.get(meta["trained_on"], 0) > 0]
    return report, stale


def apply_retention():
    removed = {}
    for name, days in RETENTION_DAYS.items():
        keep = []
        for r in STORES[name]:
            age = (TODAY - dt.date.fromisoformat(r["recorded"])).days
            (keep if age <= days else removed.setdefault(name, [])).append(r)
        STORES[name] = keep
    return {k: [r["file"] for r in v] for k, v in removed.items()}


if __name__ == "__main__":
    print("retention ->", apply_retention())
    report, stale = erase_contact("c1")
    print("erasure c1 ->")
    for store, n in report.items():
        print(f"  {store:<14} rows removed: {n}")
    print("adapters to retrain without c1:", stale)
    print("audit entry:", json.dumps({
        "action": "erasure", "subject": "c1", "date": str(TODAY),
        "stores": sorted(k for k, v in report.items() if v),
        "follow_up": stale}, indent=2))
