"""Solution sketch for exercise 41-3: l-diversity on top of k-anonymity."""
from privacy_toolkit import QI, ROWS, k_of, generalise, suppress_small_groups

STATUS = ["active", "cancelled", "active",          # AT
          "cancelled", "cancelled", "cancelled",    # DE: all the same
          "active", "active"]                       # HU


def l_diversity(rows, keys, sensitive) -> int:
    groups: dict[tuple, set] = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), set()).add(r[sensitive])
    return min(len(v) for v in groups.values())


rows = [dict(r, status=s) for r, s in zip(ROWS, STATUS)]
gen = [dict(generalise(r), status=r["status"]) for r in rows]
kept = suppress_small_groups(gen, QI, k=2)
print("k =", k_of(kept, QI), "| l =", l_diversity(kept, QI, "status"))
