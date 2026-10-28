"""threshold_eval.py - how many lines can be auto-accepted, and how often are
they wrong? Synthetic corruptions of the reference list (fixed seed)."""

import json, random, string
from bom_tools import load_reference, match_mpn, LOOKALIKE

rnd = random.Random(7)
rows, by_key, _ = load_reference()
mpns = [r["mpn"] for r in rows]


def corrupt(m):
    op = rnd.choice(
        [
            "case",
            "space",
            "nodash",
            "lookalike",
            "delete",
            "swap",
            "subst",
            "suffix",
            "truncate",
        ]
    )
    s = list(m)
    i = rnd.randrange(1, len(s) - 1)
    if op == "case":
        return m.lower(), op
    if op == "space":
        return m[:i] + " " + m[i:], op
    if op == "nodash":
        return m.replace("-", ""), op
    if op == "lookalike":
        idx = [k for k, c in enumerate(s) if c in LOOKALIKE]
        if idx:
            k = rnd.choice(idx)
            s[k] = LOOKALIKE[s[k]]
        return "".join(s), op
    if op == "delete":
        del s[i]
        return "".join(s), op
    if op == "swap":
        s[i], s[i + 1] = s[i + 1], s[i]
        return "".join(s), op
    if op == "subst":
        s[i] = rnd.choice(string.ascii_uppercase + string.digits)
        return "".join(s), op
    if op == "suffix":
        return m + rnd.choice(["TR", "/TR", "-REEL"]), op
    return m[:-2], op  # truncate


cases = [(*corrupt(m), m) for m in rnd.choices(mpns, k=400)]
cases += [
    (
        "".join(rnd.choices(string.ascii_uppercase + string.digits, k=10)),
        "unknown",
        None,
    )
    for _ in range(60)
]
results = []
for raw, op, truth in cases:
    m = match_mpn(raw, rows, by_key)
    results.append(
        (m.confidence, m.mpn == truth if truth else m.mpn is None, op)
    )

curve = []
for t in [x / 100 for x in range(50, 100, 2)]:
    auto = [ok for c, ok, _ in results if c >= t]
    curve.append(
        {
            "t": t,
            "auto_rate": len(auto) / len(results),
            "auto_precision": sum(auto) / len(auto) if auto else 1.0,
        }
    )
for row in curve:
    if row["t"] in (0.6, 0.76, 0.9, 0.96):
        print(
            f"threshold {row['t']:.2f}: auto {row['auto_rate']:.1%}, "
            f"correct among auto {row['auto_precision']:.1%}"
        )
wrong = [(op, c) for c, ok, op in results if c >= 0.9 and not ok]
print("wrong auto-accepts at 0.90:", len(wrong), wrong[:5])
json.dump(curve, open("threshold_curve.json", "w"))
