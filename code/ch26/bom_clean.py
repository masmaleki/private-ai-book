"""bom_clean.py - clean a messy BOM: map columns, normalise, match, score,
and split into auto-accepted lines and a human review queue."""

import csv, sys
from bom_tools import (
    NORMALIZER_VERSION,
    load_reference,
    match_mpn,
    match_description,
    parse_qty,
    count_refdes,
    Match,
)
from column_mapper import map_columns

MFR_ALIASES = {
    "ST": "STMicroelectronics",
    "STMICRO": "STMicroelectronics",
    "TI": "Texas Instruments",
    "NXP": "Nexperia",  # example aliases
    "EXAMPLE SEMI": "Example Semiconductors",
}


def clean(path, ref="reference_parts.csv"):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        headers, rows = reader.fieldnames, list(reader)
    mapping, how, errors = map_columns(headers, rows)
    col = {v: k for k, v in mapping.items()}
    refs, by_key, collisions = load_reference(ref)
    assert not collisions, f"normaliser collision: {collisions}"
    out = []
    for i, r in enumerate(rows, 1):
        raw_mpn = r.get(col["mpn"], "").strip()
        m = (
            match_mpn(raw_mpn, refs, by_key)
            if raw_mpn
            else match_description(r.get(col["description"], ""), refs)
        )
        flags = []
        qty = parse_qty(r.get(col["qty"]))
        n_ref = count_refdes(r.get(col["refdes"], ""))
        if qty is None:
            flags.append("qty unreadable")
        elif n_ref and qty != n_ref:
            flags.append(f"qty {qty} != {n_ref} designators")
        mfr_in = (r.get(col["manufacturer"]) or "").strip()
        if m.mpn and mfr_in:
            ref_mfr = by_key[__import__("bom_tools").mpn_key(m.mpn)][
                "manufacturer"
            ]
            mfr = MFR_ALIASES.get(mfr_in.upper(), mfr_in)
            if mfr.lower() != ref_mfr.lower():
                flags.append(f"manufacturer '{mfr_in}' vs '{ref_mfr}'")
        status = m.status
        if flags and status == "auto":
            status = "review"  # any flag sends it to a human
        out.append(
            {
                "line": i,
                "input": raw_mpn or "(none)",
                "mpn": m.mpn or "",
                "qty": qty,
                "status": status,
                "conf": m.confidence,
                "reason": m.reason,
                "flags": "; ".join(flags),
            }
        )
    return out, mapping, how, errors


if __name__ == "__main__":
    lines, mapping, how, errors = clean("messy_bom.csv")
    print("normaliser:", NORMALIZER_VERSION)
    print("column mapping:", mapping)
    print("mapping method:", how, errors or "")
    print(f"\n{'#':>2} {'input':<20}{'matched':<20}{'conf':>5}  status")
    for l in lines:
        print(
            f"{l['line']:>2} {l['input'][:19]:<20}{l['mpn'][:19]:<20}"
            f"{l['conf']:>5.2f}  {l['status']}"
        )
    print("\nreview queue (lowest confidence first):")
    for l in sorted(
        (l for l in lines if l["status"] != "auto"), key=lambda l: l["conf"]
    ):
        why = l["reason"] + (f" | {l['flags']}" if l["flags"] else "")
        print(f"  line {l['line']:>2}: {why}")
    counts = {
        s: sum(l["status"] == s for l in lines)
        for s in ("auto", "review", "unmatched")
    }
    print("\nsummary:", counts)
    with open("clean_bom.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(lines[0]))
        w.writeheader()
        w.writerows(lines)
