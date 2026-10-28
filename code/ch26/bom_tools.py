"""bom_tools.py - MPN normalisation, fuzzy matching and BOM column mapping.

Deterministic rules first, fuzzy second, LLM last; every decision carries a
confidence and a reason so a human can review it. stdlib only.
"""

import csv, json, re, unicodedata
from dataclasses import dataclass

NORMALIZER_VERSION = "mpn-norm-v3"

# ---------- normalisation ----------
DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2212"), "-")
PACKAGING_SUFFIXES = ("/TR", "-TR", "TR", "-REEL", "/CT")  # example list


def normalize_display(raw: str) -> str:
    """Clean form for display: NFKC, unify dashes, uppercase, trim spaces."""
    s = unicodedata.normalize("NFKC", raw or "").translate(DASHES)
    return re.sub(r"\s+", " ", s).strip().upper()


def mpn_key(raw: str) -> str:
    """Matching key: only letters and digits. 'tps5430 dda-r' -> TPS5430DDAR."""
    return re.sub(r"[^0-9A-Z]", "", normalize_display(raw))


def strip_packaging(display: str):
    """Return (base, suffix) if a known packaging suffix ends the MPN."""
    for suf in PACKAGING_SUFFIXES:
        if display.endswith(suf) and len(display) > len(suf) + 4:
            return display[: -len(suf)], suf
    return display, None


# ---------- edit distance ----------
def levenshtein(a: str, b: str) -> int:
    """Classic dynamic programming: insert, delete, substitute cost 1."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(
                min(
                    prev[j] + 1,  # delete from a
                    cur[j - 1] + 1,  # insert into a
                    prev[j - 1] + (ca != cb),
                )
            )  # substitute
        prev = cur
    return prev[-1]


def damerau(a: str, b: str) -> int:
    """Optimal string alignment: Levenshtein + adjacent transposition."""
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = a[i - 1] != b[j - 1]
            d[i][j] = min(
                d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost
            )
            if (
                i > 1
                and j > 1
                and a[i - 1] == b[j - 2]
                and a[i - 2] == b[j - 1]
            ):
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[-1][-1]


# ---------- reference list ----------
def load_reference(path="reference_parts.csv"):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_key, collisions = {}, []
    for r in rows:
        k = mpn_key(r["mpn"])
        if k in by_key:
            collisions.append((by_key[k]["mpn"], r["mpn"]))
        by_key[k] = r
    return rows, by_key, collisions


# ---------- matcher ----------
@dataclass
class Match:
    status: str  # auto | review | unmatched
    mpn: str | None
    confidence: float
    reason: str


LOOKALIKE = {
    "O": "0",
    "0": "O",
    "I": "1",
    "1": "I",
    "S": "5",
    "5": "S",
    "B": "8",
    "8": "B",
}
AUTO, REVIEW = 0.90, 0.60  # thresholds; tune them on labelled data


def _decide(conf):
    return (
        "auto" if conf >= AUTO else "review" if conf >= REVIEW else "unmatched"
    )


def match_mpn(raw, rows, by_key):
    disp = normalize_display(raw)
    key = mpn_key(raw)
    if not key:
        return Match("unmatched", None, 0.0, "empty MPN")
    if key in by_key:  # 1) exact key
        same = disp == normalize_display(by_key[key]["mpn"])
        conf = 0.99 if same else 0.97
        return Match(
            "auto",
            by_key[key]["mpn"],
            conf,
            "exact" if same else "exact after normalisation",
        )
    base, suf = strip_packaging(disp)  # 2) packaging
    if suf and mpn_key(base) in by_key:
        return Match(
            "auto",
            by_key[mpn_key(base)]["mpn"],
            0.93,
            f"packaging suffix {suf} removed",
        )
    # 3) look-alike characters (O/0, I/1, S/5, B/8), one position at a time
    hits = {
        by_key[f]["mpn"]
        for i, ch in enumerate(key)
        if ch in LOOKALIKE
        for f in [key[:i] + LOOKALIKE[ch] + key[i + 1 :]]
        if f in by_key
    }
    if len(hits) == 1:
        return Match(
            "auto", hits.pop(), 0.91, "one look-alike character fixed"
        )
    if len(hits) > 1:
        return Match(
            "review",
            sorted(hits)[0],
            0.7,
            f"look-alike fix is ambiguous: {sorted(hits)}",
        )
    prefix = [r for k, r in by_key.items() if k.startswith(key)]
    if len(key) >= 5 and len(prefix) == 1:  # 4) unique prefix
        return Match(
            "review",
            prefix[0]["mpn"],
            0.75,
            "input is a prefix (ordering suffix missing)",
        )
    scored = sorted((damerau(key, k), r["mpn"]) for k, r in by_key.items())
    (d1, m1), (d2, _) = scored[0], scored[1]  # 5) fuzzy
    conf = max(0.0, 1 - d1 / max(len(key), 1)) - (0.15 if d2 - d1 <= 1 else 0)
    conf = round(min(max(conf, 0.0), 0.89), 2)  # fuzzy never auto
    return Match(
        _decide(conf),
        m1 if conf >= REVIEW else None,
        conf,
        f"fuzzy: distance {d1}, runner-up distance {d2}",
    )


# ---------- description fallback ----------
def parse_description(text):
    """Pull value and package from free text like '10k Widerstand 0603 1%'."""
    t = text.lower()
    value = re.search(
        r"\b(\d+(?:[.,]\d+)?\s*[kmr]?\d*)\s*(?:ohm|Ω|widerstand|resistor)?\b",
        t,
    )
    package = re.search(r"\b(0201|0402|0603|0805|1206)\b", t)
    kind = "Resistor" if re.search(r"widerstand|resistor|ohm|Ω", t) else None
    return {
        "kind": kind,
        "value": value.group(1).replace(" ", "") if value else None,
        "package": package.group(1) if package else None,
    }


def match_description(text, rows):
    a = parse_description(text)
    if not (a["kind"] and a["value"] and a["package"]):
        return Match("unmatched", None, 0.0, "no MPN and description unclear")
    cands = [
        r
        for r in rows
        if r["category"] == a["kind"]
        and r["package"] == a["package"]
        and r["value"].lower() == a["value"]
    ]
    if len(cands) == 1:
        return Match(
            "review",
            cands[0]["mpn"],
            0.65,
            f"from description: {a['kind']} {a['value']} {a['package']}",
        )
    return Match(
        "unmatched", None, 0.3, f"{len(cands)} candidates from description"
    )


# ---------- quantities and designators ----------
def parse_qty(raw):
    """'2.500' (German thousands) -> 2500, '1k' -> 1000, '10' -> 10."""
    s = (raw or "").strip().lower().replace(" ", "")
    if re.fullmatch(r"\d+(\.\d{3})+", s):
        return int(s.replace(".", ""))
    m = re.fullmatch(r"(\d+(?:[.,]\d+)?)k", s)
    if m:
        return int(round(float(m.group(1).replace(",", ".")) * 1000))
    return int(s) if s.isdigit() else None


def count_refdes(raw):
    """'U1-U10' -> 10, 'U24, U25' -> 2, 'C1-C1000' -> 1000."""
    total = 0
    for part in filter(
        None, (p.strip() for p in re.split(r"[,;]", raw or ""))
    ):
        m = re.fullmatch(r"([A-Z]+)(\d+)\s*-\s*\1?(\d+)", part, re.I)
        total += int(m.group(3)) - int(m.group(2)) + 1 if m else 1
    return total


# ---------- column mapping ----------
CANONICAL = ["mpn", "manufacturer", "qty", "description", "refdes"]
SYNONYMS = {
    "mpn": [
        "mpn",
        "mfr p/n",
        "mfr part number",
        "manufacturer part number",
        "hersteller-teilenr.",
        "herstellerteilenummer",
        "part number",
    ],
    "manufacturer": ["manufacturer", "mfr", "hersteller", "vendor"],
    "qty": ["qty", "quantity", "menge", "anzahl", "qty/board"],
    "description": ["description", "desc", "bezeichnung", "beschreibung"],
    "refdes": ["refdes", "reference", "designator", "ref des", "bestückung"],
}


def map_columns_rules(headers):
    mapping = {}
    for h in headers:
        hl = h.strip().lower()
        for canon, names in SYNONYMS.items():
            if hl in names and canon not in mapping.values():
                mapping[h] = canon
    return mapping


def validate_mapping(proposal, headers):
    """Accept an LLM proposal only if it is well-formed and consistent."""
    errors = []
    for h, canon in proposal.items():
        if h not in headers:
            errors.append(f"unknown column {h!r}")
        if canon not in CANONICAL:
            errors.append(f"unknown target {canon!r}")
    if len(set(proposal.values())) != len(proposal):
        errors.append("two columns mapped to the same target")
    return errors
