"""pipeline.py - raw mailbox export -> clean, versioned SFT dataset (chapter 34).
Steps are counted so we can see where data is lost. FICTIONAL data only."""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import make_raw as R                     # only for the fictional CRM lists
from scrub import pseudonymise, facts_to_placeholders, residual_pii

CONTACTS = [n for _, _, _, cs in R.CUSTOMERS for n in cs]
COMPANIES = [c for _, c, _, _ in R.CUSTOMERS]
STAFF = ["Sarah"]
CONSENT = {"t-alpen", "t-danube", "t-brenner", "t-inntal", "t-moravia",
           "t-lago", "t-vltava"}           # t-nord opted out; Tenant-D never
SYSTEM_EXTRACT = ("Extract the RFQ. Reply with one line of JSON only: "
                  '{"mpn": str, "qty": int, "priority": "lead_time"|"price"|"unknown"}')
SYSTEM_REPLY = ("Write the reply to this RFQ in the customer's language, in our "
                "house style. Use the placeholders {{unit_price}}, {{lead_days}} and "
                "{{price_break_qty}} for facts; never write prices or dates yourself.")
DE = {"und", "für", "wir", "sie", "bitte", "vielen", "ihre", "stk.", "dank", "grüße",
      "der", "die", "das", "ist", "bei", "mit", "guten", "tag", "anfrage", "menge"}  # v2
EN = {"the", "for", "we", "please", "thank", "you", "your", "pcs", "regards", "of",
      "hello", "quantity", "need", "can", "could", "thanks", "in", "is", "are"}  # v2


def lang_of(text):
    w = re.findall(r"[a-zäöüß.]+", text.lower())
    de, en = sum(x in DE for x in w), sum(x in EN for x in w)
    if de == en:
        return "?"                       # v2: unsure -> do not decide
    return "de" if de > en else "en"


def norm(s):
    return re.sub(r"\s+", " ", s.lower()).strip()


def shingles(s, k=5):
    s = norm(s); return {s[i:i + k] for i in range(max(1, len(s) - k + 1))}


def jaccard(a, b):
    return len(a & b) / len(a | b)


def repair(s):
    if "Ã" in s or "Â" in s:
        try:
            return s.encode("latin-1").decode("utf-8"), True
        except (UnicodeEncodeError, UnicodeDecodeError):
            return s, False
    return s, False


def label_ok(r):
    lab, e = r["label"], r["email"].replace(",", "").replace(".", "")
    return (lab["mpn"] in r["email"] and re.search(rf"\b{lab['qty']}\b", e) is not None
            and lab["priority"] in {"lead_time", "price", "unknown"})


def split_of(thread_id):
    """Deterministic group split: a whole thread goes to ONE split."""
    h = int(hashlib.sha256(thread_id.encode()).hexdigest(), 16) % 100
    return "train" if h < 80 else "val" if h < 90 else "test"


def run(raw_path="raw_export.jsonl", out=Path("data")):
    rows = [json.loads(l) for l in open(raw_path, encoding="utf-8")]
    funnel = [("raw export", len(rows))]
    drop = Counter()

    def step(name, keep):
        nonlocal rows
        kept = [r for r in rows if keep(r)]
        drop[name] = len(rows) - len(kept); rows = kept
        funnel.append((name, len(rows)))

    step("defence tenant (must never be here)", lambda r: r["tenant_id"] != "t-defence")
    step("no consent / opted out", lambda r: r["tenant_id"] in CONSENT)
    step("not approved (draft)", lambda r: r["status"] == "sent")
    repaired = 0
    for r in rows:
        r["email"], fixed = repair(r["email"]); repaired += fixed
    step("empty or too short reply", lambda r: len(r["reply"].split()) >= 8)
    def same_lang(r):
        a, b = lang_of(r["reply"]), lang_of(r["email"])
        return a == b or "?" in (a, b)   # v2: only drop when both are known
    step("reply language != email language", same_lang)
    step("internal info in reply",
         lambda r: not re.search(r"internal|margin|Marge|cost \d", r["reply"], re.I))
    step("label does not match email", label_ok)
    seen = set()
    def first_time(r):
        h = hashlib.sha256(norm(r["email"] + r["reply"]).encode()).hexdigest()
        if h in seen:
            return False
        seen.add(h); return True
    step("exact duplicate", first_time)
    kept_sh, sims = [], []
    def not_near(r):
        sh = shingles(r["email"] + " " + r["reply"])
        best = max((jaccard(sh, o) for o in kept_sh), default=0.0)
        sims.append(best)
        if best >= 0.8:
            return False
        kept_sh.append(sh); return True
    step("near duplicate (Jaccard >= 0.8)", not_near)

    pii, residual = Counter(), 0
    for r in rows:
        r["email_s"] = pseudonymise(r["email"], CONTACTS, COMPANIES, STAFF, pii)
        r["reply_s"] = pseudonymise(r["reply"], CONTACTS, COMPANIES, STAFF, pii)
        r["reply_s"] = facts_to_placeholders(r["reply_s"], pii)
        residual += len(residual_pii(r["email_s"] + r["reply_s"], CONTACTS))
    seen.clear()
    def first_after_scrub(r):
        h = hashlib.sha256(norm(r["email_s"] + r["reply_s"]).encode()).hexdigest()
        if h in seen:
            return False
        seen.add(h); return True
    step("duplicate after pseudonymisation", first_after_scrub)

    for r in rows:
        r["split"] = split_of(r["thread_id"])
    files = {}
    out.mkdir(exist_ok=True)
    prov = open(out / "provenance.jsonl", "w", encoding="utf-8")
    for task in ("extract", "reply"):
        for sp in ("train", "val", "test"):
            path = out / f"{task}_{sp}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for r in rows:
                    if r["split"] != sp:
                        continue
                    sys_msg = SYSTEM_EXTRACT if task == "extract" else SYSTEM_REPLY
                    ans = json.dumps(r["label"]) if task == "extract" else r["reply_s"]
                    ex = {"prompt": [{"role": "system", "content": sys_msg},
                                     {"role": "user", "content": r["email_s"]}],
                          "completion": [{"role": "assistant", "content": ans}]}
                    line = json.dumps(ex, ensure_ascii=False)
                    f.write(line + "\n")
                    prov.write(json.dumps({                 # provenance, kept apart
                        "file": path.name, "sha256": hashlib.sha256(
                            line.encode()).hexdigest()[:16], "id": r["id"],
                        "thread_id": r["thread_id"], "lang": lang_of(r["email"]),
                        "source": r["source"]}) + "\n")
            files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    prov.close()
    stats = {"funnel": funnel, "dropped": dict(drop), "repaired_encoding": repaired,
             "pii_replaced": dict(pii), "residual_pii": residual,
             "near_dup_similarity": sims,
             "split_counts": dict(Counter(r["split"] for r in rows)),
             "coverage": {f"{lang_of(r['email'])}/{r['label']['priority']}": 0
                          for r in rows},
             "threads_per_split": {sp: len({r["thread_id"] for r in rows
                                            if r["split"] == sp})
                                   for sp in ("train", "val", "test")}}
    for r in rows:
        stats["coverage"][f"{lang_of(r['email'])}/{r['label']['priority']}"] += 1
    manifest = {"dataset": "rfq-extract-and-reply", "version": "v1",
                "created": "2026-09-14", "source": "mailbox:sales-shared (example export)",
                "filters": [n for n, _ in funnel[1:]], "files_sha256": files,
                "approvals": {"dpo": "pending", "sales_review": "pending"}}
    json.dump(manifest, open(out / "manifest.json", "w"), indent=1)
    json.dump(stats, open("pipeline_stats.json", "w"), indent=1)
    return rows, stats


if __name__ == "__main__":
    rows, st = run()
    for name, n in st["funnel"]:
        print(f"{n:5d}  {name}")
    print("repaired encoding:", st["repaired_encoding"])
    print("pii replaced:", st["pii_replaced"], "residual:", st["residual_pii"])
    print("split:", st["split_counts"], "threads:", st["threads_per_split"])
    print("coverage:", st["coverage"])
