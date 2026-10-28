"""make_raw.py - simulate a RAW export of a sales mailbox (FICTIONAL data).

Each record: one customer RFQ email + the reply our sales person sent,
plus the labels a colleague added (mpn, qty, priority). The export is
deliberately messy, like real exports: duplicates, drafts, PII, internal
notes, a defence tenant that should never be there, mojibake, label typos.
Everything here is synthetic; all companies and people are fictional.
"""
import json
import random

rng = random.Random(34)

CUSTOMERS = [  # tenant_id, company, language, contacts
    ("t-alpen", "AlpenBoard Example GmbH", "de", ["Markus Lechner", "Eva Moser"]),
    ("t-nord", "NordSensor Example AG", "de", ["Anna Berger", "Martin Steiner"]),
    ("t-danube", "Danube Components Example", "en", ["Tamás Kovács"]),
    ("t-brenner", "Brenner Elektronik Example GmbH", "de", ["Lukas Hofer"]),
    ("t-inntal", "Inntal Controls Example", "de", ["Sophie Wagner"]),
    ("t-moravia", "Moravia Devices Example s.r.o.", "en", ["Jakub Dvořák", "Tomáš Král"]),
    ("t-lago", "Lago Automation Example S.r.l.", "en", ["Giulia Rossi", "Elena Bianchi"]),
    ("t-vltava", "Vltava EMS Example", "en", ["Petra Novak"]),
    ("t-defence", "Tenant-D", "en", ["Contact D"]),
]
MPNS = ["STM32F407VGT6", "TPS5430DDAR", "LM2596S-ADJ", "NE555DR", "ATMEGA328P-AU",
        "GRM188R71H104KA93D", "BC847C,215", "EXS-BUCK36-SO8"]
QTYS = [100, 250, 500, 1000, 2500, 5000, 10000, 25000]

ASK = {"en": ["we need {q} pcs of {m}.", "please quote {q} pcs {m}.",
              "could you offer {q} x {m}?", "RFQ: {m}, quantity {q}.",
              "we are looking for {q} units of {m}.", "can you supply {q} pcs {m}?"],
       "de": ["wir benötigen {q} Stk. {m}.", "bitte um Angebot für {q} Stk. {m}.",
              "können Sie {q} x {m} anbieten?", "Anfrage: {m}, Menge {q}.",
              "wir suchen {q} Stück {m}.", "bitte {q} Stk. {m} anbieten."]}
HINT = {("en", "lead_time"): ["Our line starts in three weeks, delivery time is critical.",
                              "Speed matters more than price here.",
                              "We need the parts as fast as possible.",
                              "Please tell us the earliest delivery date."],
        ("en", "price"): ["Price is the main factor for this order.",
                          "We compare several offers, please give your best price.",
                          "Budget is tight on this project.",
                          "A competitive unit price is important."],
        ("en", "unknown"): ["Thanks in advance.", "Looking forward to your offer.", ""],
        ("de", "lead_time"): ["Die Lieferzeit ist für uns entscheidend.",
                              "Unsere Fertigung wartet, bitte schnellstmöglich.",
                              "Bitte nennen Sie den frühesten Liefertermin."],
        ("de", "price"): ["Der Preis ist bei diesem Auftrag entscheidend.",
                          "Wir vergleichen mehrere Angebote.",
                          "Bitte Ihren besten Preis."],
        ("de", "unknown"): ["Vielen Dank im Voraus.", "Mit freundlichen Grüßen.", ""]}
REPLY = {"en": ("Hello {first},\n\nthank you for your request. For {q} pcs {m} we can offer "
                "EUR {price} per piece, lead time {lead} days. {extra}\n\n"
                "Best regards,\nSarah\nAlpenline Example GmbH | +43 662 000{ext}"),
         "de": ("Hallo {first},\n\nvielen Dank für Ihre Anfrage. Für {q} Stk. {m} bieten wir "
                "EUR {price} pro Stück an, Lieferzeit {lead} Tage. {extra}\n\n"
                "Viele Grüße\nSarah\nAlpenline Example GmbH | +43 662 000{ext}")}
EXTRA = {("en", "lead_time"): "We can confirm the delivery date in writing today.",
         ("en", "price"): "From {q2} pcs the unit price drops further.",
         ("en", "unknown"): "Please let me know if you have any questions.",
         ("de", "lead_time"): "Den Liefertermin bestätigen wir heute noch schriftlich.",
         ("de", "price"): "Ab {q2} Stk. wird der Stückpreis günstiger.",
         ("de", "unknown"): "Bei Fragen melden Sie sich gerne."}


def phone():
    return f"+43 {rng.randint(600, 699)} {rng.randint(1000000, 9999999)}"


def record(i, thread):
    tid, company, lang, contacts = rng.choices(CUSTOMERS, [12] * 8 + [2])[0]
    if rng.random() < 0.15:                     # some customers write in English
        lang = "en"
    name = rng.choice(contacts); first = name.split()[0]
    m, q = rng.choice(MPNS), rng.choice(QTYS)
    prio = rng.choices(["lead_time", "price", "unknown"], [0.45, 0.35, 0.20])[0]
    hello = {"en": "Hello,", "de": "Guten Tag,"}[lang]
    body = (f"{hello}\n{rng.choice(ASK[lang]).format(q=f'{q:,}', m=m)} "
            f"{rng.choice(HINT[(lang, prio)])}\n\n{name}\n{company}\n"
            f"Tel. {phone()} | {first.lower()}@{tid[2:]}.example")
    price = round(rng.uniform(0.02, 12.0), 2)
    reply = REPLY[lang].format(first=first, q=f"{q:,}", m=m, price=f"{price:.2f}",
                               lead=rng.choice([7, 14, 21, 35, 56]),
                               extra=EXTRA[(lang, prio)].format(q2=f"{q * 2:,}"),
                               ext=rng.randint(10, 99))
    return {"id": f"msg-{i:04d}", "thread_id": f"th-{thread:04d}", "tenant_id": tid,
            "company": company, "lang": lang, "status": "sent",
            "created_at": f"2026-{rng.randint(1, 8):02d}-{rng.randint(1, 28):02d}",
            "email": body, "reply": reply,
            "label": {"mpn": m, "qty": q, "priority": prio},
            "source": "mailbox:sales-shared (example export)"}


if __name__ == "__main__":
    rows, thread = [], 0
    while len(rows) < 595:
        thread += 1
        for _ in range(rng.choice([1, 1, 1, 2, 3])):  # threads with 1..3 RFQs
            rows.append(record(len(rows), thread))
    base = list(rows)
    def add(r):
        r = dict(r); r["id"] = f"msg-{len(rows):04d}"; rows.append(r); return r
    for r in rng.sample(base, 25):                  # exact duplicates (export twice)
        add(r)
    for r in rng.sample(base, 20):                  # near duplicates (forwarded)
        n = add(r); n["email"] = "FW: " + r["email"].replace("\n\n", "\n") + "\nSent from phone"
    for r in rng.sample(base, 15):                  # useless short replies
        r["reply"] = rng.choice(["OK, done.", "See attached.", "Thanks!", "ok"])
    for r in rng.sample(rows, 30):
        r["status"] = "draft"                       # never approved / never sent
    for r in rng.sample(rows, 8):                   # internal note leaked into reply
        r["reply"] += "\n(internal: cost 8.10, margin 18%, do not show)"
    for r in rng.sample(rows, 6):                   # mojibake from a broken export
        r["email"] = r["email"].encode("utf-8").decode("latin-1")
    for r in rng.sample(rows, 10):                  # label typo by a colleague
        r["label"] = dict(r["label"], qty=r["label"]["qty"] + 1)
    for r in rng.sample(rows, 9):                   # reply in the wrong language
        r["reply"] = REPLY["en" if r["lang"] == "de" else "de"].format(
            first="X", q="1", m=r["label"]["mpn"], price="1.00", lead=7, extra="", ext=10)
    for r in rng.sample(rows, 5):
        r["reply"] = ""                             # empty reply
    rng.shuffle(rows)
    with open("raw_export.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("raw records:", len(rows), "threads:", thread)
