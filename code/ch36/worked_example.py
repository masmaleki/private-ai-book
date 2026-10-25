"""worked_example.py - three raw pairs -> clean JSONL, and five bad samples.
All emails, names and numbers are FICTIONAL."""
import json
from pipeline import SYSTEM_REPLY, CONTACTS, COMPANIES, STAFF
from scrub import pseudonymise, facts_to_placeholders
import validate as V

pairs = [
    ("Hello,\nwe need 2,500 pcs STM32F407VGT6. Our line starts in three weeks, "
     "delivery time is critical.\n\nMarkus Lechner\nAlpenBoard Example GmbH\n"
     "Tel. +43 664 1234567",
     "Hello Markus,\n\nthank you for your request. For 2,500 pcs STM32F407VGT6 we can "
     "offer EUR 10.19 per piece, lead time 21 days. We can confirm the delivery date "
     "in writing today.\n\nBest regards,\nSarah"),
    ("Guten Tag,\nbitte um Angebot für 1,000 Stk. TPS5430DDAR. Wir vergleichen mehrere "
     "Angebote.\n\nLukas Hofer\nBrenner Elektronik Example GmbH",
     "Hallo Lukas,\n\nvielen Dank für Ihre Anfrage. Für 1,000 Stk. TPS5430DDAR bieten "
     "wir EUR 1.84 pro Stück an, Lieferzeit 14 Tage. Ab 2,000 Stk. wird der "
     "Stückpreis günstiger.\n\nViele Grüße\nSarah"),
    ("Hello,\nRFQ: NE555DR, quantity 10,000. Thanks in advance.\n\nPetra Novak\n"
     "petra@vltava.example",
     "Hello Petra,\n\nthank you for your request. For 10,000 pcs NE555DR we can offer "
     "EUR 0.09 per piece, lead time 7 days. Please let me know if you have any "
     "questions.\n\nBest regards,\nSarah"),
]
counts = {}
with open("worked_clean.jsonl", "w", encoding="utf-8") as f:
    for email, reply in pairs:
        e = pseudonymise(email, CONTACTS, COMPANIES, STAFF, counts)
        r = facts_to_placeholders(pseudonymise(reply, CONTACTS, COMPANIES, STAFF,
                                               counts), counts)
        ex = {"prompt": [{"role": "system", "content": SYSTEM_REPLY},
                         {"role": "user", "content": e}],
              "completion": [{"role": "assistant", "content": r}]}
        print("errors:", V.errors(ex, "reply")); print(r, "\n---")
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")
print("replaced:", counts)

S = "Extract the RFQ."
def ex(user, ans):
    return {"prompt": [{"role": "system", "content": S}, {"role": "user", "content": user}],
            "completion": [{"role": "assistant", "content": ans}]}
bad = [("reply", ex("please quote 500 pcs NE555DR.",
                    "Hello {{contact_first_name}}, for 500 pcs NE555DR we offer EUR 0.11 "
                    "per piece, lead time 7 days.")),
       ("reply", ex("we need 250 pcs LM2596S-ADJ.\nAnna Berger, Tel. +49 170 5550123",
                    "Hello Anna, thank you, the offer for 250 pcs follows today.")),
       ("extract", ex("we need 2,500 pcs TPS5430DDAR, price matters.",
                      '{"mpn": "TPS5430DDAR", "qty": 25000, "priority": "price"}')),
       ("extract", ex("RFQ: BC847C,215, quantity 1,000.",
                      'Sure! Here is the JSON: {"mpn": "BC847C,215", "qty": 1000}')),
       ("reply", ex("we need 100 pcs ATMEGA328P-AU, urgent.",
                    "Hello {{contact_first_name}}, no problem at all, we guarantee "
                    "delivery tomorrow, whatever happens. Best regards, {{sender_name}}"))]
for i, (task, b) in enumerate(bad, 1):
    print(f"bad sample {i} ({task}):", V.errors(b, task) or ["passes validator!"])
