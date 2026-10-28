"""heldout.py - 12 FICTIONAL emails written AFTER the lexicon was frozen.
Evaluates the baseline honestly on data it was not tuned on."""
from collections import Counter
import json
from extract import baseline_extract, scrub, EmailSignals

T = [
 ("Can you expedite the order? Our customer is pushing hard.", 2, "lead_time", "neu", False),
 ("Bitte senden Sie ein Angebot für 5k BC847C, kein Zeitdruck.", 0, "none", "neu", False),
 ("We are switching to another distributor unless the price drops.", 1, "price", "neg", True),
 ("Production halts Thursday without these regulators!!", 3, "lead_time", "neg", False),
 ("Quick question: is the AMS1117-3.3 RoHS compliant?", 1, "none", "neu", False),
 ("Danke für die pünktliche Lieferung, weiter so!", 0, "none", "pos", False),
 ("Budget is tight this year, any volume discount on 30k pcs?", 1, "price", "neu", False),
 ("Need the parts by 3 March latest, price must stay under 1.20 EUR.", 2, "both", "neu", False),
 ("Ihre Lieferzeit von 12 Wochen ist für uns inakzeptabel.", 1, "lead_time", "neg", False),
 ("Not urgent at all, just updating our BOM costing.", 0, "price", "neu", False),
 ("Mouser lists it cheaper. Why should we buy from you?", 1, "price", "neg", True),
 ("Wir brauchen dringend eine Alternative, der Hersteller hat abgekündigt.", 3, "none", "neu", False),
]
if __name__ == "__main__":
    res, conf, errs = Counter(), Counter(), []
    for i, (txt, u, s, tone, comp) in enumerate(T, 1):
        o = EmailSignals(**baseline_extract(scrub(txt)))
        gold = {"urgency": u, "sensitivity": s, "tone": tone, "competitor": comp}
        for k, v in gold.items():
            ok = getattr(o, k) == v
            res[k] += ok
            if not ok:
                errs.append((f"H{i}", k, v, getattr(o, k)))
        conf[(u, o.urgency)] += 1
    print("held-out accuracy:", {k: f"{v}/{len(T)}" for k, v in res.items()})
    print("urgency confusion (gold, pred):", dict(sorted(conf.items())))
    for e in errs:
        print("  ", e)
