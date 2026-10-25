"""emails.py - FICTIONAL client emails (EN/DE) + human labels (example data).
Labels were assigned with the rubric in chapter 30 (urgency 0-3,
sensitivity: price / lead_time / both / none, tone: pos / neu / neg).
"""
EMAILS = [
 {"id": "E1", "customer": "AlpenBoard Example GmbH", "lang": "en",
  "text": "Hi Sarah, we need 2.500 pcs STM32F407VGT6 for a run starting in 3 "
          "weeks, delivery is critical. Last time you were late. Can you "
          "confirm a date? Markus, +43 660 1234567",
  "label": {"urgency": 3, "sensitivity": "lead_time", "tone": "neg",
            "competitor": False}},
 {"id": "E2", "customer": "NordSensor Example AG", "lang": "de",
  "text": "Hallo Frau Sarah, Ihr Angebot für 50k GRM188R71H104KA93D ist leider "
          "zu teuer. Ein anderer Lieferant bietet 8% günstiger an. Unser "
          "Zielpreis liegt bei 0,0081 EUR. Mit freundlichen Grüßen, J. Weber",
  "label": {"urgency": 1, "sensitivity": "price", "tone": "neu",
            "competitor": True}},
 {"id": "E3", "customer": "Danube Components Example", "lang": "en",
  "text": "Hello, could you send us your stock and price for NE555DR, 10k pcs? "
          "No rush, planning for Q1. Thanks, Peter (peter@danube-example.hu)",
  "label": {"urgency": 0, "sensitivity": "none", "tone": "neu",
            "competitor": False}},
 {"id": "E4", "customer": "AlpenBoard Example GmbH", "lang": "de",
  "text": "Dringend: unsere SMT-Linie steht ab Montag, wenn TPS5430DDAR nicht "
          "kommt. Preis ist zweitrangig, wir brauchen die Ware ASAP. Markus",
  "label": {"urgency": 3, "sensitivity": "lead_time", "tone": "neg",
            "competitor": False}},
 {"id": "E5", "customer": "NordSensor Example AG", "lang": "en",
  "text": "Thanks for the quick quote. Price is OK, but we need delivery by "
          "end of month and best price for 20k next quarter. Regards, J. Weber",
  "label": {"urgency": 2, "sensitivity": "both", "tone": "pos",
            "competitor": False}},
 {"id": "E6", "customer": "Danube Components Example", "lang": "en",
  "text": "Your competitor quoted LM2596S-ADJ 12% cheaper with the same lead "
          "time. Can you match? We decide by Friday.",
  "label": {"urgency": 2, "sensitivity": "price", "tone": "neu",
            "competitor": True}},
 {"id": "E7", "customer": "AlpenBoard Example GmbH", "lang": "en",
  "text": "Great job on the last delivery, everything arrived on time. Please "
          "quote ATMEGA328P-AU 1k pcs for our next prototype batch.",
  "label": {"urgency": 0, "sensitivity": "none", "tone": "pos",
            "competitor": False}},
 {"id": "E8", "customer": "NordSensor Example AG", "lang": "de",
  "text": "Wir sind enttäuscht: die Lieferung kam 2 Wochen zu spät und der Preis "
          "war höher als vereinbart. Bitte um Rückruf bis morgen. J. Weber, "
          "Tel. +49 170 9876543",
  "label": {"urgency": 2, "sensitivity": "both", "tone": "neg",
            "competitor": False}},
]
