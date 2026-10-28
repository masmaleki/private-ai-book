"""rfq_data.py - synthetic RFQ sentences (example data only, no real customers or prices)."""
import random

PARTS = ["STM32F407VGT6", "STM32F103C8T6", "TPS5430DDAR", "TPS5450DDAR", "LM2596S-5.0",
         "GRM188R71H104KA93D", "ATMEGA328P-AU", "CL10B104KB8NNNC"]
QTY = ["100", "250", "500", "1000", "2000", "5000"]
WEEKS = ["4", "6", "8", "10", "12"]
TEMPLATES = [
    "please quote {q} pcs {p}, lead time {w} weeks.",
    "we need {q} x {p} by week {w}.",
    "rfq: {p} qty {q}, max lead time {w} weeks.",
    "can you offer {q} pcs of {p}? delivery in {w} weeks.",
    "quote {p} for {q} units, {w} weeks or less.",
]


def sentence(rng):
    return rng.choice(TEMPLATES).format(q=rng.choice(QTY), p=rng.choice(PARTS), w=rng.choice(WEEKS))


def corpus(n=6000, seed=0):
    rng = random.Random(seed)
    return "\n".join(sentence(rng) for _ in range(n)) + "\n"
