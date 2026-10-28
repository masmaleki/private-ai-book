"""Generate synthetic demo_* tables in SQLite. All data is example data.

Prices follow a simple, known recipe so we can check the model later:
base price x supplier markup x quantity discount x trend x noise."""

import sqlite3
import numpy as np
import pandas as pd
from .config import DATA, DB_PATH, SEED, TENANTS

PARTS = [  # manufacturer, mpn, category, package, base EUR (fictional)
    ("STMicroelectronics", "STM32F407VGT6", "mcu", "LQFP100", 9.80),
    ("Microchip", "ATMEGA328P-AU", "mcu", "TQFP32", 2.10),
    ("Example Semiconductors", "EXS-MCU407-LQFP100", "mcu", "LQFP100", 7.40),
    ("Texas Instruments", "TPS5430DDAR", "buck", "SO8-PowerPAD", 2.60),
    ("Texas Instruments", "LM2596S-ADJ", "buck", "TO263-5", 1.90),
    ("Example Semiconductors", "EXS-BUCK36-SO8", "buck", "SO8", 1.35),
    ("Example Semiconductors", "EXS-LDO500-33", "ldo", "SOT223", 0.42),
    ("Texas Instruments", "NE555DR", "timer", "SO8", 0.21),
    ("Murata", "GRM188R71H104KA93D", "cap", "0603", 0.012),
    ("Nexperia", "BC847C,215", "transistor", "SOT23", 0.025),
]
SUPPLIERS = [  # id, name, region, authorized, markup, mean lead days
    (1, "Supplier A (example)", "EU", 1, 1.00, 12),
    (2, "Supplier B (example)", "EU", 1, 0.94, 25),
    (3, "Broker C (example)", "Asia", 0, 1.18, 5),
    (4, "Supplier D (example)", "EU", 1, 1.05, 18),
]
QTYS = [10, 100, 1000, 5000, 20000]
SCHEMA = """
CREATE TABLE demo_parts (part_id INTEGER PRIMARY KEY, manufacturer TEXT,
    mpn TEXT, mpn_norm TEXT, category TEXT, package TEXT, lifecycle TEXT);
CREATE TABLE demo_suppliers (supplier_id INTEGER PRIMARY KEY, name TEXT,
    region TEXT, authorized INTEGER);
CREATE TABLE demo_offers (offer_id INTEGER PRIMARY KEY, tenant_id TEXT,
    supplier_id INTEGER, part_id INTEGER, qty INTEGER, unit_price REAL,
    currency TEXT, lead_days INTEGER, observed_at TEXT);
CREATE TABLE demo_quote_lines (quote_id TEXT PRIMARY KEY, tenant_id TEXT,
    customer_id TEXT, part_id INTEGER, qty INTEGER, unit_price REAL,
    margin_pct REAL, valid_until TEXT, status TEXT, created_by TEXT,
    approved_by TEXT, explanation TEXT, warnings TEXT, issued_at TEXT);
CREATE TABLE demo_feedback (prediction_id TEXT, model_version TEXT,
    accepted INTEGER, reason TEXT, created_at TEXT);
"""


def qty_discount(q):
    # Larger orders get lower unit prices; about -9% per decade of quantity.
    return 0.91 ** np.log10(q)


def make_offers(n=3000, seed=SEED):
    rng = np.random.default_rng(seed)
    days = pd.date_range("2025-01-01", "2026-06-30", freq="D")
    tenants = list(TENANTS)
    rows = []
    for i in range(n):
        pid = int(rng.integers(len(PARTS)))
        s = SUPPLIERS[rng.integers(len(SUPPLIERS))]
        q = int(rng.choice(QTYS))
        d = days[rng.integers(len(days))]
        months = (d.year - 2025) * 12 + d.month - 1
        trend = 1 + 0.006 * months  # slow upward drift, +0.6% per month
        noise = rng.lognormal(0, 0.08)
        price = PARTS[pid][4] * s[4] * qty_discount(q) * trend * noise
        lead = max(1, int(rng.normal(s[5], s[5] * 0.3)))
        tenant = tenants[rng.choice(3, p=[0.45, 0.45, 0.10])]
        rows.append((i + 1, tenant, s[0], pid + 1, q, round(price, 4),
                     "EUR", lead, d.date().isoformat()))
    return rows


def build_db(path=DB_PATH):
    DATA.mkdir(exist_ok=True)
    path.unlink(missing_ok=True)
    db = sqlite3.connect(path)
    db.executescript(SCHEMA)
    db.executemany(
        "INSERT INTO demo_parts VALUES (?,?,?,?,?,?,?)",
        [(i + 1, m, p, p.upper(), c, pk, "active")
         for i, (m, p, c, pk, _) in enumerate(PARTS)],
    )
    db.executemany("INSERT INTO demo_suppliers VALUES (?,?,?,?)",
                   [s[:4] for s in SUPPLIERS])
    offers = make_offers()
    db.executemany("INSERT INTO demo_offers VALUES (?,?,?,?,?,?,?,?,?)",
                   offers)
    db.commit()
    db.close()
    return len(PARTS), len(SUPPLIERS), len(offers)
