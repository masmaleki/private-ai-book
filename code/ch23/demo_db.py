"""demo_db.py - synthetic demo_* tables in SQLite (example data, fixed seed)."""

import sqlite3, random


def build(seed: int = 42) -> sqlite3.Connection:
    rnd = random.Random(seed)
    db = sqlite3.connect(":memory:", check_same_thread=False)
    db.executescript("""
    CREATE TABLE demo_parts (part_id INTEGER PRIMARY KEY, manufacturer TEXT,
        mpn TEXT, mpn_norm TEXT, category TEXT, lifecycle TEXT);
    CREATE TABLE demo_suppliers (supplier_id INTEGER PRIMARY KEY, name TEXT,
        region TEXT, authorized INTEGER);
    CREATE TABLE demo_offers (offer_id INTEGER PRIMARY KEY, tenant_id TEXT,
        supplier_id INTEGER, part_id INTEGER, qty INTEGER, unit_price REAL,
        currency TEXT, lead_days INTEGER, observed_at TEXT);
    CREATE TABLE demo_inventory (part_id INTEGER, tenant_id TEXT,
        warehouse_id TEXT, on_hand INTEGER, reserved INTEGER, snapshot_at TEXT);
    """)
    parts = [
        (1, "STMicroelectronics", "STM32F407VGT6", "MCU", "active"),
        (2, "Texas Instruments", "TPS5430DDAR", "DC-DC", "active"),
        (3, "Murata", "GRM188R71H104KA93D", "MLCC", "active"),
        (4, "Texas Instruments", "NE555DR", "Timer", "active"),
    ]
    db.executemany(
        "INSERT INTO demo_parts VALUES (?,?,?,?,?,?)",
        [(i, m, p, p.upper(), c, l) for i, m, p, c, l in parts],
    )
    db.executemany(
        "INSERT INTO demo_suppliers VALUES (?,?,?,?)",
        [
            (1, "Supplier A (example)", "EU", 1),
            (2, "Supplier B (example)", "EU", 1),
            (3, "Broker C (example)", "Asia", 0),
        ],
    )
    oid = 0
    for tenant in ("alpenboard", "nordsensor"):
        for part_id in range(1, 5):
            for sup in (1, 2, 3):
                oid += 1
                base = {1: 9.0, 2: 1.0, 3: 0.01, 4: 0.12}[part_id]
                db.execute(
                    "INSERT INTO demo_offers VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        oid,
                        tenant,
                        sup,
                        part_id,
                        rnd.choice([500, 2000, 4000, 12000]),
                        round(base * rnd.uniform(0.85, 1.25), 4),
                        "EUR",
                        rnd.choice([5, 21, 35, 98]),
                        "2026-10-16",
                    ),
                )
            db.execute(
                "INSERT INTO demo_inventory VALUES (?,?,?,?,?,?)",
                (
                    part_id,
                    tenant,
                    "WH-1",
                    rnd.randint(0, 3000),
                    rnd.randint(0, 500),
                    "2026-10-17",
                ),
            )
    db.commit()
    return db
