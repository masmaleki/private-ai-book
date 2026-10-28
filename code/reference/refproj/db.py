"""One place that opens the demo database (SQLite stands in for MariaDB)."""

import sqlite3
import pandas as pd
from .config import DB_PATH


def connect(path=DB_PATH):
    db = sqlite3.connect(path, check_same_thread=False)
    db.row_factory = sqlite3.Row
    return db


def offers_frame(db):
    sql = """SELECT o.*, p.mpn, p.category, s.authorized
             FROM demo_offers o JOIN demo_parts p USING(part_id)
             JOIN demo_suppliers s USING(supplier_id)"""
    df = pd.read_sql_query(sql, db)
    df["observed_at"] = pd.to_datetime(df["observed_at"])
    return df
