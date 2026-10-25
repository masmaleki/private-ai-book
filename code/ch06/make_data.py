"""Generate a synthetic quote history for chapter 6.

Everything here is invented: part ids, prices and dates are random.
Each row is one quote line: a part, a date, a quantity and a unit price.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(6)
OUT = Path(__file__).parent / "data"
N_PARTS = 60
N_ROWS = 3000
START = pd.Timestamp("2024-01-01")
DAYS = 730  # two years


def make_parts():
    # catalogue reference price per part, spread over several orders of magnitude
    ref = np.exp(RNG.normal(loc=0.0, scale=1.6, size=N_PARTS)).round(4)
    cats = RNG.choice(["passive", "mcu", "power", "connector"], size=N_PARTS,
                      p=[0.4, 0.2, 0.2, 0.2])
    return pd.DataFrame({"part_id": [f"P-{i:03d}" for i in range(N_PARTS)],
                         "category": cats, "ref_price": ref})


def make_quotes(parts):
    idx = RNG.integers(0, N_PARTS, size=N_ROWS)
    day = np.sort(RNG.integers(0, DAYS, size=N_ROWS))
    qty = (10 ** RNG.uniform(0, 4.5, size=N_ROWS)).round().astype(int).clip(1)
    ref = parts["ref_price"].to_numpy()[idx]
    # volume discount: about 6% cheaper per 10x quantity
    discount = 1 - 0.06 * np.log10(qty)
    # slow market drift: +8% per year, plus a step of +12% after day 550
    drift = 1 + 0.08 * day / 365 + 0.12 * (day > 550)
    noise = np.exp(RNG.normal(0, 0.08, size=N_ROWS))
    price = (ref * discount * drift * noise).round(4)
    return pd.DataFrame({"quote_date": START + pd.to_timedelta(day, unit="D"),
                         "part_id": parts["part_id"].to_numpy()[idx],
                         "quantity": qty, "unit_price_eur": price})


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    parts = make_parts()
    quotes = make_quotes(parts)
    parts.to_csv(OUT / "parts.csv", index=False)
    quotes.to_csv(OUT / "quotes.csv", index=False)
    print(f"parts: {len(parts)} rows, quotes: {len(quotes)} rows")
    print(quotes.head(5).to_string(index=False))
    print("price range:", quotes.unit_price_eur.min(), "..", quotes.unit_price_eur.max())
