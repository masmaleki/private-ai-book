"""Generate small synthetic demo_* tables for chapter 7 (all data invented).

Tables written to data/: demo_parts, demo_suppliers, demo_offers,
demo_quote_lines, demo_orders, plus fx_rates (USD per 1 EUR, daily).
"""

from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)
OUT = Path(__file__).parent / "data"
START, END = pd.Timestamp("2024-01-01"), pd.Timestamp("2025-12-31")
DAYS = pd.date_range(START, END, freq="D")
N_PARTS, N_OFFERS, N_QUOTES = 40, 9000, 2400


def fx_rates():
    # random walk around 1.08 USD per EUR
    steps = RNG.normal(0, 0.003, len(DAYS))
    usd_per_eur = 1.08 + np.cumsum(steps) - np.cumsum(steps).mean()
    return pd.DataFrame({"day": DAYS, "usd_per_eur": usd_per_eur.round(4)})


def parts():
    eol_day = np.where(
        RNG.random(N_PARTS) < 0.2, RNG.integers(300, 650, N_PARTS), 10_000
    )
    lifecycle = np.where(eol_day < 10_000, "eol", "active")  # state as of today
    return pd.DataFrame(
        {
            "part_id": np.arange(1, N_PARTS + 1),
            "manufacturer": RNG.choice([f"Mfr-{c}" for c in "ABCDE"], N_PARTS),
            "mpn": [f"EX-{1000 + i}" for i in range(N_PARTS)],
            "category": RNG.choice(["passive", "mcu", "power", "connector"], N_PARTS),
            "lifecycle": lifecycle,
            "base_eur": np.exp(RNG.normal(0, 1.4, N_PARTS)).round(4),
            "_eol_day": eol_day,  # hidden helper, dropped before saving
        }
    )


def suppliers():
    return pd.DataFrame(
        {
            "supplier_id": [1, 2, 3, 4, 5],
            "name": [f"Supplier-{c}" for c in "ABCDE"],
            "region": ["EU", "EU", "US", "US", "ASIA"],
            "authorized": [True, True, True, False, False],
        }
    )


def market_price(p, day_idx, qty):
    """True market price in EUR (unknown to models)."""
    drift = 1 + 0.06 * day_idx / 365 + 0.04 * np.sin(2 * np.pi * day_idx / 365)
    eol = np.where(day_idx > p["_eol_day"].to_numpy(), 1.35, 1.0)
    return p["base_eur"].to_numpy() * drift * eol * (1 - 0.05 * np.log10(qty))


def offers(pt, fx):
    idx = RNG.integers(0, N_PARTS, N_OFFERS)
    day = RNG.integers(0, len(DAYS), N_OFFERS)
    qty = (10 ** RNG.uniform(0, 4.5, N_OFFERS)).round().astype(int).clip(1)
    sup = RNG.integers(1, 6, N_OFFERS)
    eur = market_price(pt.iloc[idx], day, qty) * (1 + 0.03 * (sup - 3))
    eur *= np.exp(RNG.normal(0, 0.07, N_OFFERS))
    eur[RNG.random(N_OFFERS) < 0.004] *= 100  # rare data-entry outliers
    usd = sup >= 3  # US/ASIA suppliers quote in USD
    price = np.where(usd, eur * fx["usd_per_eur"].to_numpy()[day], eur)
    lead = RNG.gamma(4, 7, N_OFFERS).round().astype(int)
    lead = np.where(RNG.random(N_OFFERS) < 0.05, -1, lead)  # -1 = unknown
    ts = DAYS[day] + pd.to_timedelta(RNG.integers(8, 18, N_OFFERS), unit="h")
    df = pd.DataFrame(
        {
            "offer_id": np.arange(1, N_OFFERS + 1),
            "tenant_id": 7,
            "supplier_id": sup,
            "part_id": pt["part_id"].to_numpy()[idx],
            "qty": qty,
            "unit_price": price.round(4),
            "currency": np.where(usd, "USD", "EUR"),
            "lead_days": lead,
            "observed_at": ts,
        }
    )
    df.loc[df.lead_days < 0, "lead_days"] = np.nan
    df = df.sort_values("observed_at").reset_index(drop=True)
    df["offer_id"] = np.arange(1, len(df) + 1)  # ids follow time order
    return df


def quotes(pt):
    idx = RNG.integers(0, N_PARTS, N_QUOTES)
    day = RNG.integers(30, len(DAYS), N_QUOTES)
    qty = (10 ** RNG.uniform(1, 4.5, N_QUOTES)).round().astype(int)
    margin = RNG.uniform(0.12, 0.15, N_QUOTES)
    cost = market_price(pt.iloc[idx], day, qty) * np.exp(RNG.normal(0, 0.02, N_QUOTES))
    price = cost * (1 + margin)
    ts = DAYS[day] + pd.to_timedelta(RNG.integers(9, 17, N_QUOTES), unit="h")
    q = pd.DataFrame(
        {
            "quote_id": 5000 + np.arange(N_QUOTES),
            "tenant_id": 7,
            "customer_id": RNG.integers(1, 30, N_QUOTES),
            "part_id": pt["part_id"].to_numpy()[idx],
            "qty": qty,
            "unit_price": price.round(4),
            "margin_pct": (margin * 100).round(1),
            "issued_at": ts,
        }
    )
    # 10% of quotes are revisions: same customer/part/qty, re-issued 1-5 days later
    rev = q.sample(frac=0.10, random_state=1).copy()
    rev["quote_id"] = 9000 + np.arange(len(rev))
    rev["issued_at"] += pd.to_timedelta(RNG.integers(1, 6, len(rev)), unit="D")
    rev["unit_price"] = (rev["unit_price"] * RNG.uniform(0.98, 1.0, len(rev))).round(4)
    q = pd.concat([q, rev]).sort_values("issued_at").reset_index(drop=True)
    q.insert(0, "quote_line_id", np.arange(1, len(q) + 1))
    q["valid_until"] = q["issued_at"] + pd.Timedelta(days=14)
    return q


def orders(q):
    won = q[RNG.random(len(q)) < 0.35]
    return pd.DataFrame(
        {
            "order_id": np.arange(1, len(won) + 1),
            "quote_line_id": won["quote_line_id"].to_numpy(),
            "accepted_price": won["unit_price"].to_numpy(),
            "ordered_at": won["issued_at"].to_numpy()
            + pd.to_timedelta(RNG.integers(1, 14, len(won)), unit="D"),
        }
    )


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fx, pt = fx_rates(), parts()
    tables = {
        "fx_rates": fx,
        "demo_suppliers": suppliers(),
        "demo_offers": offers(pt, fx),
    }
    tables["demo_quote_lines"] = quotes(pt)
    tables["demo_orders"] = orders(tables["demo_quote_lines"])
    tables["demo_parts"] = pt.drop(columns=["base_eur", "_eol_day"])
    for name, df in tables.items():
        df.to_csv(OUT / f"{name}.csv", index=False)
        print(f"{name:<17} {len(df):>5} rows")
