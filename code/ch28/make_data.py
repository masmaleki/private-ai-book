"""make_data.py - synthetic offer history for the price chapter.
example data: every part price, supplier, lead time here is FICTIONAL.
Writes demo_offers.csv and demo_parts.csv (generic demo_* schema).
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
CATS = ["mcu", "power_ic", "mlcc", "discrete", "analog"]
PUBLIC_MPNS = ["STM32F407VGT6", "TPS5430DDAR", "LM2596S-ADJ",
               "GRM188R71H104KA93D", "ATMEGA328P-AU", "NE555DR", "BC847C,215"]
QTY_BREAKS = np.array([100, 250, 500, 1000, 2500, 5000, 10000])
DAYS = pd.date_range("2024-01-01", "2026-09-30", freq="D")
N_PARTS = 60
N_SUPPLIERS = 8

def market_index(cat_idx: int) -> np.ndarray:
    """Slow category-level market factor with a tight period in 2025."""
    t = np.arange(len(DAYS))
    center = 520 + 40 * cat_idx            # tight market peak (day index)
    bump = 0.25 * np.exp(-0.5 * ((t - center) / 70) ** 2)
    drift = 0.04 * np.sin(2 * np.pi * t / 365)
    return 1 + bump + drift                # multiplicative

parts, offers = [], []
cat_index = {c: market_index(i) for i, c in enumerate(CATS)}
base_price_by_cat = {"mcu": 6.0, "power_ic": 1.2, "mlcc": 0.012,
                     "discrete": 0.03, "analog": 0.6}
supplier_mult = rng.normal(1.0, 0.05, N_SUPPLIERS)
supplier_lead = rng.integers(10, 60, N_SUPPLIERS)
offer_id = 0
for p in range(N_PARTS):
    cat = CATS[p % len(CATS)]
    mpn = PUBLIC_MPNS[p] if p < len(PUBLIC_MPNS) else f"EXP-{cat[:3].upper()}-{p:03d}"
    lifecycle = rng.choice(["ACTIVE", "ACTIVE", "ACTIVE", "NRND"])
    base = base_price_by_cat[cat] * rng.lognormal(0, 0.5)
    elasticity = rng.uniform(0.04, 0.12)   # price falls with qty
    # part-specific random walk on log scale
    rw = np.cumsum(rng.normal(0, 0.004, len(DAYS)))
    rw -= rw[0]
    parts.append({"part_id": p, "manufacturer": f"MFR-{p % 9}", "mpn": mpn,
                  "mpn_norm": mpn.replace(",", "").upper(), "category": cat,
                  "package": "n/a", "lifecycle": lifecycle})
    n_obs = rng.integers(60, 220)          # some parts are sparse
    day_idx = np.sort(rng.integers(0, len(DAYS), n_obs))
    for d in day_idx:
        s = rng.integers(0, N_SUPPLIERS)
        qty = int(rng.choice(QTY_BREAKS))
        tight = cat_index[cat][d] - 1
        nrnd = 1.08 if lifecycle == "NRND" else 1.0
        price = (base * cat_index[cat][d] * np.exp(rw[d]) * nrnd
                 * (qty / 100) ** (-elasticity) * supplier_mult[s]
                 * rng.lognormal(0, 0.06))
        lead = max(1, int(supplier_lead[s] * (1 + 3 * tight)
                          + rng.normal(0, 6)))
        offers.append({"offer_id": offer_id, "tenant_id": "T-EXAMPLE",
                       "supplier_id": int(s), "part_id": p, "qty": qty,
                       "unit_price": round(price, 5), "currency": "EUR",
                       "lead_days": lead, "observed_at": DAYS[d]})
        offer_id += 1

pd.DataFrame(parts).to_csv("demo_parts.csv", index=False)
df = pd.DataFrame(offers)
df.to_csv("demo_offers.csv", index=False)
print(f"parts={len(parts)} offers={len(df)} "
      f"from {df.observed_at.min().date()} to {df.observed_at.max().date()}")
