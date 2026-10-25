"""Exercise 7-8: the vandalism test catches the leaky centred median."""
import pandas as pd

from features import build, load

T = pd.Timestamp("2025-03-01")
tables = load()
before = build(tables, slope=-0.05, leaky=True)
o = tables["demo_offers"]
o.loc[o["observed_at"] >= T, "unit_price"] *= 10
after = build(tables, slope=-0.05, leaky=True)
past = before["issued_at"] < T
for col in ["med30", "med30_centred"]:
    changed = (before.loc[past, col] != after.loc[past, col]).sum()
    print(f"{col:<14} past rows changed by future vandalism: {changed}")
