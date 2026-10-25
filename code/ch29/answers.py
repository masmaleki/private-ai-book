"""answers.py - runnable answers for exercises 28-5 and 28-6 (synthetic)."""
import numpy as np, pandas as pd

def stockout_prob_open(on_hand, daily_mean, lead_samples,
                       open_qty=0, open_eta=None, n=20000, seed=7):
    r = np.random.default_rng(seed)
    L = r.choice(lead_samples, size=n)
    if open_eta is None:
        return float(np.mean(r.poisson(daily_mean * L) > on_hand))
    d1 = r.poisson(daily_mean * np.minimum(L, open_eta))
    short_before = d1 > on_hand
    d2 = r.poisson(daily_mean * np.maximum(L - open_eta, 0))
    arrives = L > open_eta
    short_after = arrives & (d1 + d2 > on_hand + open_qty)
    return float(np.mean(short_before | short_after))

po = pd.read_csv("demo_po_history.csv")
s1 = po[(po.supplier_id == 1) & (po.promised_days == 28)].actual_days.values
print(stockout_prob_open(1200, 45, s1))
print(stockout_prob_open(1200, 45, s1, open_qty=500, open_eta=10))

train, test = po.iloc[:2400], po.iloc[2400:]
rate = train.groupby("supplier_id").late.mean()
p_sup = test.supplier_id.map(rate).values
print(round(np.mean((p_sup - test.late.values) ** 2), 4))
