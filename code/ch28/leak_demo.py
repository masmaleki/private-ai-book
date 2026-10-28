"""leak_demo.py - what Reza did first: random split + a 'future' feature.
Shows how leakage makes test error look much better than reality. example data.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from features import build_rows, FEATURES

offers = pd.read_csv("demo_offers.csv", parse_dates=["observed_at"])
parts = pd.read_csv("demo_parts.csv")
df = build_rows(offers, parts)

# LEAKY feature: median price of the same part within +-15 days of the
# target date (uses the future relative to as-of!).
o = offers.set_index("observed_at").sort_index()
def centered_median(r):
    g = o[o.part_id == r.part_id].loc[r.target_date - pd.Timedelta(days=15):
                                      r.target_date + pd.Timedelta(days=15)]
    return np.log(g.unit_price.median() / r.ref_price)
df["leaky_med_pm15"] = df.apply(centered_median, axis=1)

params = dict(n_estimators=400, learning_rate=0.03, num_leaves=15,
              min_child_samples=30, random_state=42, verbose=-1, n_jobs=2)

def wape(d, feats, train_mask):
    m = lgb.LGBMRegressor(objective="quantile", alpha=0.5, **params)
    m.fit(d.loc[train_mask, feats], d.loc[train_mask, "y"])
    t = d[~train_mask]
    pred = t.ref_price * np.exp(m.predict(t[feats]))
    return np.abs(t.y_price - pred).sum() / t.y_price.sum()

rng = np.random.default_rng(42)
rand_mask = pd.Series(rng.random(len(df)) < 0.8, index=df.index)
time_mask = df.target_date < pd.Timestamp("2026-03-01")
honest = df.asof_date >= pd.Timestamp("2026-03-01")
d_time = df[time_mask | honest]
print(f"A) random split + leaky feature : WAPE = "
      f"{wape(df, FEATURES + ['leaky_med_pm15'], rand_mask):.3f}")
print(f"B) random split, honest features: WAPE = "
      f"{wape(df, FEATURES, rand_mask):.3f}")
print(f"C) time split,  honest features : WAPE = "
      f"{wape(d_time, FEATURES, time_mask[d_time.index]):.3f}")
