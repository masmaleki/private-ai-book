"""supply_risk.py - lead time as a distribution, supplier reliability,
late-delivery probability and stockout risk. ALL DATA SYNTHETIC (example).
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

rng = np.random.default_rng(42)

# ---------- 1) synthetic purchase-order history ----------
N_SUP = 6
sup_bias = np.array([-2, 0, -3, 4, -1, 9])        # days late on average
sup_spread = np.array([2, 4, 2, 6, 3, 10])       # variability
rows = []
for i in range(3000):
    s = int(rng.integers(0, N_SUP))
    promised = int(rng.choice([14, 21, 28, 42, 56, 84]))
    tight = rng.uniform(0, 1)                    # market tightness 0..1
    qty = int(rng.choice([500, 1000, 2500, 5000, 10000]))
    delay = (sup_bias[s] + sup_spread[s] * rng.standard_normal()
             + 6 * tight ** 2 * rng.exponential(1.0)
             + 0.0004 * qty * rng.random())
    actual = max(1, round(promised + delay))
    rows.append({"supplier_id": s, "promised_days": promised,
                 "tightness": round(tight, 3), "qty": qty,
                 "actual_days": actual})
po = pd.DataFrame(rows)
po["late"] = (po.actual_days > po.promised_days + 3).astype(int)  # >3d grace
po.to_csv("demo_po_history.csv", index=False)

# ---------- 2) lead time distribution for one supplier ----------
s1 = po[(po.supplier_id == 1) & (po.promised_days == 28)].actual_days
q = s1.quantile([0.5, 0.8, 0.9, 0.95]).round(1)
print("supplier 1, promised 28 days: n =", len(s1))
print("  P50/P80/P90/P95 actual days:", q.to_dict())
print(f"  share later than promised+3: {np.mean(s1 > 31):.3f}")

# ---------- 3) supplier reliability with Beta smoothing ----------
PRIOR_A, PRIOR_B = 8, 2        # prior: ~80% on time, worth 10 orders
g = po.groupby("supplier_id").agg(n=("late", "size"),
                                  late=("late", "sum"))
g["on_time"] = g.n - g.late
g["raw_rate"] = g.on_time / g.n
g["smoothed"] = (g.on_time + PRIOR_A) / (g.n + PRIOR_A + PRIOR_B)
print("\nsupplier reliability (on-time rate):")
print(g.round(3).to_string())

# ---------- 4) late-delivery probability model ----------
po_sorted = po.reset_index(drop=True)
train, test = po_sorted.iloc[:2400], po_sorted.iloc[2400:]
X_cols = ["supplier_id", "promised_days", "tightness", "qty"]
clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15,
                         random_state=42, verbose=-1, n_jobs=2)
clf.fit(train[X_cols], train.late, categorical_feature=["supplier_id"])
p = clf.predict_proba(test[X_cols])[:, 1]
base = np.full(len(test), train.late.mean())
def brier(y, p): return np.mean((p - y) ** 2)
print(f"\nlate-model Brier={brier(test.late, p):.4f} "
      f"vs base-rate Brier={brier(test.late, base):.4f}")
bins = pd.cut(p, [0, .2, .4, .6, .8, 1.0])
cal = pd.DataFrame({"bin": bins, "p": p, "y": test.late.values}) \
        .groupby("bin", observed=True).agg(mean_p=("p", "mean"),
                                           obs=("y", "mean"), n=("y", "size"))
print("calibration table:")
print(cal.round(3).to_string())

# ---------- 5) lead-time quantile model ----------
qm = {}
for a in (0.5, 0.9):
    m = lgb.LGBMRegressor(objective="quantile", alpha=a, n_estimators=300,
                          learning_rate=0.05, num_leaves=15, random_state=42,
                          verbose=-1, n_jobs=2)
    m.fit(train[X_cols], train.actual_days, categorical_feature=["supplier_id"])
    qm[a] = m
p90 = qm[0.9].predict(test[X_cols])
print(f"lead-time P90 coverage on test: {np.mean(test.actual_days <= p90):.3f}")
example = pd.DataFrame([{"supplier_id": 5, "promised_days": 28,
                         "tightness": 0.8, "qty": 10000},
                        {"supplier_id": 2, "promised_days": 28,
                         "tightness": 0.8, "qty": 10000}])
for a, m in qm.items():
    print(f"  P{int(a*100)} for supplier 5 vs 2 (promised 28d, tight):",
          np.round(m.predict(example), 1))
print("  P(late) supplier 5 vs 2:",
      np.round(clf.predict_proba(example)[:, 1], 3))

# ---------- 6) stockout probability by simulation ----------
def stockout_prob(on_hand, daily_mean, lead_samples, n=20000, seed=7):
    """P(demand during lead time > on_hand). Demand ~ Poisson per day."""
    r = np.random.default_rng(seed)
    L = r.choice(lead_samples, size=n)
    demand = r.poisson(daily_mean * L)
    return float(np.mean(demand > on_hand))

lead_samples = s1.values
for oh in (1200, 1500, 1800, 2100):
    print(f"on_hand={oh}: P(stockout before replenishment) = "
          f"{stockout_prob(oh, daily_mean=45, lead_samples=lead_samples):.3f}")
