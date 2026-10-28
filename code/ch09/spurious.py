"""Chapter 9: why 'market signals' need out-of-sample proof (synthetic data).

We create 200 random 'indicators' that have NO relation to demand, pick the one
that best 'predicts' demand two months ahead in 2022-2024, then test it in 2025.
"""
import numpy as np

from demand import make

d = make()
total = d[d.kind == "smooth"].groupby("month").qty.sum().to_numpy(float)
growth = np.diff(np.log(total))                     # month-over-month change, 47 values
rng = np.random.default_rng(7)
signals = rng.normal(size=(200, 48)).cumsum(1)       # 200 random walks
sig_change = np.diff(signals, axis=1)                # their monthly changes
LAG = 2
x = sig_change[:, :-LAG]                             # signal at month t
y = growth[LAG:]                                     # demand change at t + 2
train, test = slice(0, 33), slice(33, None)          # 2022-2024 vs 2025


def corr(a, b):
    return np.corrcoef(a, b)[0, 1]


ins = np.array([corr(s[train], y[train]) for s in x])
best = np.argmax(np.abs(ins))
print(f"best of 200 random signals: #{best}")
print(f"in-sample correlation (2022-2024): {ins[best]:+.2f}")
print(f"out-of-sample correlation (2025):  {corr(x[best][test], y[test]):+.2f}")
print(f"signals with |in-sample corr| > 0.35: {(np.abs(ins) > 0.35).sum()} of 200")
oos = np.array([corr(s[test], y[test]) for s in x])
top = np.argsort(-np.abs(ins))[:10]
print("top-10 in-sample signals:")
print(f"  mean |corr| in={np.abs(ins[top]).mean():.2f}  "
      f"out={np.abs(oos[top]).mean():.2f}, same sign kept: "
      f"{(np.sign(ins[top]) == np.sign(oos[top])).sum()} of 10")
