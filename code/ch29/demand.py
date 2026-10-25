"""demand.py - 30/90-day demand forecasts with a rolling-origin backtest.
Three synthetic parts: smooth, seasonal, intermittent. example data.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
days = pd.date_range("2024-01-01", "2026-09-30", freq="D")
t = np.arange(len(days))
season = 1 + 0.35 * np.sin(2 * np.pi * (t - 60) / 365)
demand = {
    "smooth":       rng.poisson(40 * np.ones(len(t))),
    "seasonal":     rng.poisson(40 * season),
    # intermittent: few big orders
    "intermittent": rng.binomial(1, 0.04, len(t)) * rng.integers(200, 1500, len(t)),
}
D = pd.DataFrame(demand, index=days)
D.to_csv("demo_daily_demand.csv")

def forecasts(hist: pd.Series, h: int) -> dict:
    return {
        "last_h_days": hist.iloc[-h:].sum(),                  # naive
        "mean_90d_x_h": hist.iloc[-90:].mean() * h,           # moving avg
        "same_period_last_year": hist.iloc[-365:-365 + h].sum(),  # seasonal naive
    }

def backtest(series: pd.Series, h: int, origins: int = 12) -> pd.DataFrame:
    out = []
    last = len(series) - h
    for k in range(origins):
        o = last - k * 30                         # origin moves back 30 days
        hist, fut = series.iloc[:o], series.iloc[o:o + h]
        f = forecasts(hist, h)
        out.append({m: abs(v - fut.sum()) for m, v in f.items()}
                   | {"actual": fut.sum()})
    r = pd.DataFrame(out)
    return (r.drop(columns="actual").sum() / r.actual.sum()).round(3)  # WAPE

for h in (30, 90):
    res = pd.DataFrame({p: backtest(D[p], h) for p in D.columns})
    print(f"\nWAPE, horizon {h} days, 12 rolling origins:")
    print(res.to_string())

# Probabilistic 30-day demand for the intermittent part: bootstrap of
# historical 30-day windows (last 365 days)
s = D["intermittent"].iloc[-365:]
sums = np.array([s.iloc[i:i + 30].sum() for i in range(0, 336)])
print("\nintermittent part, 30-day demand quantiles from history:")
print({f"P{int(q*100)}": int(np.quantile(sums, q)) for q in (0.1, 0.5, 0.9)})
print("share of 30-day windows with zero demand:", round(np.mean(sums == 0), 3))
