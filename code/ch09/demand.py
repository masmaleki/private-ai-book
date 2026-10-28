"""Chapter 9: synthetic monthly demand for 60 example parts (fictional data).

Three kinds of series: smooth (trend + seasonality), intermittent, lumpy.
A shared "market cycle" mimics a boom followed by a destocking dip.
"""
import numpy as np
import pandas as pd

MONTHS = pd.date_range("2022-01-01", "2025-12-01", freq="MS")   # 48 months
# European industrial pattern (illustrative): August and December are weak
SEASON = np.array([1.00, 1.05, 1.15, 1.05, 1.00, 1.00,
                   0.95, 0.70, 1.10, 1.10, 1.05, 0.85])


def market_cycle(n=len(MONTHS)):
    t = np.arange(n)
    # boom in 2022, destocking dip around mid-2023, recovery afterwards
    return 1 + 0.25 * np.exp(-((t - 3) / 6) ** 2) - 0.22 * np.exp(-((t - 18) / 5) ** 2)


def make(seed=42):
    rng = np.random.default_rng(seed)
    cyc = market_cycle()
    rows = []
    for i in range(60):
        part = f"EX-PART-{i + 1:03d}"
        t = np.arange(len(MONTHS))
        if i < 30:                                    # smooth
            kind = "smooth"
            level = rng.lognormal(np.log(600), 0.6)
            growth = rng.normal(0.006, 0.008)
            amp = rng.uniform(0.6, 1.2)               # strength of the seasonal pattern
            mu = level * (1 + growth) ** t * (1 + amp * (SEASON[t % 12] - 1)) * cyc
            y = rng.negative_binomial(20, 20 / (20 + mu))
        elif i < 50:                                  # intermittent
            kind = "intermittent"
            p = rng.uniform(0.15, 0.45)
            size = rng.lognormal(np.log(rng.uniform(100, 800)), 0.3, len(t))
            y = np.where(rng.random(len(t)) < p * cyc.clip(0.5, 1.5), size, 0).round()
        else:                                         # lumpy
            kind = "lumpy"
            p = rng.uniform(0.10, 0.30)
            size = rng.lognormal(np.log(rng.uniform(200, 2000)), 0.9, len(t))
            y = np.where(rng.random(len(t)) < p, size, 0).round()
        rows.append(pd.DataFrame({"part": part, "kind": kind, "month": MONTHS,
                                  "qty": y.astype(int)}))
    return pd.concat(rows, ignore_index=True)


def classify(y):
    """Syntetos-Boylan style: ADI (avg interval between demands) and CV^2 of sizes."""
    nz = y[y > 0]
    if len(nz) < 2:
        return np.inf, np.nan, "lumpy"
    adi = len(y) / len(nz)
    cv2 = (nz.std(ddof=0) / nz.mean()) ** 2
    if adi < 1.32:
        label = "smooth" if cv2 < 0.49 else "erratic"
    else:
        label = "intermittent" if cv2 < 0.49 else "lumpy"
    return adi, cv2, label


if __name__ == "__main__":
    d = make()
    print(d.head(3).to_string(index=False))
    print(f"rows={len(d)}  parts={d.part.nunique()}  months={d.month.nunique()}")
    print(d.groupby("kind").qty.agg(["mean", lambda s: (s == 0).mean()])
          .rename(columns={"<lambda_0>": "zero_share"}).round(2).to_string())
    cls = d.groupby("part").qty.apply(lambda s: classify(s.to_numpy())[2])
    kind = d.groupby("part").kind.first()
    print(pd.crosstab(kind, cls).to_string())
