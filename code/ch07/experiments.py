"""Chapter 7: honest features vs leaky features (synthetic demo_* data)."""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from features import build, load, offers_in_eur, qty_slope

CUTOFF = pd.Timestamp("2025-07-01")
BASE = [
    "log_med30",
    "log_last",
    "log10_qty",
    "trend",
    "month_sin",
    "month_cos",
    "med30_missing",
    "lead_med30",
]


def prepare(q):
    q = q.copy()
    q["log_med30"] = np.log(q["med30"])
    q["log_last"] = np.log(q["last_price"]).fillna(q["log_med30"])
    q["lead_med30"] = q["lead_med30"].fillna(q["lead_med30"].median())
    q["y"] = np.log(q["unit_price"])
    if "med30_centred" in q:
        q["log_med30_centred"] = np.log(q["med30_centred"]).fillna(q["log_med30"])
        q["log_med30_centred_prod"] = np.log(q["med30_centred_prod"]).fillna(
            q["log_med30"]
        )
    return q.dropna(subset=["log_med30"])


def mape(y_log, p_log):
    y, p = np.exp(y_log), np.exp(p_log)
    return float(np.mean(np.abs(p - y) / y) * 100)


def fit_eval(train, test, cols):
    m = LinearRegression().fit(train[cols], train["y"])
    return mape(test["y"], m.predict(test[cols]))


def main():
    t = load()
    # estimate the quantity slope on TRAINING offers only (no peeking)
    o = offers_in_eur(t["demo_offers"], t["fx_rates"])
    slope = qty_slope(o[o.observed_at < CUTOFF])
    print(f"slope from training offers only: {slope:.4f}")
    q = prepare(build(t, slope=slope, leaky=True))
    tr, te = q[q.issued_at < CUTOFF], q[q.issued_at >= CUTOFF]
    print(f"train rows={len(tr)}  test rows={len(te)}")

    # baseline: market median adjusted to the quote's quantity, no model
    def market(d):
        return d["log_med30"] + slope * (d["log10_qty"] - 3)

    markup = float(np.median(tr["y"] - market(tr)))  # typical margin, from train
    base = market(te) + markup
    print(f"{'baseline: med30 at qty':<36} MAPE={mape(te['y'], base):5.2f}%")
    rows = [
        ("A honest features (temporal split)", BASE),
        ("B + centred 30d median (future leak)", BASE + ["log_med30_centred"]),
        ("C + is_eol_now (join/snapshot leak)", BASE + ["is_eol_now"]),
    ]
    for name, cols in rows:
        print(f"{name:<36} MAPE={fit_eval(tr, te, cols):5.2f}%")
    # B in production: trained with the leaky feature, served with what really exists
    cols_b = BASE + ["log_med30_centred"]
    m = LinearRegression().fit(tr[cols_b], tr["y"])
    te_prod = te.assign(log_med30_centred=te["log_med30_centred_prod"])
    served = mape(te["y"], m.predict(te_prod[cols_b]))
    print(f"{'B served in production':<36} MAPE={served:5.2f}%")
    eol = te["lifecycle"] == "eol"
    print(
        f"EOL parts only ({eol.sum()} rows): A={fit_eval(tr, te[eol], BASE):.2f}%  "
        f"B backtest={fit_eval(tr, te[eol], cols_b):.2f}%"
    )

    # D: random split vs grouped split (revisions of the same quote are near-duplicates)
    key = (
        q["customer_id"].astype(str)
        + "-"
        + q["part_id"].astype(str)
        + "-"
        + q["qty"].astype(str)
    )
    a, b = train_test_split(q, test_size=0.25, random_state=0)
    dup = b.index.isin(q.index[key.isin(key[a.index])])
    print(f"random split: {dup.mean():.1%} of test rows have a twin in train")
    gi, ge = next(
        GroupShuffleSplit(test_size=0.25, random_state=0).split(q, groups=key)
    )
    # a flexible model (chapter 8) that can memorise ids
    ids = BASE + ["customer_id", "part_id"]

    def gbm(train, test):
        m = lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=2,
            verbose=-1,
        )
        m.fit(train[ids], train["y"])
        return mape(test["y"], m.predict(test[ids]))

    print(f"{'D1 boosting, random split':<36} MAPE={gbm(a, b):5.2f}%")
    print(
        f"{'D2 boosting, grouped split':<36} MAPE={gbm(q.iloc[gi], q.iloc[ge]):5.2f}%"
    )
    print(f"{'D3 boosting, temporal split':<36} MAPE={gbm(tr, te):5.2f}%")

    # E: target leak, accepted_price exists only for won quotes, after the fact
    won = te["accepted_price"].notna()
    same = np.isclose(te.loc[won, "accepted_price"], te.loc[won, "unit_price"])
    print(
        f"accepted_price known for {won.mean():.0%} of test rows; "
        f"equals the target in {same.mean():.0%}"
    )


if __name__ == "__main__":
    main()
