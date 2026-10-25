"""Chapter 7: point-in-time feature builder for quote lines (synthetic demo_* data).

Rule: every feature for a quote issued at time t uses only rows observed BEFORE t.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
REF_QTY_LOG10 = 3.0  # normalise every offer to a 1000-piece equivalent
FORBIDDEN = {"accepted_price", "ordered_at", "lifecycle", "margin_pct"}


def load(data=DATA):
    t = {
        n: pd.read_csv(data / f"{n}.csv")
        for n in [
            "demo_offers",
            "demo_quote_lines",
            "demo_orders",
            "demo_parts",
            "fx_rates",
        ]
    }
    t["demo_offers"]["observed_at"] = pd.to_datetime(t["demo_offers"]["observed_at"])
    q = t["demo_quote_lines"]
    q["issued_at"] = pd.to_datetime(q["issued_at"])
    t["fx_rates"]["day"] = pd.to_datetime(t["fx_rates"]["day"])
    return t


def offers_in_eur(offers, fx):
    """Convert with the FX rate valid at observed_at (as-of join), not today's."""
    # stable order with a tie-breaker: many offers share the same timestamp
    o = offers.sort_values(["observed_at", "offer_id"], kind="stable")
    o = pd.merge_asof(
        o,
        fx.rename(columns={"day": "fx_day"}),
        left_on="observed_at",
        right_on="fx_day",
        direction="backward",
    )
    rate = np.where(o["currency"] == "USD", o["usd_per_eur"], 1.0)
    o["price_eur"] = o["unit_price"] / rate
    return o


def qty_slope(o):
    """Volume-discount slope in log space, estimated within parts (fixed effects)."""
    lp = np.log(o["price_eur"])
    lq = np.log10(o["qty"])
    lp_d = lp - lp.groupby(o["part_id"]).transform("mean")
    lq_d = lq - lq.groupby(o["part_id"]).transform("mean")
    return float((lp_d * lq_d).sum() / (lq_d**2).sum())


def normalise(o, slope):
    """Price as if the offer had been for 1000 pieces."""
    o = o.copy()
    o["norm_price"] = o["price_eur"] * np.exp(
        -slope * (np.log10(o["qty"]) - REF_QTY_LOG10)
    )
    return o


def drop_outliers(o, ratio=5.0):
    """Drop offers far from the part's median of its PREVIOUS 20 offers."""
    o = o.sort_values(["observed_at", "offer_id"], kind="stable").copy()
    prev_med = o.groupby("part_id")["norm_price"].transform(
        lambda s: s.rolling(20, min_periods=5).median().shift(1)
    )
    r = o["norm_price"] / prev_med
    bad = (r > ratio) | (r < 1 / ratio)
    o.attrs["n_outliers"] = int(bad.sum())
    return o[~bad]


def window_stats(o, q, days, col="norm_price", stat="median", closed_future=False):
    """Stat of `col` over offers in [t - days, t) per part; strictly before t.

    closed_future=True builds the LEAKY centred window [t - days/2, t + days/2]
    used only to demonstrate leakage.
    """
    out = np.full(len(q), np.nan)
    groups = {p: g for p, g in o.groupby("part_id")}
    for i, (p, t) in enumerate(zip(q["part_id"].to_numpy(), q["issued_at"].to_numpy())):
        g = groups.get(p)
        if g is None:
            continue
        ts = g["observed_at"].to_numpy()
        if closed_future:
            lo, hi = t - np.timedelta64(days // 2, "D"), t + np.timedelta64(
                days // 2, "D"
            )
            sel = (ts >= lo) & (ts <= hi)
        else:
            sel = (ts >= t - np.timedelta64(days, "D")) & (ts < t)
        vals = g[col].to_numpy()[sel]
        vals = vals[~np.isnan(vals)]
        if len(vals):
            out[i] = np.median(vals) if stat == "median" else len(vals)
        elif stat == "count":
            out[i] = 0
    return out


def build(t, slope=None, leaky=False):
    o = offers_in_eur(t["demo_offers"], t["fx_rates"])
    slope = qty_slope(o) if slope is None else slope
    o = drop_outliers(normalise(o, slope))
    q = (
        t["demo_quote_lines"]
        .sort_values(["issued_at", "quote_line_id"], kind="stable")
        .copy()
    )
    # last offer for the same part strictly before the quote (as-of join)
    last = o[["part_id", "observed_at", "offer_id", "norm_price"]].rename(
        columns={"observed_at": "last_offer_at", "norm_price": "last_price"}
    )
    last = last.sort_values(["last_offer_at", "offer_id"], kind="stable")
    q = pd.merge_asof(
        q,
        last,
        left_on="issued_at",
        right_on="last_offer_at",
        by="part_id",
        direction="backward",
        allow_exact_matches=False,
    )
    q["med30"] = window_stats(o, q, 30)
    q["med90"] = window_stats(o, q, 90)
    q["n30"] = window_stats(o, q, 30, stat="count")
    q["lead_med30"] = window_stats(o, q, 30, col="lead_days")
    # missing values: fall back from 30 to 90 days, keep a flag for the model
    q["med30_missing"] = q["med30"].isna().astype(int)
    q["med30"] = q["med30"].fillna(q["med90"])
    q["trend"] = q["med30"] / q["med90"]
    q["log10_qty"] = np.log10(q["qty"])
    month = q["issued_at"].dt.month
    q["month_sin"], q["month_cos"] = np.sin(2 * np.pi * month / 12), np.cos(
        2 * np.pi * month / 12
    )
    q = q.merge(
        t["demo_parts"][["part_id", "category", "manufacturer", "lifecycle"]],
        on="part_id",
        how="left",
    )
    if leaky:
        q["med30_centred"] = window_stats(o, q, 30, closed_future=True)
        # what the same feature becomes in production: the future half is empty
        q["med30_centred_prod"] = window_stats(o, q, 15)
        q["is_eol_now"] = (q["lifecycle"] == "eol").astype(int)  # today's snapshot!
        q = q.merge(
            t["demo_orders"][["quote_line_id", "accepted_price"]],
            on="quote_line_id",
            how="left",
        )
    q.attrs["slope"] = slope
    q.attrs["n_outliers"] = o.attrs["n_outliers"]
    return q.reset_index(drop=True)


if __name__ == "__main__":
    tables = load()
    feats = build(tables)
    pd.set_option("display.width", 100)
    print(f"quantity slope (log price per 10x qty): {feats.attrs['slope']:.4f}")
    cols = [
        "quote_line_id",
        "part_id",
        "issued_at",
        "qty",
        "last_price",
        "med30",
        "n30",
        "trend",
        "unit_price",
    ]
    show = feats[cols].dropna().head(6).round(3)
    show = show.rename(columns={"quote_line_id": "line", "unit_price": "label"})
    show["issued_at"] = show["issued_at"].dt.strftime("%Y-%m-%d %H:%M")
    print(show.to_string(index=False))
    print("outlier offers dropped:", feats.attrs["n_outliers"])
    print(f"rows: {len(feats)}  med30 missing before fallback: "
          f"{int(feats['med30_missing'].sum())}  still missing: "
          f"{int(feats['med30'].isna().sum())}")
