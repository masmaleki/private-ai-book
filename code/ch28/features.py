"""features.py - point-in-time features for price prediction (teaching).
For each target offer observed at T we build features AS OF t = T - horizon,
using only offers with observed_at <= t (no future data = no leakage).
"""
import numpy as np
import pandas as pd

HORIZON_DAYS = 30

def _med(x):
    return float(np.median(x)) if len(x) else np.nan

def build_rows(offers: pd.DataFrame, parts: pd.DataFrame,
               horizon: int = HORIZON_DAYS) -> pd.DataFrame:
    rows = []
    offers = offers.sort_values("observed_at")
    for pid, g in offers.groupby("part_id"):
        t_obs = g.observed_at.values.astype("datetime64[D]")
        price = g.unit_price.values
        qty = g.qty.values
        lead = g.lead_days.values
        for i in range(len(g)):
            T = t_obs[i]
            asof = T - np.timedelta64(horizon, "D")
            hist = t_obs <= asof                       # point-in-time mask
            if hist.sum() == 0:
                continue                               # cold start: skip
            w90 = hist & (t_obs > asof - np.timedelta64(90, "D"))
            w30 = hist & (t_obs > asof - np.timedelta64(30, "D"))
            same_q = hist & (qty == qty[i])
            same_q90 = w90 & (qty == qty[i])
            last = np.where(hist)[0][-1]
            last_sq = np.where(same_q)[0][-1] if same_q.any() else None
            med90_all = _med(price[w90])
            ref = med90_all if not np.isnan(med90_all) else price[last]
            rows.append({
                "part_id": pid, "target_date": T, "asof_date": asof,
                "qty": qty[i], "log_qty": np.log(qty[i]),
                "y_price": price[i],
                "ref_price": ref,
                "last_price_any": price[last],
                "last_qty_any": qty[last],
                "last_price_same_qty": price[last_sq] if last_sq is not None
                else np.nan,
                "med90_same_qty": _med(price[same_q90]),
                "med90_all": med90_all,
                "med90_log_qty": _med(np.log(qty[w90])),
                "trend_30_90": np.log(_med(price[w30]) / med90_all)
                if w30.any() and w90.any() else np.nan,
                "n_offers_90": int(w90.sum()),
                "days_since_last": int((asof - t_obs[last]).astype(int)),
                "lead_med30": _med(lead[w30]),
                "lead_med90": _med(lead[w90]),
            })
    df = pd.DataFrame(rows)
    df["lead_trend"] = df.lead_med30 - df.lead_med90
    df["log_qty_vs_hist"] = df.log_qty - df.med90_log_qty
    df = df.merge(parts[["part_id", "category", "lifecycle", "mpn"]],
                  on="part_id", how="left")
    for c in ["category", "lifecycle"]:
        df[c] = df[c].astype("category")
    # Target: log ratio of the future price to the reference price
    df["y"] = np.log(df.y_price / df.ref_price)
    return df

FEATURES = ["log_qty", "log_qty_vs_hist", "trend_30_90", "n_offers_90",
            "days_since_last", "lead_med30", "lead_med90", "lead_trend",
            "category", "lifecycle"]
