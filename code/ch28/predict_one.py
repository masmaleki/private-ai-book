"""predict_one.py - one worked prediction with explanation (example data)."""
import numpy as np
import pandas as pd
import joblib
from features import build_rows, FEATURES

bundle = joblib.load("price_model.joblib")
offers = pd.read_csv("demo_offers.csv", parse_dates=["observed_at"])
parts = pd.read_csv("demo_parts.csv")

def features_for(part_id: int, qty: int, asof: str) -> pd.DataFrame:
    """Build a single feature row by appending a fake 'target' offer
    at asof + 30 days; the price is a placeholder and is NOT used."""
    T = pd.Timestamp(asof) + pd.Timedelta(days=30)
    probe = pd.DataFrame([{"offer_id": -1, "tenant_id": "T-EXAMPLE",
                           "supplier_id": -1, "part_id": part_id, "qty": qty,
                           "unit_price": 1.0, "currency": "EUR",
                           "lead_days": 0, "observed_at": T}])
    hist = offers[(offers.part_id == part_id) &
                  (offers.observed_at <= pd.Timestamp(asof))]
    rows = build_rows(pd.concat([hist, probe]), parts)
    return rows[rows.target_date == T.to_datetime64()].tail(1)

if __name__ == "__main__":
    row = features_for(part_id=0, qty=2500, asof="2026-09-30")
    X = row[FEATURES]
    ref = float(row.ref_price.iloc[0])
    pred = {k: ref * float(np.exp(m.predict(X)[0]))
            for k, m in bundle["models"].items()}
    print("MPN:", row.mpn.iloc[0], "| qty 2500 | as-of 2026-09-30 | horizon 30d")
    print(f"reference (90-day median, all qty): {ref:.3f} EUR")
    for c in ["log_qty_vs_hist", "trend_30_90", "lead_med30", "lead_med90",
              "lead_trend", "n_offers_90", "days_since_last"]:
        print(f"  {c:16s} = {float(row[c].iloc[0]):.3f}")
    print(f"P10={pred['p10']:.3f}  P50={pred['p50']:.3f}  P90={pred['p90']:.3f} EUR")

    # Per-feature contributions (TreeSHAP built into LightGBM), log-ratio scale
    contrib = bundle["models"]["p50"].predict(X, pred_contrib=True)[0]
    s = pd.Series(contrib[:-1], index=FEATURES).sort_values(key=abs,
                                                            ascending=False)
    print("top contributions to log(P50/ref):")
    for k, v in s.head(4).items():
        print(f"  {k:16s} {v:+.4f}")
    print(f"  {'(base value)':16s} {contrib[-1]:+.4f}")
