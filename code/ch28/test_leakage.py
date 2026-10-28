import pandas as pd
from features import build_rows, FEATURES

def test_no_future_leak():
    offers = pd.read_csv("demo_offers.csv", parse_dates=["observed_at"])
    parts = pd.read_csv("demo_parts.csv")
    base = build_rows(offers, parts)
    row = base.iloc[500]
    fake = offers.iloc[[0]].copy()
    fake["offer_id"] = -999
    fake["part_id"] = row.part_id
    fake["unit_price"] = 999.0
    fake["observed_at"] = row.asof_date + pd.Timedelta(days=1)
    again = build_rows(pd.concat([offers, fake]), parts)
    key = ["part_id", "target_date", "qty"]
    a = base.set_index(key).loc[tuple(row[key])]
    b = again.set_index(key).loc[tuple(row[key])]
    num = [f for f in FEATURES if f not in ("category", "lifecycle")]
    pd.testing.assert_series_equal(
        a[num].astype(float).squeeze(), b[num].astype(float).squeeze())

if __name__ == "__main__":
    test_no_future_leak(); print("no leak: OK")
