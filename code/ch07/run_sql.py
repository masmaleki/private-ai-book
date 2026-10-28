"""Run point_in_time.sql with DuckDB on the CSV files and cross-check with pandas."""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
con = duckdb.connect()
for name in ["demo_offers", "demo_quote_lines", "fx_rates"]:
    con.execute(f"CREATE TABLE {name} AS SELECT * FROM read_csv_auto(?)",
                [str(HERE / "data" / f"{name}.csv")])
res = con.execute((HERE / "point_in_time.sql").read_text()).df()
pd.set_option("display.width", 100)
print(res.drop(columns=["issued_at"]).to_string(index=False))

# cross-check n30 for the same rows with plain pandas
o = pd.read_csv(HERE / "data/demo_offers.csv", parse_dates=["observed_at"])
q = pd.read_csv(HERE / "data/demo_quote_lines.csv", parse_dates=["issued_at"])
check = []
for _, r in q.head(5).iterrows():
    sel = (o.part_id == r.part_id) & (o.observed_at < r.issued_at)
    sel &= o.observed_at >= r.issued_at - pd.Timedelta(days=30)
    check.append(int(sel.sum()))
print("pandas n30:", check, " match:", np.array_equal(check, res["n30"].to_numpy()))
