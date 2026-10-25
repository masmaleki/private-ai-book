# fill missing months with zero demand (example data)
import pandas as pd

raw = pd.DataFrame({"part_id": [7, 7], "qty": [300, 200],
                    "month": pd.to_datetime(["2025-01-01", "2025-04-01"])})
months = pd.date_range("2025-01-01", "2025-06-01", freq="MS")
full = (raw.pivot_table(index="part_id", columns="month", values="qty",
                        aggfunc="sum")
           .reindex(columns=months, fill_value=0)
           .fillna(0).astype(int))
full.columns = full.columns.strftime("%Y-%m")
print(full)
