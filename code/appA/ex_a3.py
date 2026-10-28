"""Exercise A-3: count and median price per part and quantity (synthetic data)."""
import contextlib
import io
import runpy

with contextlib.redirect_stdout(io.StringIO()):          # silence the demo prints
    df = runpy.run_path("pandas_demo.py")["df"]
g = df.groupby(["mpn", "qty"]).unit_price
print(g.agg(["count", "median"]).round(3))
