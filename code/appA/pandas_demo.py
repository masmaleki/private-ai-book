"""Appendix A: pandas in ten minutes on example demo_* data (synthetic)."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
parts = pd.DataFrame({
    "part_id": [1, 2, 3],
    "mpn": ["STM32F407VGT6", "TPS5430DDAR", "NE555DR"],
    "category": ["MCU", "Power", "Timer"],
})
n = 600
offers = pd.DataFrame({
    "offer_id": np.arange(1, n + 1),
    "part_id": rng.choice([1, 2, 3], n, p=[0.5, 0.3, 0.2]),
    "qty": rng.choice([100, 1000, 5000], n),
    "observed_at": pd.Timestamp("2026-01-01")
    + pd.to_timedelta(rng.integers(0, 180, n), unit="D"),
})
base = offers.part_id.map({1: 9.0, 2: 2.5, 3: 0.30})          # example price levels
offers["unit_price"] = (base * rng.lognormal(0, 0.08, n)).round(4)
offers.loc[rng.choice(n, 5, replace=False), "unit_price"] = np.nan   # missing values

print("== head ==")
print(offers.head(3).to_string(index=False))
print("\n== info: rows, missing ==")
print(len(offers), "rows;", int(offers.unit_price.isna().sum()), "missing prices")

df = offers.merge(parts, on="part_id", how="left")              # join like SQL
df = df.dropna(subset=["unit_price"])
big = df[df.qty >= 1000]                                        # filter rows
print("\n== median price by part, qty >= 1000 ==")
print(big.groupby("mpn").unit_price.median().round(3).to_string())

df["month"] = df.observed_at.dt.to_period("M").dt.to_timestamp()
monthly = df.pivot_table(index="month", columns="mpn", values="unit_price",
                         aggfunc="median").round(3)
print("\n== monthly median price (pivot) ==")
print(monthly.to_string())

if __name__ == "__main__":
    from fa_plot import fa, plt, save
    norm = monthly / monthly.iloc[0]                             # index: first month = 1
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for c, col in zip(["#1f4e79", "#c0504d", "#4f8a3a"], norm.columns):
        ax.plot(norm.index, norm[col], "o-", color=c, label=col)
    ax.axhline(1, color="k", lw=0.6)
    ax.set_ylabel(fa("میانه‌ی قیمت نسبت به ژانویه"))
    ax.legend(fontsize=8)
    ax.set_title(fa("خروجی pivot_table به شکل نمودار (داده‌ی مصنوعی)", False))
    save(fig, "appA", "pandas_monthly.png")
