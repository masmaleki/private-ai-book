"""pricing.py - win-probability model + expected-profit price optimiser
+ business rules + quote draft JSON. ALL DATA SYNTHETIC (example).
"""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

rng = np.random.default_rng(42)

# ---------- 1) synthetic quote history ----------
SEG = {"lead_time_focused": 0, "price_focused": 1, "balanced": 2}
N = 4000
seg = rng.choice(list(SEG), N, p=[0.3, 0.3, 0.4])
markup = rng.uniform(1.00, 1.45, N)                 # price / expected cost
we_can_deliver_fast = rng.binomial(1, 0.5, N)       # our lead time beats market
n_competitors = rng.integers(1, 6, N)
log_qty = np.log(rng.choice([250, 1000, 2500, 10000], N))
# hidden "true" behaviour: price-focused customers hate markup
slope = np.where(seg == "price_focused", -14, np.where(
    seg == "lead_time_focused", -5, -9))
fast_bonus = np.where(seg == "lead_time_focused", 1.4, 0.4) * we_can_deliver_fast
logit = 3.2 + slope * (markup - 1) + fast_bonus - 0.25 * n_competitors \
        - 0.08 * (log_qty - 7)
won = rng.binomial(1, 1 / (1 + np.exp(-logit)))
hist = pd.DataFrame({"segment": seg, "markup": markup,
                     "fast": we_can_deliver_fast, "n_comp": n_competitors,
                     "log_qty": log_qty, "won": won})
hist.to_csv("demo_quote_history.csv", index=False)
print(f"quotes={N} overall win rate={won.mean():.3f}")

# ---------- 2) logistic win-probability model (time split) ----------
def X_of(d: pd.DataFrame) -> np.ndarray:
    m1 = d.markup - 1
    cols = [m1, d.fast, d.n_comp, d.log_qty]
    for s in ("price_focused", "lead_time_focused"):     # balanced = base
        ind = (d.segment == s).astype(float)
        cols += [ind, ind * m1, ind * d.fast]            # interactions
    return np.column_stack(cols)

tr, te = hist.iloc[:3200], hist.iloc[3200:]
model = LogisticRegression(C=10.0, max_iter=2000).fit(X_of(tr), tr.won)
p = model.predict_proba(X_of(te))[:, 1]
print(f"test AUC={roc_auc_score(te.won, p):.3f}  "
      f"Brier={brier_score_loss(te.won, p):.4f}  "
      f"base Brier={brier_score_loss(te.won, np.full(len(te), tr.won.mean())):.4f}")

def win_prob(segment, markup, fast, n_comp, qty):
    d = pd.DataFrame({"segment": [segment] * len(markup), "markup": markup,
                      "fast": fast, "n_comp": n_comp,
                      "log_qty": np.log(qty)})
    return model.predict_proba(X_of(d))[:, 1]

# ---------- 3) expected-profit optimiser with business rules ----------
RULES = {"min_margin_pct": 4.0, "max_markup": 1.40, "price_step": 0.01}

def optimise(cost_p50, cost_p90, qty, segment, fast, n_comp):
    grid = np.round(np.arange(cost_p50 * 1.0, cost_p50 * RULES["max_markup"],
                              RULES["price_step"]), 2)
    floor = cost_p90 * (1 + RULES["min_margin_pct"] / 100)  # protect vs P90
    grid = grid[grid >= floor]
    pw = win_prob(segment, grid / cost_p50, fast, n_comp, qty)
    exp_profit = pw * (grid - cost_p50) * qty
    i = int(np.argmax(exp_profit))
    return {"price": float(grid[i]), "p_win": float(pw[i]),
            "exp_profit": float(exp_profit[i]), "floor": round(floor, 2),
            "grid": grid, "pw": pw, "ep": exp_profit}

if __name__ == "__main__":
    # example: STM32F407VGT6, 2500 pcs, cost from ch27 API (example numbers)
    cost_p50, cost_p90, qty = 7.585, 8.4882, 2500
    for seg_name, fast in [("lead_time_focused", 1), ("price_focused", 0)]:
        r = optimise(cost_p50, cost_p90, qty, seg_name, fast, n_comp=3)
        print(f"\n{seg_name:18s} fast={fast}: floor={r['floor']:.2f} "
              f"best price={r['price']:.2f} EUR  P(win)={r['p_win']:.3f}  "
              f"E[profit]={r['exp_profit']:.0f} EUR")
        for pr in (8.85, 9.10, 9.40, 9.70, 10.00):
            j = int(np.argmin(np.abs(r["grid"] - pr)))
            print(f"   price {r['grid'][j]:.2f}: P(win)={r['pw'][j]:.3f} "
                  f"E[profit]={r['ep'][j]:.0f}")
    r = optimise(cost_p50, cost_p90, qty, "lead_time_focused", 1, 3)
    draft = {"quote_draft_id": "QD-EXAMPLE-0001", "customer": "AlpenBoard Example GmbH",
             "lines": [{"mpn": "STM32F407VGT6", "qty": qty,
                        "unit_price": r["price"], "currency": "EUR",
                        "lead_time_days_p90": 38, "valid_days": 14}],
             "model": {"win_prob": round(r["p_win"], 3),
                       "cost_p50": cost_p50, "cost_p90": cost_p90,
                       "price_floor": r["floor"],
                       "versions": {"price": "price-demo-0.1",
                                    "winprob": "winprob-demo-0.1"}},
             "status": "DRAFT_NEEDS_APPROVAL", "data": "synthetic"}
    json.dump(draft, open("quote_draft.json", "w"), indent=1)
    print("\n" + json.dumps(draft["lines"][0]))
