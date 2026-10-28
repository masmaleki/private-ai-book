"""math_basics.py - Chapter 5: every math idea of the chapter in NumPy.
All data is fictional example data (synthetic).
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True)

print("== 1) vectors: an RFQ line as numbers ==")
# features: [log10(qty), lead_days/100, is_authorized, target_price_eur]
rfq = np.array([np.log10(2500), 0.21, 1.0, 7.80])
print("rfq vector:", rfq)

print("\n== 2) dot product and cosine similarity of part specs ==")
# specs: [vin_max/10, iout_A, fsw_MHz, package_is_soic8] (example parts)
a = np.array([3.6, 3.0, 0.5, 1.0])
b = np.array([4.0, 3.0, 0.5, 1.0])
c = np.array([1.2, 0.5, 2.0, 0.0])
def cosine(x, y):
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
print("a.b =", round(float(a @ b), 4), " a.c =", round(float(a @ c), 4))
print("cos(a,b) =", round(cosine(a, b), 4), " cos(a,c) =", round(cosine(a, c), 4))

print("\n== 3) matrix times vector = transformation ==")
# convert [price_usd, price_huf] style vector to EUR with a diagonal matrix
W = np.array([[0.92, 0.0], [0.0, 0.0025]])   # example FX rates
v = np.array([10.0, 4000.0])
print("W @ v =", W @ v)
# a 3x4 weight matrix maps a 4-feature part to 3 scores
rng = np.random.default_rng(42)
M = rng.normal(size=(3, 4)).round(2)
print("M =\n", M)
print("M @ a =", M @ a)

print("\n== 4) norms and distances ==")
print("L2(a-b) =", round(float(np.linalg.norm(a - b)), 4),
      " L1(a-b) =", round(float(np.abs(a - b).sum()), 4))

print("\n== 5) probability: mean, variance, std of lead times ==")
lead_days = np.array([21, 25, 19, 30, 22, 60, 24, 23])   # example offers
print("mean =", lead_days.mean(), " median =", np.median(lead_days))
print("var  =", round(lead_days.var(), 2), " std =", round(lead_days.std(), 2))

print("\n== 6) normal distribution and quantiles for a price band ==")
prices = rng.normal(loc=7.85, scale=0.40, size=10_000)
p10, p50, p90 = np.quantile(prices, [0.10, 0.50, 0.90])
print(f"P10={p10:.3f}  P50={p50:.3f}  P90={p90:.3f}")
print("theory: 7.85 -/+ 1.2816*0.40 =",
      round(7.85 - 1.2816 * 0.40, 3), round(7.85 + 1.2816 * 0.40, 3))

print("\n== 7) why log prices ==")
unit_prices = np.array([0.004, 0.02, 0.15, 1.08, 7.85, 42.0, 310.0])
print("raw   mean =", round(unit_prices.mean(), 3),
      " median =", np.median(unit_prices))
logp = np.log(unit_prices)
print("log   values:", logp.round(2))
print("10% up on 0.02 and on 310 in log space:",
      round(np.log(0.022) - np.log(0.02), 4),
      round(np.log(341.0) - np.log(310.0), 4))

print("\n== 8) derivative and gradient descent on a 1-D loss ==")
# loss(w) = (w - 3)^2 ; derivative 2(w - 3)
w, lr = 0.0, 0.1
for step in range(1, 6):
    grad = 2 * (w - 3)
    w = w - lr * grad
    print(f"step {step}: grad={grad:7.4f}  w={w:.4f}  loss={(w - 3)**2:.4f}")

print("\n== 9) softmax and cross-entropy by hand ==")
logits = np.array([2.0, 1.0, 0.1])     # e.g. scores for 3 intents
exp = np.exp(logits - logits.max())    # subtract max for stability
probs = exp / exp.sum()
print("softmax:", probs.round(4), " sum =", probs.sum().round(4))
true_class = 0
print("cross-entropy (true=0):", round(float(-np.log(probs[true_class])), 4))
print("cross-entropy (true=2):", round(float(-np.log(probs[2])), 4))
