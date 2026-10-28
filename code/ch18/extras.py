"""extras.py - chapter 17 arithmetic: constrained decoding mask, capacity planning,
decode speed bound, and the numbers used in the exercises. Toy/illustrative only."""
import numpy as np
from sampling import softmax, with_temperature, top_p, repetition_penalty, kv_cache_bytes

GiB = 2**30

# constrained decoding: after '"weeks": ' only digit tokens are allowed
vocab = ['"8"', "8", "eight", "about", "6", "}"]
logits = np.array([1.2, 2.0, 1.5, 0.9, 1.0, 0.3])
allowed = np.array([t.isdigit() for t in vocab])
print("vocab        :", vocab)
print("unconstrained:", softmax(logits).round(3))
print("constrained  :", softmax(np.where(allowed, logits, -np.inf)).round(3))

# capacity: how many 4k-token conversations fit next to a 7B model on one 48 GB GPU?
weights, overhead, gpu = 13.5e9, 3e9, 48e9          # bytes; overhead is an assumption
free = gpu - weights - overhead
per_tok = kv_cache_bytes(32, 8, 128, 1)             # GQA, fp16
print(f"free for KV: {free / 1e9:.1f} GB, per token {per_tok / 1024:.0f} KiB, "
      f"tokens {free / per_tok:,.0f}, 4k-conversations {free / per_tok / 4096:.0f}")

# decode upper bound: each new token must read all weights once from memory
for bw in (1.0e12, 2.0e12):                          # assumed memory bandwidth, bytes/s
    print(f"bandwidth {bw / 1e12:.0f} TB/s -> <= {bw / weights:.0f} tokens/s (one stream)")

# exercise 17-1 / 17-2
z = np.array([1.5, 1.0, 0.0, -0.5])
print("ex1 T=1  :", softmax(z).round(4))
print("ex1 T=0.5:", with_temperature(z, 0.5).round(4))
print("ex2 top-p 0.8:", top_p(z, 0.8).round(4))
# exercise 17-3
b = kv_cache_bytes(28, 4, 128, 32768)
print(f"ex3 one 32k seq: {b / GiB:.2f} GiB, 10 users: {10 * b / GiB:.1f} GiB, "
      f"per token {kv_cache_bytes(28, 4, 128, 1) / 1024:.0f} KiB")
# exercise 17-4
z4 = np.array([2.0, -0.5, 1.0])
p4 = repetition_penalty(z4, [0, 1], 1.2)
print("ex4 before:", softmax(z4).round(4), "after logits:", p4.round(4), "->", softmax(p4).round(4))
# exercise 17-8
free8 = 48e9 - 15e9 - 3e9
for bytes_per, lab in ((2, "fp16"), (1, "fp8")):
    t = free8 / kv_cache_bytes(32, 8, 128, 1, bytes_per=bytes_per)
    print(f"ex8 {lab}: tokens {t:,.0f}, 6k requests {t / 6144:.1f}")
