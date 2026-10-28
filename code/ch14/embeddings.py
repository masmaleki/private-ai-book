"""embeddings.py - cosine similarity, a tiny co-occurrence embedding and MPN n-gram vectors.
All text is synthetic (example data). Pure NumPy.
"""
import re
import numpy as np

np.set_printoptions(precision=3, suppress=True)


def cosine(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


# ---- 1) hand example: three part descriptions as 4-d feature vectors ----
# dims: [is_regulator, is_capacitor, max_input_voltage/40, switching(1)/linear(0)]
desc = {
    "buck regulator 36V 3A": [1, 0, 0.90, 1],
    "step-down converter 40V": [1, 0, 1.00, 1],
    "ceramic capacitor 100nF 50V": [0, 1, 1.25, 0],
}
names = list(desc)
print("=== hand example: cosine between descriptions ===")
for i in range(3):
    for j in range(i + 1, 3):
        print(f"{names[i]:<28} vs {names[j]:<28} {cosine(desc[names[i]], desc[names[j]]):.3f}")

# ---- 2) tiny co-occurrence embedding on synthetic sentences ----
rng = np.random.default_rng(42)
templates = {
    "regulator": ["the {p} buck regulator needs an inductor",
                  "{p} converter output voltage ripple",
                  "replace the {p} regulator with a buck converter",
                  "{p} switching regulator inductor and capacitor"],
    "mcu": ["the {p} microcontroller firmware flash",
            "{p} mcu with flash and ram",
            "program the {p} microcontroller over swd",
            "{p} mcu firmware update"],
    "capacitor": ["{p} ceramic capacitor 100nf",
                  "decoupling capacitor {p} near the mcu",
                  "{p} capacitor voltage rating x7r",
                  "{p} ceramic capacitor tape and reel"],
}
parts = {"regulator": ["tps5430", "tps5450", "lm2596"],
         "mcu": ["stm32f407", "atmega328p", "stm32f103"],
         "capacitor": ["grm188", "cl10b104"]}
sentences = []
for fam, ps in parts.items():
    for _ in range(60):
        p = rng.choice(ps)
        sentences.append(rng.choice(templates[fam]).format(p=p).split())

vocab = sorted({w for s in sentences for w in s})
idx = {w: i for i, w in enumerate(vocab)}
C = np.zeros((len(vocab), len(vocab)))
WINDOW = 3
for s in sentences:
    for i, w in enumerate(s):
        for j in range(max(0, i - WINDOW), min(len(s), i + WINDOW + 1)):
            if i != j:
                C[idx[w], idx[s[j]]] += 1
# positive pointwise mutual information, then truncated SVD -> dense vectors
total = C.sum()
row, col = C.sum(1, keepdims=True), C.sum(0, keepdims=True)
with np.errstate(divide="ignore"):
    pmi = np.log((C * total) / (row * col))
ppmi = np.maximum(pmi, 0)
U, S, _ = np.linalg.svd(ppmi)
DIM = 8
E = U[:, :DIM] * S[:DIM]              # embedding table: one row per word
print(f"\n=== co-occurrence embedding: vocab={len(vocab)} dim={DIM} ===")


def nearest(word, k=4):
    v = E[idx[word]]
    sims = [(cosine(v, E[i]), w) for w, i in idx.items() if w != word]
    return sorted(sims, reverse=True)[:k]


for w in ["tps5430", "stm32f407", "grm188", "inductor"]:
    print(f"{w:<10}", ", ".join(f"{n} {s:.2f}" for s, n in nearest(w)))

# ---- 3) character n-gram vectors for MPNs (why similar strings look close) ----
def ngram_vec(s, n=3, dim=512):
    s = f"^{s.upper()}$"
    v = np.zeros(dim)
    for i in range(len(s) - n + 1):
        v[hash_ngram(s[i:i + n]) % dim] += 1
    return v


def hash_ngram(g):
    # deterministic hash (Python's hash() is randomized per process)
    h = 0
    for ch in g:
        h = (h * 131 + ord(ch)) % 1_000_003
    return h


pairs = [("TPS5430", "TPS5450"), ("TPS5430", "TPS54331"), ("TPS5430DDAR", "TPS5430DDA"),
         ("STM32F407VGT6", "STM32F407ZGT6"), ("STM32F407VGT6", "STM32F103C8T6"),
         ("TPS5430", "GRM188R71H104KA93D")]
print("\n=== character 3-gram cosine between MPNs ===")
for a, b in pairs:
    print(f"{a:<14} vs {b:<20} {cosine(ngram_vec(a), ngram_vec(b)):.3f}")

np.save("embedding_table.npy", E)
with open("vocab.txt", "w") as f:
    f.write("\n".join(vocab))
