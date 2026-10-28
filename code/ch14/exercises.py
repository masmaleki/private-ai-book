"""Answer code for exercise 13-5 (example data)."""
import numpy as np
from embeddings import E, idx, cosine


def embed(words):
    vecs = [E[idx[w]] for w in words if w in idx]
    return np.mean(vecs, axis=0) if vecs else np.zeros(E.shape[1])


def search(query_words, docs):
    q = embed(query_words)
    scored = [(cosine(q, embed(d)), " ".join(d)) for d in docs]
    return sorted(scored, reverse=True)


docs = [  # example data
    "the tps5430 buck regulator needs an inductor".split(),
    "stm32f407 mcu firmware update".split(),
    "grm188 ceramic capacitor 100nf".split(),
]
print("\n=== exercise 13-5 ===")
for s, d in search("buck converter inductor".split(), docs):
    print(f"{s:.2f}  {d}")
