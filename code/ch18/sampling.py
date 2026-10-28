"""sampling.py - decoding strategies in NumPy on a 4-token toy distribution, plus KV-cache arithmetic."""
import numpy as np

np.set_printoptions(precision=4, suppress=True)
TOKENS = ["8", "6", "12", "4"]                 # candidate next tokens after "lead time "
LOGITS = np.array([2.0, 1.0, 0.5, -1.0])


def softmax(z):
    z = np.asarray(z, float)
    e = np.exp(z - z.max())
    return e / e.sum()


def with_temperature(logits, T):
    return softmax(np.asarray(logits) / T)


def top_k(logits, k):
    z = np.asarray(logits, float).copy()
    kth = np.sort(z)[-k]
    z[z < kth] = -np.inf                         # keep only the k largest
    return softmax(z)


def top_p(logits, p):
    probs = softmax(logits)
    order = np.argsort(probs)[::-1]
    cum = np.cumsum(probs[order])
    keep = order[: int(np.searchsorted(cum, p) + 1)]   # smallest set whose mass >= p
    out = np.zeros_like(probs)
    out[keep] = probs[keep]
    return out / out.sum()


def repetition_penalty(logits, previous_ids, penalty=1.3):
    """Same rule as many open-source servers: divide positive logits, multiply negative ones."""
    z = np.asarray(logits, float).copy()
    for i in set(previous_ids):
        z[i] = z[i] / penalty if z[i] > 0 else z[i] * penalty
    return z


def sample(probs, rng):
    return int(rng.choice(len(probs), p=probs))


def kv_cache_bytes(layers, kv_heads, head_dim, tokens, batch=1, bytes_per=2):
    return 2 * layers * kv_heads * head_dim * tokens * batch * bytes_per   # 2 = K and V


if __name__ == "__main__":
    print("tokens:", TOKENS, "logits:", LOGITS)
    print("softmax T=1   :", softmax(LOGITS))
    for T in (0.5, 2.0):
        print(f"T={T:<4}        :", with_temperature(LOGITS, T))
    print("greedy        :", TOKENS[int(np.argmax(LOGITS))])
    print("top-k k=2     :", top_k(LOGITS, 2))
    print("top-p p=0.9   :", top_p(LOGITS, 0.9))
    print("rep. penalty (token '8' seen):", repetition_penalty(LOGITS, [0]), "->", softmax(repetition_penalty(LOGITS, [0])))
    rng = np.random.default_rng(0)
    for name, pr in [("T=1", softmax(LOGITS)), ("T=0.5", with_temperature(LOGITS, 0.5)),
                     ("T=2", with_temperature(LOGITS, 2.0)), ("top-p 0.9", top_p(LOGITS, 0.9))]:
        draws = [sample(pr, rng) for _ in range(10_000)]
        print(f"10k samples {name:<9}:", np.bincount(draws, minlength=4) / 10_000)
    # KV cache for a 7B-class config (32 layers, head_dim 128), fp16
    for kvh, label in [(32, "MHA 32 kv-heads"), (8, "GQA 8 kv-heads")]:
        per_tok = kv_cache_bytes(32, kvh, 128, 1)
        print(f"{label}: {per_tok / 2**10:.0f} KiB/token, 4k ctx {kv_cache_bytes(32, kvh, 128, 4096) / 2**30:.2f} GiB, "
              f"4k ctx x 16 users {kv_cache_bytes(32, kvh, 128, 4096, 16) / 2**30:.1f} GiB")
