"""ex17_answers.py - runnable answers for exercises 17-5 (min-p) and 17-6 (cached greedy)."""
import numpy as np
from sampling import LOGITS, TOKENS, softmax


def min_p(logits, p_base=0.1):
    """Keep tokens whose probability is at least p_base * (probability of the top token)."""
    probs = softmax(logits)
    keep = probs >= p_base * probs.max()
    out = np.where(keep, probs, 0.0)
    return out / out.sum()


print("tokens:", TOKENS)
for pb in (0.05, 0.1, 0.3):
    print(f"min-p {pb}:", min_p(LOGITS, pb).round(4))
flat = np.array([1.0, 0.9, 0.8, 0.7])                 # a flat distribution keeps more tokens
print("flat logits, min-p 0.3:", min_p(flat, 0.3).round(4))

if __name__ == "__main__":
    import torch
    from kvcache import m, tok, step_with_cache
    from generate import generate
    prompt = "rfq: STM32F407VGT6 qty "
    ids = torch.tensor([tok.encode(prompt)])
    cache = [None] * len(m.blocks)
    logits = step_with_cache(ids, cache)              # prefill
    out = []
    for _ in range(40):                               # greedy decode with the cache
        nxt = int(logits[0, -1].argmax())
        out.append(nxt)
        if tok.decode([nxt]) == "\n":
            break
        logits = step_with_cache(torch.tensor([[nxt]]), cache)
    cached_text = tok.decode(out).rstrip("\n")
    naive_text = generate(prompt, n=40)
    print("cached:", repr(cached_text))
    print("naive :", repr(naive_text))
    print("identical:", cached_text == naive_text)
