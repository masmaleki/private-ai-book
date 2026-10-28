"""generate.py - generate with the trained mini-GPT using several decoding strategies,
and measure prefill vs decode time with and without a KV cache (naive re-computation).
Needs ../ch16/minigpt_long.pt."""
import sys, time
sys.path.insert(0, "../ch15")
import torch
import torch.nn.functional as F
from minigpt import load

torch.set_num_threads(4)
m, tok = load("../ch16/minigpt_long.pt")


@torch.no_grad()
def generate(prompt, n=60, strategy="greedy", T=1.0, k=None, p=None, rep=None, seed=0, stop="\n"):
    g = torch.Generator().manual_seed(seed)
    ids = tok.encode(prompt)
    for _ in range(n):
        logits = m(torch.tensor([ids[-m.c.block_size:]]))[0][0, -1].clone()
        if rep:                                          # repetition penalty on already generated chars
            for i in set(ids[len(prompt):]):
                logits[i] = logits[i] / rep if logits[i] > 0 else logits[i] * rep
        if strategy == "greedy":
            nxt = int(torch.argmax(logits))
        else:
            logits = logits / T
            if k:
                kth = torch.topk(logits, k).values[-1]
                logits[logits < kth] = -float("inf")
            probs = F.softmax(logits, -1)
            if p:
                sp, si = torch.sort(probs, descending=True)
                cut = int((torch.cumsum(sp, 0) < p).sum()) + 1
                mask = torch.zeros_like(probs); mask[si[:cut]] = 1
                probs = probs * mask / (probs * mask).sum()
            nxt = int(torch.multinomial(probs, 1, generator=g))
        ids.append(nxt)
        if stop and tok.decode([nxt]) == stop:
            break
    return tok.decode(ids[len(tok.encode(prompt)):]).rstrip("\n")


if __name__ == "__main__":
    prompt = "please quote 500 pcs "
    print("prompt:", repr(prompt))
    print("greedy        :", generate(prompt))
    for name, kw in [("T=0.7", dict(T=0.7)), ("T=1.0", dict(T=1.0)), ("T=2.0", dict(T=2.0)),
                     ("top-k=3", dict(T=1.0, k=3)), ("top-p=0.9", dict(T=1.0, p=0.9))]:
        outs = [generate(prompt, strategy="sample", seed=s, **kw) for s in range(3)]
        print(f"{name:<14}:", " | ".join(outs))
    # how often does each strategy produce a part number that is NOT in the training list?
    from rfq_data import PARTS
    for name, kw in [("T=0.7", dict(T=0.7)), ("T=1.0", dict(T=1.0)), ("T=2.0", dict(T=2.0)), ("top-p=0.9", dict(T=1.0, p=0.9))]:
        bad = 0
        for s in range(100):
            out = generate("rfq: ", strategy="sample", seed=s, **kw)
            mpn = out.split(" ")[0]
            bad += mpn not in PARTS
        print(f"invented MPN rate {name:<10}: {bad}/100")

    # prefill vs decode timing (no KV cache in this toy: every step re-runs the full prefix)
    ids = torch.tensor([tok.encode("please quote 500 pcs STM32F407VGT6, lead time 8 weeks. we need ")])
    t0 = time.perf_counter(); m(ids); prefill = time.perf_counter() - t0
    steps = []
    x = ids
    for _ in range(20):
        t0 = time.perf_counter(); lg = m(x[:, -m.c.block_size:])[0]; steps.append(time.perf_counter() - t0)
        x = torch.cat([x, lg[:, -1:].argmax(-1)], 1)
    print(f"prefill {ids.shape[1]} tokens: {prefill * 1e3:.1f} ms; decode per token (mean of 20): {sum(steps) / 20 * 1e3:.1f} ms")
