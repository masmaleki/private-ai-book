"""experiments.py - chapter 16 experiments with the chapter-15 mini-GPT (synthetic data, CPU).
1) longer pretraining run  2) toy scaling (model size)  3) memorization with a tiny corpus.
Writes JSON logs next to this file. Usage: python3 experiments.py [long|scale|memo|all]"""
import json, math, sys, time, random
sys.path.insert(0, "../ch15")
import torch
torch.set_num_threads(2)
from minigpt import Config, MiniGPT, CharTok, batches
from rfq_data import corpus, PARTS, QTY, WEEKS, TEMPLATES


def run(text_train, text_val, chars, d=96, L=4, steps=3000, lr=3e-3, bs=32, eval_every=100, seed=0, save=None):
    torch.manual_seed(seed)
    tok = CharTok("".join(chars))
    tr, va = torch.tensor(tok.encode(text_train)), torch.tensor(tok.encode(text_val))
    c = Config(len(chars), d_model=d, n_layers=L, n_heads=4)
    m = MiniGPT(c)
    n_params = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    g = torch.Generator().manual_seed(seed + 1)
    hist, t0 = [], time.time()
    for step in range(1, steps + 1):
        x, y = batches(tr, c.block_size, bs, g)
        _, loss = m(x, y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sched.step()
        if step % eval_every == 0 or step == 1:
            m.eval()
            with torch.no_grad():
                tl = sum(m(*batches(tr, c.block_size, 64, g))[1].item() for _ in range(5)) / 5
                vl = sum(m(*batches(va, c.block_size, 64, g))[1].item() for _ in range(5)) / 5
            m.train()
            hist.append({"step": step, "train": tl, "val": vl, "sec": round(time.time() - t0, 1)})
    if save:
        torch.save({"state": m.state_dict(), "chars": tok.chars, "cfg": vars(c)}, save)
    return n_params, hist


def entropy_floor(text):
    """Entropy rate of the generator: log(#templates * #qty * #parts * #weeks) nats per sentence / chars per sentence."""
    per_sentence = math.log(len(TEMPLATES) * len(QTY) * len(PARTS) * len(WEEKS))
    n_sent = text.count("\n")
    return per_sentence / (len(text) / n_sent)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    full = corpus(6000, seed=0)
    split = int(0.9 * len(full))
    chars = sorted(set(full))
    print("entropy floor (nats/char):", round(entropy_floor(full), 4), flush=True)
    if what in ("long", "all"):
        n, h = run(full[:split], full[split:], chars, steps=3000, save="minigpt_long.pt")
        json.dump(h, open("long.json", "w"), indent=1)
        print("long run params", n, "final", h[-1], flush=True)
    if what in ("scale", "all"):
        res = []
        for d in (16, 32, 64, 128):
            n, h = run(full[:split], full[split:], chars, d=d, steps=800)
            res.append({"d": d, "params": n, "val": h[-1]["val"], "sec": h[-1]["sec"]})
            print("scale", res[-1], flush=True)
        json.dump(res, open("scale.json", "w"), indent=1)
    if what in ("memo", "all"):
        tiny = corpus(40, seed=7)           # only 40 sentences -> the model can memorise them
        val = corpus(600, seed=99)
        n, h = run(tiny, val, chars, steps=1500)
        json.dump(h, open("memo.json", "w"), indent=1)
        print("memo final", h[-1], flush=True)
