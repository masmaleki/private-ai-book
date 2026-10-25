"""sft.py - toy supervised fine-tuning (SFT) of the pretrained mini-GPT on tiny RFQ -> extraction pairs.
Loss is computed only on the answer characters (prompt positions are masked with -100).
Synthetic example data only. Usage: python3 sft.py (needs minigpt_long.pt from experiments.py)"""
import random, sys, time, json
sys.path.insert(0, "../ch15")
import torch
import torch.nn.functional as F
from minigpt import Config, MiniGPT, CharTok
from rfq_data import PARTS, QTY, WEEKS, TEMPLATES

torch.manual_seed(0)
SEP = "\nextract: "
MASK = "--no-mask" not in sys.argv          # exercise 16-5: train on prompt too


def example(rng):
    q, p, w = rng.choice(QTY), rng.choice(PARTS), rng.choice(WEEKS)
    prompt = rng.choice(TEMPLATES).format(q=q, p=p, w=w) + SEP
    answer = f"part {p}, qty {q}, weeks {w}.\n"
    return prompt, answer, (p, q, w)


def load_base(path="minigpt_long.pt", block=128):
    ck = torch.load(path)
    cfg = dict(ck["cfg"]); cfg["block_size"] = block      # RoPE has no position table -> longer block is ok
    m = MiniGPT(Config(**cfg)); m.load_state_dict(ck["state"])
    return m, CharTok("".join(ck["chars"]))


def extract(m, tok, prompt, n=48):
    m.eval()
    ids = torch.tensor([tok.encode(prompt)])
    out = m.generate(ids, n, temperature=0)[0, ids.shape[1]:].tolist()
    return tok.decode(out).split("\n")[0]


def accuracy(m, tok, tests):
    ok = 0
    for prompt, answer, _ in tests:
        ok += extract(m, tok, prompt) == answer.strip()
    return ok / len(tests)


if __name__ == "__main__":
    rng = random.Random(123)
    train = [example(rng) for _ in range(300)]          # 300 tiny Q/A pairs
    tests = [example(random.Random(10_000 + i)) for i in range(100)]
    m, tok = load_base()
    print("base model on a test prompt:", repr(tests[0][0] + extract(m, tok, tests[0][0])))
    print("base exact-match accuracy:", accuracy(m, tok, tests))

    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.0)
    g = random.Random(5)
    t0, log = time.time(), []
    m.train()
    for step in range(1, 301):
        batch = [g.choice(train) for _ in range(16)]
        seqs = [tok.encode(p + a) for p, a, _ in batch]
        T = max(len(s) for s in seqs)
        x = torch.zeros(len(seqs), T - 1, dtype=torch.long)
        y = torch.full((len(seqs), T - 1), -100, dtype=torch.long)
        for i, ((p, a, _), s) in enumerate(zip(batch, seqs)):
            x[i, :len(s) - 1] = torch.tensor(s[:-1])
            tgt = torch.tensor(s[1:])
            if MASK:
                tgt[:len(p) - 1] = -100                # do not train on the prompt, only the answer
            y[i, :len(s) - 1] = tgt
        logits, _ = m(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                               ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0:
            log.append({"step": step, "loss": loss.item()})
            print(f"sft step {step} answer-loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
    print("after SFT on a test prompt:", repr(tests[0][0] + extract(m, tok, tests[0][0])))
    print("SFT exact-match accuracy:", accuracy(m, tok, tests))
    wrong = {"part": 0, "qty": 0, "weeks": 0}
    for prompt, _, (p, q, w) in tests:
        out = extract(m, tok, prompt)
        wrong["part"] += f"part {p}," not in out
        wrong["qty"] += f"qty {q}," not in out
        wrong["weeks"] += f"weeks {w}." not in out
    print("wrong fields out of 100:", wrong)
    # does it still write plain RFQ text? (forgetting check)
    print("after SFT, free generation:", repr(tok.decode(m.generate(torch.tensor([tok.encode("please quote ")]), 50, temperature=0)[0].tolist())))
    # a prompt format it never saw in SFT
    odd = "Hello, we would need 750 pieces of STM32F407VGT6." + SEP
    print("unseen format:", repr(odd + extract(m, tok, odd)))
    json.dump(log, open("sft.json" if MASK else "sft_nomask.json", "w"), indent=1)
