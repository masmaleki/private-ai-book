"""demo.py - trace the shapes through the trained mini-GPT and generate a few samples."""
import torch
from minigpt import load

torch.manual_seed(0)
m, tok = load("minigpt.pt")
ids = torch.tensor([tok.encode("we need 500 x STM32")])
x = m.tok(ids); print("token ids", tuple(ids.shape), "-> embeddings", tuple(x.shape))
for i, b in enumerate(m.blocks):
    x = b(x); print(f"after block {i}:", tuple(x.shape))
logits = m.head(m.norm(x)); print("logits", tuple(logits.shape))
p = torch.softmax(logits[0, -1], -1)
top = torch.topk(p, 5)
print("next-char top-5 after 'we need 500 x STM32':",
      [(tok.decode([i]), round(v, 3)) for v, i in zip(top.values.tolist(), top.indices.tolist())])
for prompt in ["please quote ", "rfq: TPS", "we need 2000 x "]:
    out = m.generate(torch.tensor([tok.encode(prompt)]), 50, temperature=0)[0].tolist()
    print(repr(tok.decode(out).split("\n")[0]))
