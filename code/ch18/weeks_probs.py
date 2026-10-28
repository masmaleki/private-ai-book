"""weeks_probs.py - what does mini-GPT believe after 'lead time '? (synthetic model)"""
import torch
import torch.nn.functional as F
from generate import m, tok

prompt = "please quote 500 pcs TPS5450DDAR, lead time "
with torch.no_grad():
    p = F.softmax(m(torch.tensor([tok.encode(prompt)]))[0][0, -1], -1)
for ch in "468" + "1":
    print(f"p(next char = {ch!r}) = {p[tok.stoi[ch]]:.3f}")
