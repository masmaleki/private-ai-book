"""infer_compare.py - base vs base+adapter on held-out emails, then merge.

python infer_compare.py --adapter out/adapter
"""
import argparse
import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
p.add_argument("--adapter", default="out/adapter")
p.add_argument("--n", type=int, default=16)
args = p.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)
rows = [json.loads(l) for l in open("eval.jsonl", encoding="utf-8")][: args.n]


def generate(model, msgs):
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=48, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:],
                      skip_special_tokens=True).strip()


def score(model, label):
    valid = exact = 0
    for i, r in enumerate(rows):
        gold = json.loads(r["completion"][0]["content"])
        text = generate(model, r["prompt"])
        try:
            pred = json.loads(text)
            valid += 1
            exact += pred == gold
        except json.JSONDecodeError:
            pred = None
        if i < 2:
            print(f"[{label}] {text[:80]}")
    n = len(rows)
    print(f"[{label}] valid JSON {valid}/{n}, exact match {exact}/{n}")


base = AutoModelForCausalLM.from_pretrained(args.model)
base.eval()
score(base, "base")

tuned = PeftModel.from_pretrained(base, args.adapter)
tuned.eval()
score(tuned, "base+LoRA")

# merge adapter into the weights -> a normal model, no PEFT needed to serve
merged = tuned.merge_and_unload()
merged.save_pretrained("out/merged")
tok.save_pretrained("out/merged")
print("merged model saved to out/merged")
