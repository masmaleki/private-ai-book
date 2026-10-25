"""real_tokenizers.py - how real open-model tokenizers split MPNs, numbers and languages.
Needs the `tokenizers` and `tiktoken` packages. Tokenizer files are fetched once from the
public model hub; on an air-gapped server, copy them in through the approved import path
and use Tokenizer.from_file(...) instead.
"""
import tiktoken
from tokenizers import Tokenizer

TOKENIZERS = {
    "Qwen2.5": Tokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct"),
    "Mistral-v0.3": Tokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3"),
}
TIKTOKEN = {"cl100k": tiktoken.get_encoding("cl100k_base"),
            "o200k": tiktoken.get_encoding("o200k_base")}


def pieces(name, text):
    if name in TOKENIZERS:
        enc = TOKENIZERS[name].encode(text, add_special_tokens=False)
        return enc.tokens, enc.ids
    enc = TIKTOKEN[name]
    ids = enc.encode(text)
    return [enc.decode_single_token_bytes(i).decode("utf-8", "replace") for i in ids], ids


ALL = list(TOKENIZERS) + list(TIKTOKEN)

print("=== MPNs ===")
for mpn in ["STM32F407VGT6", "TPS5430DDAR", "GRM188R71H104KA93D", "BC847C,215"]:
    for name in ["Qwen2.5", "Mistral-v0.3"]:
        toks, _ = pieces(name, mpn)
        print(f"{name:<13} {mpn:<20} {len(toks):2d} {toks}")

print("\n=== same RFQ sentence in three languages (token counts) ===")
SENT = {  # example sentence, translated by hand
    "EN": "We need 2,500 pcs of STM32F407VGT6 within three weeks; "
          "delivery time is more important than price.",
    "DE": "Wir benötigen 2.500 Stück STM32F407VGT6 innerhalb von drei Wochen; "
          "die Lieferzeit ist wichtiger als der Preis.",
    "FA": "ما ظرف سه هفته به ۲۵۰۰ عدد STM32F407VGT6 نیاز داریم؛ "
          "زمان تحویل از قیمت مهم‌تر است.",
}
print(f"{'lang':<5} {'chars':>5} " + " ".join(f"{n:>13}" for n in ALL))
for lang, text in SENT.items():
    counts = [len(pieces(n, text)[0]) for n in ALL]
    print(f"{lang:<5} {len(text):>5} " + " ".join(f"{c:>13}" for c in counts))

print("\n=== numbers and prices ===")
for t in ["1.08 EUR", "1,08 €", "0.97", "12500", "12,500", "2026-10-17", "۲۵۰۰"]:
    toks, _ = pieces("Qwen2.5", t)
    toks_m, _ = pieces("Mistral-v0.3", t)
    print(f"{t:<12} qwen={toks}  mistral={toks_m}")

print("\n=== special tokens (Qwen2.5 chat format) ===")
q = TOKENIZERS["Qwen2.5"]
for st in ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]:
    print(f"{st:<15} id={q.token_to_id(st)}")
chat = "<|im_start|>user\nPrice for TPS5430DDAR?<|im_end|>\n<|im_start|>assistant\n"
enc = q.encode(chat, add_special_tokens=False)
print("chat prompt tokens:", len(enc.ids))
print(enc.tokens)
print("vocab sizes:", {n: t.get_vocab_size() for n, t in TOKENIZERS.items()},
      {n: e.n_vocab for n, e in TIKTOKEN.items()})
