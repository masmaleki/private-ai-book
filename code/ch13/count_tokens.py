"""count_tokens.py - usage: python3 count_tokens.py FILE TOKENIZER_JSON"""
import sys
from tokenizers import Tokenizer

path, tok_path = sys.argv[1], sys.argv[2]
text = open(path, encoding="utf-8").read()
tok = Tokenizer.from_file(tok_path)         # local file only, no network
enc = tok.encode(text, add_special_tokens=False)
n_chars, n_tok = len(text), len(enc.ids)
print(f"chars={n_chars} tokens={n_tok} tokens/char={n_tok / max(n_chars, 1):.3f}")
print("first 20 tokens:", enc.tokens[:20])
