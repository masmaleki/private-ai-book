"""tiny_lm.py - the world's smallest language model (teaching only).
A bigram model: for every word, count which words came right after it.
"""
import random
from collections import Counter, defaultdict

# Tiny, fictional training corpus (example data)
corpus = """
the customer needs 500 pcs of STM32F407VGT6 with short lead time
the customer needs 1000 pcs of TPS5430DDAR at a good price
the customer asks for an alternate for TPS5430DDAR
the supplier offers 500 pcs of STM32F407VGT6 with lead time of 12 weeks
the supplier offers 2000 pcs of TPS5430DDAR at a good price
the customer accepts the quote for STM32F407VGT6
""".split()

# 1) "Training": count what follows each word
counts = defaultdict(Counter)
for current_word, next_word in zip(corpus, corpus[1:]):
    counts[current_word][next_word] += 1

def next_word_probs(word):
    c = counts[word]
    total = sum(c.values())
    return {w: n / total for w, n in c.most_common()}

# 2) Inspect the model's "weights" for one word
print("after 'customer':", next_word_probs("customer"))
print("after 'of':", next_word_probs("of"))

# 3) "Inference": generate text, one word at a time
def generate(start, n_words=10, seed=None):
    rng = random.Random(seed)
    words = [start]
    for _ in range(n_words):
        probs = next_word_probs(words[-1])
        if not probs:
            break
        choices, weights = zip(*probs.items())
        words.append(rng.choices(choices, weights=weights)[0])
    return " ".join(words)

for s in (1, 2, 3, 4):
    print(f"seed={s}:", generate("the", seed=s))
