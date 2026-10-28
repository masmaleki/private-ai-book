"""bpe.py - a tiny Byte-Pair-Encoding trainer and encoder in pure Python (teaching only).
Character-level start, merges the most frequent adjacent pair each round.
Corpus = fictional word counts of MPN-like strings (example data).
"""
from collections import Counter

CORPUS = {          # word -> how often it appears in our toy corpus (example data)
    "STM32F407": 5,
    "STM32F103": 4,
    "STM32L476": 2,
    "TPS5430": 3,
    "TPS5450": 2,
    "LM2596": 2,
}


def pair_counts(words):
    counts = Counter()
    for symbols, freq in words.items():
        for a, b in zip(symbols, symbols[1:]):
            counts[(a, b)] += freq
    return counts


def merge(words, pair):
    a, b = pair
    out = {}
    for symbols, freq in words.items():
        new, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                new.append(a + b)
                i += 2
            else:
                new.append(symbols[i])
                i += 1
        out[tuple(new)] = freq
    return out


def train(corpus, n_merges, verbose=True):
    words = {tuple(w): f for w, f in corpus.items()}
    vocab = sorted({c for w in corpus for c in w})
    merges = []
    if verbose:
        print(f"start vocab ({len(vocab)}): {' '.join(vocab)}")
    for step in range(1, n_merges + 1):
        counts = pair_counts(words)
        if not counts:
            break
        # most frequent pair; ties broken alphabetically so the run is reproducible
        best = min(counts, key=lambda p: (-counts[p], p))
        merges.append(best)
        words = merge(words, best)
        vocab.append(best[0] + best[1])
        if verbose:
            top = ", ".join(f"{a}+{b}:{c}" for (a, b), c in
                            sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3])
            print(f"{step:2d}: {best[0] + '+' + best[1]:<12} -> {best[0] + best[1]:<10}"
                  f" n={counts[best]:2d} | top: {top}")
    return merges, vocab, words


def encode(word, merges):
    """Apply learned merges in the order they were learned."""
    symbols = list(word)
    for a, b in merges:
        i, new = 0, []
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                new.append(a + b)
                i += 2
            else:
                new.append(symbols[i])
                i += 1
        symbols = new
    return symbols


if __name__ == "__main__":
    merges, vocab, words = train(CORPUS, 12)
    print(f"\nfinal vocab size: {len(vocab)}")
    print("corpus after merges:")
    for w, f in words.items():
        print(f"  {'|'.join(w):<22} x{f}")
    ids = {tok: i for i, tok in enumerate(vocab)}
    print("\nencoding new strings:")
    for w in ["STM32F407VGT6", "STM32F411", "TPS5430DDAR", "TPS54331", "LM2596S-ADJ"]:
        toks = encode(w, merges)
        shown = [str(ids[t]) if t in ids else "UNK" for t in toks]
        print(f"  {w:<15} -> {toks}  ids={shown}")
