"""Runnable answers for exercises 12-1, 12-4 and 12-5 (example data)."""
import re
from bpe import CORPUS, train, encode

# 12-1
m, v, w = train({"LM317": 4, "LM337": 2, "LM358": 3}, 3, verbose=True)
print("12-1 corpus tokens:", sum(len(k) * f for k, f in w.items()))


# 12-4
def encode_by_rank(word, merges):
    """Repeatedly merge the adjacent pair with the lowest merge rank."""
    rank = {pair: i for i, pair in enumerate(merges)}
    symbols = list(word)
    while len(symbols) > 1:
        pairs = [(rank.get((a, b), float("inf")), i)
                 for i, (a, b) in enumerate(zip(symbols, symbols[1:]))]
        best_rank, i = min(pairs)
        if best_rank == float("inf"):
            break
        symbols = symbols[:i] + [symbols[i] + symbols[i + 1]] + symbols[i + 2:]
    return symbols


merges, _, _ = train(CORPUS, 12, verbose=False)
for wd in ["STM32F407VGT6", "STM32F411", "TPS5430DDAR", "TPS54331", "LM2596S-ADJ"]:
    assert encode_by_rank(wd, merges) == encode(wd, merges), wd
print("12-4: same results")

# 12-5
MPN_RE = re.compile(r"\b(?=[A-Z0-9,\-]*\d)[A-Z0-9][A-Z0-9,\-]{5,}\b")


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def mpn_guard(draft, allowed_mpns):
    problems = []
    for mm in MPN_RE.findall(draft):
        if mm not in allowed_mpns:
            best = min(allowed_mpns, key=lambda a: edit_distance(mm, a))
            problems.append((mm, best, edit_distance(mm, best)))
    return problems


draft = "Dear Markus, we can offer GRM188R71H104KA39D and STM32F407VGT6 ..."  # example
print("12-5:", mpn_guard(draft, {"GRM188R71H104KA93D", "STM32F407VGT6"}))
