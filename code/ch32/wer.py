"""wer.py - word error rate with an explicit alignment (no libraries)."""
import json, re, glob

def norm(text, numbers=True):
    t = text.lower()
    if numbers:   # spoken -> written forms used in our reference
        for a, b in [("twenty five hundred", "2500"), ("three weeks", "3 weeks"),
                     ("zero six six zero one two three four", "06601234")]:
            t = t.replace(a, b)
    t = re.sub(r"[^\w\s]", " ", t)
    return t.split()

def align(ref, hyp):
    """Levenshtein on words; returns (S, D, I, ops)."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): d[i][0] = i
    for j in range(m + 1): d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = 0 if ref[i-1] == hyp[j-1] else 1
            d[i][j] = min(d[i-1][j-1] + c, d[i-1][j] + 1, d[i][j-1] + 1)
    i, j, ops = n, m, []
    while i or j:
        if i and j and d[i][j] == d[i-1][j-1] + (ref[i-1] != hyp[j-1]):
            ops.append(("=" if ref[i-1] == hyp[j-1] else "S", ref[i-1], hyp[j-1])); i -= 1; j -= 1
        elif i and d[i][j] == d[i-1][j] + 1:
            ops.append(("D", ref[i-1], "")); i -= 1
        else:
            ops.append(("I", "", hyp[j-1])); j -= 1
    ops.reverse()
    S = sum(o[0] == "S" for o in ops); D = sum(o[0] == "D" for o in ops)
    I = sum(o[0] == "I" for o in ops)
    return S, D, I, ops

def wer(ref_text, hyp_text, numbers=True):
    r, h = norm(ref_text, numbers), norm(hyp_text, numbers)
    S, D, I, ops = align(r, h)
    return (S + D + I) / len(r), S, D, I, len(r), ops

if __name__ == "__main__":
    # 12-word sentence from the chapter (worked by hand)
    ref = "we need the parts by thursday because the line stops on monday"
    hyp = "we need parts by thursday because the lines stops on a monday"
    w, S, D, I, N, ops = wer(ref, hyp)
    print(f"hand example: N={N} S={S} D={D} I={I} WER={w:.3f}")
    print(" ".join(f"{o[0]}:{o[1] or '-'}/{o[2] or '-'}" for o in ops if o[0] != "="))
    turns = json.load(open("turns.json"))
    REF = " ".join(t["text"] for t in turns)
    rows = []
    for f in sorted(glob.glob("out_*.json")):
        o = json.load(open(f))
        hyp = " ".join(s["text"] for s in o["segments"])
        w1 = wer(REF, hyp, numbers=False)[0]
        w2, S, D, I, N, _ = wer(REF, hyp)
        rows.append({"model": o["model"], "audio": o["wav"], "wer_raw": round(w1, 3),
                     "wer_norm": round(w2, 3), "S": S, "D": D, "I": I,
                     "rtf": round(o["seconds"] / o["duration"], 3)})
        print(f"{o['model']:5s} {o['wav']:15s} raw={w1:.3f} norm={w2:.3f} "
              f"S={S} D={D} I={I} rtf={o['seconds']/o['duration']:.2f}")
    json.dump(rows, open("wer_results.json", "w"), indent=1)
