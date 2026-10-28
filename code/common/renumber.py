"""renumber.py - renumber figure/table captions in order of appearance.
Usage: python3 renumber.py chapters/ch26.md 26
Also rewrites in-text references like «شکل ۲۶-۳» / «جدول ۲۶-۳».
"""
import re, sys
FA = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
EN = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
path, ch = sys.argv[1], sys.argv[2]
chfa = ch.translate(FA)
s = open(path, encoding="utf-8").read()
for kind in ("شکل", "جدول"):
    cap = re.compile(rf"^{kind} {chfa}-([۰-۹]+):", re.M)
    olds = [m.group(1) for m in cap.finditer(s)]
    mapping = {o: str(i + 1).translate(FA) for i, o in enumerate(olds)}
    if len(set(olds)) != len(olds):
        print("WARNING duplicate captions", kind, olds)
    ref = re.compile(rf"{kind} {chfa}-([۰-۹]+)")
    s = ref.sub(lambda m: f"{kind} {chfa}-\x00{mapping.get(m.group(1), m.group(1))}", s)
    s = s.replace("\x00", "")
    print(kind, "count", len(olds))
open(path, "w", encoding="utf-8").write(s)
