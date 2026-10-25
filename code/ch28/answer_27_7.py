import json, re
FA = str.maketrans("۰۱۲۳۴۵۶۷۸۹٫", "0123456789.")

def numbers_in(s):
    return re.findall(r"\d+(?:\.\d+)?", s.translate(FA))

def json_numbers(obj):
    if isinstance(obj, dict):
        for v in obj.values(): yield from json_numbers(v)
    elif isinstance(obj, list):
        for v in obj: yield from json_numbers(v)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield float(obj)

def unsupported(text, data):
    vals = list(json_numbers(data))
    bad = []
    for n in numbers_in(text):
        d = len(n.split(".")[1]) if "." in n else 0
        tol = 0.5 * 10 ** (-d) + 1e-9          # allow rounding to d digits
        if not any(abs(float(n) - x) <= tol for x in vals):
            bad.append(n)
    return bad

data = {"p50": 7.585, "p10": 6.83, "p90": 8.4882, "qty": 2500}
print(unsupported("حدود ۷٫۵۹ یورو، بازه ۶٫۸۳ تا ۸٫۴۹، تعداد 2500", data))
print(unsupported("قیمت ۷٫۲ یورو خواهد بود", data))
