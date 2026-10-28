"""number_guard.py - answer to exercise 33-4: flag numbers in an LLM draft
that did not come from any tool output. Example data only."""
import re

NUM = re.compile(r"\d+(?:[.,]\d+)*")


def norm(n: str) -> str:
    """'2,500' -> '2500', '4,70' (German decimal) -> '4.70'."""
    if re.fullmatch(r"\d{1,3}(,\d{3})+", n):
        return n.replace(",", "")
    return n.replace(",", ".")


def unsupported_numbers(draft: str, tool_outputs: list[dict]) -> list[str]:
    allowed = set()
    for out in tool_outputs:
        for v in out.values():
            for n in NUM.findall(str(v)):
                allowed.add(norm(n))
                allowed.add(str(float(norm(n))))   # 4.7 and 4.70 match
    bad = []
    for n in NUM.findall(draft):
        x = norm(n)
        if x not in allowed and str(float(x)) not in allowed:
            bad.append(n)
    return bad


if __name__ == "__main__":
    tools = [{"mpn": "STM32F407VGT6", "qty": 2500},
             {"unit_price": 10.19, "lead_days": 21}]
    draft = ("Dear Markus, for 2,500 pcs STM32F407VGT6 we offer 10.19 EUR "
             "per piece, delivery in 21 days. Price valid for 14 days.")
    print(unsupported_numbers(draft, tools))
