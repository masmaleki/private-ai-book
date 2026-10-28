"""scrub.py - PII pseudonymisation and fact placeholders (chapter 34).
Example data only. Names come from the (fictional) CRM contact list."""
import re

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE = re.compile(r"\+?\d[\d /-]{7,}\d")
IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){3,7}\b")
PRICE = re.compile(r"EUR \d+(?:[.,]\d+)?")
LEAD = re.compile(r"(lead time|Lieferzeit) \d+ (days|Tage)")
BREAK = re.compile(r"(From|Ab) [\d,.]+ (pcs|Stk\.)")
PLACEHOLDERS = {"{{contact_name}}", "{{contact_first_name}}", "{{customer_company}}",
                "{{email}}", "{{phone}}", "{{iban}}", "{{sender_name}}",
                "{{unit_price}}", "{{lead_days}}", "{{price_break_qty}}"}


def pseudonymise(text: str, contacts: list[str], companies: list[str],
                 staff: list[str], counts: dict) -> str:
    """Replace personal data with typed placeholders; count what we replaced."""
    def sub(pattern, repl, s, key):
        s, n = pattern.subn(repl, s)
        counts[key] = counts.get(key, 0) + n
        return s
    text = sub(EMAIL, "{{email}}", text, "email")
    text = sub(IBAN, "{{iban}}", text, "iban")
    text = sub(PHONE, "{{phone}}", text, "phone")
    for full in sorted(contacts, key=len, reverse=True):
        text = sub(re.compile(re.escape(full)), "{{contact_name}}", text, "name")
        first = full.split()[0]
        text = sub(re.compile(rf"\b{re.escape(first)}\b"), "{{contact_first_name}}",
                   text, "name")
    for c in companies:
        text = sub(re.compile(re.escape(c)), "{{customer_company}}", text, "company")
    for s in staff:
        text = sub(re.compile(rf"\b{re.escape(s)}\b"), "{{sender_name}}", text, "staff")
    return text


def facts_to_placeholders(reply: str, counts: dict) -> str:
    """Facts come from tools at inference time; the model learns the slots."""
    reply, n1 = PRICE.subn("EUR {{unit_price}}", reply)
    reply, n2 = LEAD.subn(lambda m: f"{m.group(1)} {{{{lead_days}}}} {m.group(2)}", reply)
    reply, n3 = BREAK.subn(lambda m: f"{m.group(1)} {{{{price_break_qty}}}} {m.group(2)}",
                           reply)
    counts["fact"] = counts.get("fact", 0) + n1 + n2 + n3
    return reply


def residual_pii(text: str, contacts: list[str]) -> list[str]:
    """Anything that still looks like personal data after scrubbing."""
    hits = EMAIL.findall(text) + PHONE.findall(text) + IBAN.findall(text)
    hits += [c for c in contacts if c.split()[0] in text]
    return hits


if __name__ == "__main__":
    c = {}
    t = ("Hello Markus,\nplease call me: +43 664 1234567, markus@alpen.example\n"
         "Markus Lechner\nAlpenBoard Example GmbH")
    print(pseudonymise(t, ["Markus Lechner"], ["AlpenBoard Example GmbH"], ["Sarah"], c))
    print(c)
