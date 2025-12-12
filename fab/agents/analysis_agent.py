import re
from decimal import Decimal, getcontext
getcontext().prec = 12

def normalize_number_str(s: str):
    if s is None: return None
    s = s.strip()
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1].strip()
    s = s.replace(",", "").replace("$", "").replace("AED", "").replace("US$", "")
    # unit suffix
    m = re.search(r"(?i)(-?\d+(\.\d+)?)\s*(bn|billion|m|million|k|thousand)\b", s)
    if m:
        num = Decimal(m.group(1))
        unit = m.group(3).lower()
        mult = Decimal(1)
        if unit.startswith("b"): mult = Decimal(1_000_000_000)
        if unit.startswith("m"): mult = Decimal(1_000_000)
        if unit.startswith("k") or unit.startswith("t"): mult = Decimal(1_000)
        val = num * mult
        return -val if negative else val
    # fallback numeric
    m2 = re.search(r"-?\d+(\.\d+)?", s.replace(",", ""))
    if m2:
        val = Decimal(m2.group(0))
        return -val if negative else val
    return None

def find_number_in_text(text: str, keywords=None):
    if keywords is None:
        keywords = [r"net profit", r"net income", r"profit for the period", r"profit for the year",
                    r"profit attributable to owners", r"profit after tax"]
    lines = re.split(r"\n|\r", text)
    for idx, line in enumerate(lines):
        for kw in keywords:
            if re.search(kw, line, re.I):
                # consider current + next 2 lines
                window = " ".join(lines[idx: idx+3])
                nums = re.findall(r"\(?-?\d[\d,\.]*(?:\s*(?:bn|billion|m|million|k|thousand))?\)?", window, flags=re.I)
                for nm in nums:
                    val = normalize_number_str(nm)
                    if val is not None:
                        return {"value": val, "matched_text": nm, "context": window.strip(), "line_index": idx}
    # fallback global search
    gm = re.findall(r"\(?-?\d[\d,\.]*(?:\s*(?:bn|billion|m|million|k|thousand))?\)?", text, flags=re.I)
    for g in gm:
        v = normalize_number_str(g)
        if v is not None:
            return {"value": v, "matched_text": g, "context": text[:300]}
    return None

def calc_pct_change(old, new):
    if old is None or new is None:
        return None
    try:
        if old == 0:
            return None
        return (new - old) / old * Decimal(100)
    except Exception:
        return None
