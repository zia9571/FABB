import os
import re
from decimal import Decimal, getcontext
getcontext().prec = 12

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

CHROMA_PATH = r"C:\Users\shaik\OneDrive\Desktop\fab\chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def infer_year_quarter_from_source(src: str):
    """
    Robustly infer year and quarter from a filename or source text.
    Returns dict: {"year": int or None, "quarter": "Q1"/"Q2"/"Q3"/"Q4"/None}
    """
    lower = src.lower()
    yr = None
    m = re.search(r"(20\d{2})", lower)
    if m:
        yr = int(m.group(1))
    q = None
    m = re.search(r"\bq([1-4])\b", src, re.I)
    if m:
        q = f"Q{int(m.group(1))}"
    else:
        # months mapping to quarter
        months = {
            "march": "Q1", "mar": "Q1",
            "june": "Q2", "jun": "Q2",
            "sept": "Q3", "september": "Q3", "sep": "Q3",
            "december": "Q4", "dec": "Q4"
        }
        for mon, qv in months.items():
            if mon in lower:
                q = qv
                break
    return {"year": yr, "quarter": q}

def normalize_number_str(s: str):
    """Convert strings like '(1,234)' or '1,234' or '1.2bn' to Decimal or None."""
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1].strip()
    # remove common symbols
    s = s.replace(",", "").replace("$", "").replace("AED", "").replace("US$", "")
    s = s.replace("—", "-").replace("–", "-")
    # handle million/billion short forms
    multiplier = Decimal(1)
    m = re.search(r"(?i)([\d\.\-]+)\s*(bn|billion|m|million|k|thousand)\b", s)
    if m:
        num = m.group(1)
        unit = m.group(2).lower()
        if unit.startswith("b"):
            multiplier = Decimal(1_000_000_000)
        elif unit.startswith("m"):
            multiplier = Decimal(1_000_000)
        elif unit.startswith("k") or unit.startswith("t"):
            multiplier = Decimal(1_000)
        try:
            val = Decimal(num) * multiplier
        except:
            return None
        return -val if negative else val
    m2 = re.search(r"(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+\.\d+|-?\d+)", s.replace(",", ""))
    if m2:
        try:
            val = Decimal(m2.group(0))
            return -val if negative else val
        except:
            return None
    return None

def find_number_in_text(text: str, keywords=None):
    """
    Search the text for lines containing keywords and return the first numeric value found.
    keywords: list of keyword regex strings (case-insensitive)
    """
    if keywords is None:
        keywords = [r"net profit", r"net income", r"profit for the period", r"profit for the year"]
    text_lower = text.lower()
    # split into lines and search near keyword
    lines = re.split(r"\n|\r", text)
    for i, line in enumerate(lines):
        for kw in keywords:
            if re.search(kw, line, re.I):
                window = " ".join(lines[i:i+3])
                # find numeric patterns in window
                num_matches = re.findall(r"\(?-?\d[\d,\.]*(?:\s*(?:bn|billion|m|million|k|thousand))?\)?", window, flags=re.I)
                if num_matches:
                    # return first that parses
                    for nm in num_matches:
                        val = normalize_number_str(nm)
                        if val is not None:
                            return {"value": val, "matched_text": nm, "context": window.strip(), "line_index": i}
    global_matches = re.findall(r"\(?-?\d[\d,\.]*(?:\s*(?:bn|billion|m|million|k|thousand))?\)?", text, flags=re.I)
    for gm in global_matches:
        val = normalize_number_str(gm)
        if val is not None:
            return {"value": val, "matched_text": gm, "context": text[:300]}
    return None

def load_db_and_extract(net_profit_query_text="net profit", sources_filter=None, k=6):
    if not os.path.exists(CHROMA_PATH):
        raise FileNotFoundError(f"Chroma directory not found at {CHROMA_PATH}")

    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    docs = db.similarity_search(net_profit_query_text, k=k)
    if sources_filter:
        docs = [d for d in docs if any(sf.lower() in (d.metadata.get("source") or "").lower() for sf in sources_filter)]

    extracted = []
    for d in docs:
        src = d.metadata.get("source", "")
        meta_inferred = infer_year_quarter_from_source(src)
        res = find_number_in_text(d.page_content, keywords=[r"net profit", r"net income", r"profit for the period", r"profit for the year"])
        extracted.append({
            "source": src,
            "metadata": d.metadata,
            "inferred": meta_inferred,
            "snippet": d.page_content[:400].replace("\n", " "),
            "extract": res
        })
    return extracted

def calc_pct_change(old: Decimal, new: Decimal):
    if old == 0:
        return None
    return (new - old) / old * Decimal(100)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python extract_and_calc.py <from_source_substring> <to_source_substring>")
        print("Example: python extract_and_calc.py Q3-2023 Q3-2024")
        sys.exit(1)

    from_sub = sys.argv[1]
    to_sub = sys.argv[2]

    print(f"Loading DB and searching for Net Profit near '{from_sub}' and '{to_sub}' ...\n")

    from_results = load_db_and_extract(sources_filter=[from_sub], k=12)
    to_results = load_db_and_extract(sources_filter=[to_sub], k=12)

    def choose_best_extract(results):
        best = None
        for r in results:
            if r["extract"] is None:
                continue
            # compute score: presence of matched_text and small line_index
            score = 1000
            if r["extract"].get("line_index") is not None:
                score = r["extract"]["line_index"]
            if best is None or score < best[0]:
                best = (score, r)
        return best[1] if best else None

    from_best = choose_best_extract(from_results)
    to_best = choose_best_extract(to_results)

    print("FROM best extract:", from_best)
    print("\nTO best extract:", to_best)

    if (not from_best) or (not to_best):
        print("\n Could not find numeric Net Profit for one or both quarters.")
        print("Diagnostics:")
        print(" - Number of candidate chunks for FROM:", len(from_results))
        print(" - Number of candidate chunks for TO:", len(to_results))
        print(" - For each candidate, we printed snippet + whether extract succeeded above.")
        sys.exit(1)

    old_val = from_best["extract"]["value"]
    new_val = to_best["extract"]["value"]
    pct = calc_pct_change(old_val, new_val)
    print("\n---- FINAL RESULT ----")
    print(f"FROM: {from_best['source']} inferred {from_best['inferred']} value {old_val}")
    print(f"TO:   {to_best['source']} inferred {to_best['inferred']} value {new_val}")
    print(f"YoY % change = {pct:.6f}%")
    print("\nCITATIONS:")
    print("FROM snippet:", from_best["extract"]["context"][:400])
    print("TO snippet:", to_best["extract"]["context"][:400])
