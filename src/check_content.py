import pymupdf, glob, re

KEYS = {
    "scope1":    r"scope\s*1",
    "scope2":    r"scope\s*2",
    "energy":    r"total\s+energy\s+consumption",
    "electricity": r"electricity",
    "water":     r"water\s+withdrawal|withdrawn",
    "discharge": r"water\s+discharge|discharged",
    "waste":     r"total\s+waste\s+generated",
    "hazardous": r"hazardous",
    "employees": r"employees",
    "women":     r"female|women",
    "grievance": r"grievance",
}

for f in sorted(glob.glob("raw_pdfs/*.pdf")):
    doc = pymupdf.open(f)
    text = " ".join(p.get_text() for p in doc).lower()
    missing = [k for k, v in KEYS.items() if not re.search(v, text)]
    print(f"{f.split('/')[-1]:42s} missing: {missing if missing else 'none'}")
