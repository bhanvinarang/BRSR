import glob, json, random, re, os
import pandas as pd

SECTOR = {
    "ambuja": "Cement", "shreecement": "Cement", "ultratechcement": "Cement",
    "bajajauto": "Auto", "mahindra": "Auto", "marutisuzuki": "Auto",
    "cipla": "Pharma", "drreddys": "Pharma", "sunpharma": "Pharma",
    "hcltech": "IT", "infosys": "IT", "tcs": "IT",
    "hindalco": "Steel/metals", "jswsteel": "Steel/metals", "tatasteel": "Steel/metals",
    "hindustanunilever": "FMCG", "itc": "FMCG", "nestle": "FMCG",
    "ntpc": "Energy/power", "powergrid": "Energy/power",
}

rows = []
for f in sorted(glob.glob("raw_pdfs/*.pdf")):
    name = os.path.basename(f)
    m = re.match(r"(.+)_FY(\d{2})(\d{2})\.pdf", name)
    cid, a, b = m.group(1), m.group(2), m.group(3)
    rows.append({"company_id": cid, "sector": SECTOR[cid],
                 "fy": f"FY20{a}-{b}", "pdf_filename": name, "source_url": ""})

df = pd.DataFrame(rows)
assert len(df) == 20, f"expected 20 companies, found {len(df)}"

ids = sorted(df.company_id)
random.Random(42).shuffle(ids)
folds = {cid: i // 5 for i, cid in enumerate(ids)}
df["fold"] = df.company_id.map(folds)
df["annotator"] = ""

df.to_csv("data/companies.csv", index=False)
json.dump(folds, open("data/folds.json", "w"), indent=2)

print(df[["company_id", "sector", "fy", "fold"]].sort_values("fold").to_string(index=False))
print("\nFY counts:\n", df.fy.value_counts())
print("\nFold sizes:\n", df.fold.value_counts().sort_index())
