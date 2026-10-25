"""Validate the 16-field annotation CSV and write the clean master CSV.

Usage (from the project root, venv active):
    python src/prepare_dataset.py data/annotations_raw.csv
"""
import os, re, sys
import pandas as pd

RAW = sys.argv[1] if len(sys.argv) > 1 else "data/annotations_raw.csv"
OUT = "data/annotations.csv"
FIELDS = [f"F{i:02d}" for i in range(1, 17)]
PROBES = {"F13", "F14", "F15", "F16"}
NO_UNIT_OK = {"F10", "F11", "F12"}

# Dash-printed rows with no tag; by the paper's definition they are printed_blank.
RETAG = {"mahindra_F07": "printed_blank", "nestle_F05": "printed_blank"}

if os.path.abspath(RAW) == os.path.abspath(OUT):
    sys.exit("Input and output are the same file. Rename the raw CSV to data/annotations_raw.csv first.")
df = pd.read_csv(RAW, dtype=str, encoding="utf-8-sig").fillna("")
need = {"row_id", "company_id", "fold", "fy", "pdf_filename", "field_id", "status",
        "gold_value", "gold_unit", "page", "annotator", "notes", "reviewer"}
if not need <= set(df.columns):
    sys.exit(f"Wrong sheet exported. Missing columns: {sorted(need - set(df.columns))}")
df = df.apply(lambda c: c.str.strip())
errors, warnings = [], []

# Excel/Sheets exports often end with empty rows that are just commas
blank_rows = (df == "").all(axis=1)
if blank_rows.any():
    warnings.append(f"dropped {int(blank_rows.sum())} completely empty rows from the CSV")
    df = df[~blank_rows].reset_index(drop=True)
no_pdf = df[df.pdf_filename == ""]
for rid in no_pdf.row_id:
    errors.append(f"{rid}: pdf_filename is empty")

# Excel/Sheets exports often end with empty rows that are just commas
blank_rows = (df == "").all(axis=1)
if blank_rows.any():
    warnings.append(f"dropped {int(blank_rows.sum())} completely empty rows from the CSV")
    df = df[~blank_rows].reset_index(drop=True)
no_pdf = df[df.pdf_filename == ""]
for rid in no_pdf.row_id:
    errors.append(f"{rid}: pdf_filename is empty")

df.loc[df.field_id == "F12", "gold_value"] = df.loc[df.field_id == "F12", "gold_value"].str.capitalize()

sci = df[df.gold_value.str.contains(r"\d[eE][+-]?\d", regex=True)]
for _, r in sci.iterrows():
    errors.append(f"{r.row_id}: '{r.gold_value}' looks like Excel scientific notation; retype as plain text")

if len(df) != 320: errors.append(f"expected 320 rows, found {len(df)}")
if df.row_id.duplicated().any(): errors.append("duplicate row_ids")
bad = df[~df.field_id.isin(FIELDS)]
if len(bad): errors.append(f"unknown field ids: {sorted(bad.field_id.unique())}")
per_co = df.groupby("company_id").size()
if not (per_co == 16).all(): errors.append(f"companies without 16 rows: {per_co[per_co != 16].to_dict()}")

def fy_from_name(n):
    m = re.search(r"_FY(\d{2})(\d{2})\.pdf$", n)
    return f"FY20{m.group(1)}-{m.group(2)}" if m else ""
df["fy_from_filename"] = df.pdf_filename.map(fy_from_name)
for _, r in df[df.fy != df.fy_from_filename].drop_duplicates("company_id").iterrows():
    warnings.append(f"FY mismatch {r.company_id}: sheet says {r.fy}, filename says {r.fy_from_filename}. Check the PDF cover and fix whichever is wrong.")

pages = {}
if os.path.isdir("raw_pdfs"):
    import pymupdf
    pages = {f: len(pymupdf.open(f"raw_pdfs/{f}")) for f in df.pdf_filename.unique()
             if f and os.path.isfile(f"raw_pdfs/{f}")}
    missing = [f for f in df.pdf_filename.unique() if f and not os.path.isfile(f"raw_pdfs/{f}")]
    if missing: warnings.append(f"PDF not found in raw_pdfs/: {missing}")
else:
    warnings.append("raw_pdfs/ not found: page-range check skipped")

for _, r in df.iterrows():
    rid = r.row_id
    if r.status not in {"answered", "zero", "not_disclosed"}:
        errors.append(f"{rid}: bad status '{r.status}'"); continue
    if r.status == "not_disclosed":
        if r.gold_value: errors.append(f"{rid}: not_disclosed but has gold_value")
        continue
    if not r.gold_value: errors.append(f"{rid}: {r.status} but gold_value empty")
    if r.status == "zero" and r.gold_value not in {"0", "0.0", "0.00", "Nil"}:
        errors.append(f"{rid}: zero but value '{r.gold_value}'")
    if r.field_id not in NO_UNIT_OK and not r.gold_unit:
        errors.append(f"{rid}: missing unit")
    if r.field_id == "F12" and r.gold_value not in {"Yes", "No"}:
        errors.append(f"{rid}: F12 must be Yes/No")
    if not r.page.isdigit(): errors.append(f"{rid}: page missing")
    elif r.pdf_filename in pages and not 1 <= int(r.page) <= pages[r.pdf_filename]:
        errors.append(f"{rid}: page {r.page} outside 1..{pages[r.pdf_filename]}")

def label_type(r):
    if r.status == "answered": return "answered"
    if r.status == "zero": return "zero"
    if r.row_id in RETAG: return RETAG[r.row_id]
    for tag, name in [("PARTS_ONLY", "parts_only"), ("ROW_ABSENT", "row_absent"),
                      ("PRINTED_BLANK", "printed_blank")]:
        if r.notes.startswith(tag): return name
    return "other_nd"
df["label_type"] = df.apply(label_type, axis=1)
df["is_probe"] = df.field_id.isin(PROBES)
df["far_eligible"] = (df.status == "not_disclosed") & (df.label_type != "parts_only")
df["acc_main_eligible"] = df.label_type != "parts_only"

print(f"{len(df)} rows, {df.company_id.nunique()} companies, {len(errors)} errors, {len(warnings)} warnings")
for e in errors: print("  ERROR  ", e)
for w in warnings: print("  WARNING", w)
print("\nTable II counts:\n", df.label_type.value_counts().to_string())
print(f"\nFAR-eligible: {df.far_eligible.sum()}   accuracy set (no parts_only): {df.acc_main_eligible.sum()}")
print("FAR-eligible per fold:", df[df.far_eligible].groupby("fold").size().to_dict())
print("Probe fields F13-F16:", df[df.is_probe].status.value_counts().to_dict())
print("Reports per FY (filename):", df.drop_duplicates("company_id").fy_from_filename.value_counts().to_dict())
print("Reports per FY (sheet):   ", df.drop_duplicates("company_id").fy.value_counts().to_dict())

if not errors:
    df.to_csv(OUT, index=False)
    print(f"\nWrote {OUT}")
else:
    print("\nFix the errors in the sheet, re-export the CSV and run again (nothing written).")
