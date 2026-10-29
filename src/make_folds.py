"""Build the fixed company-level folds and the report provenance manifest.

Usage (from the project root, venv active):
    python src/make_folds.py

This script is deterministic. It preserves an existing annotator assignment and
computes the file hash from the exact local PDF used for the experiment.
"""
import glob
import hashlib
import json
import os
import random
import re
from datetime import date

import pandas as pd
import pymupdf


SECTOR = {
    "ambuja": "Cement", "shreecement": "Cement", "ultratechcement": "Cement",
    "bajajauto": "Auto", "mahindra": "Auto", "marutisuzuki": "Auto",
    "cipla": "Pharma", "drreddys": "Pharma", "sunpharma": "Pharma",
    "hcltech": "IT", "infosys": "IT", "tcs": "IT",
    "hindalco": "Steel/metals", "jswsteel": "Steel/metals", "tatasteel": "Steel/metals",
    "hindustanunilever": "FMCG", "itc": "FMCG", "nestle": "FMCG",
    "ntpc": "Energy/power", "powergrid": "Energy/power",
}

# Official report PDFs where stable direct links are available; otherwise the
# official investor-relations or exchange record from which the report was obtained.
SOURCE_URL = {
    "ambuja": "https://www.ambujacement.com/investors/investor-downloads",
    "bajajauto": "https://www.bajajauto.com/-/media/images/bajajauto/media-kit/press-release/new-annual-reports/fy2024-25/bal_annual_report_2025.pdf",
    "cipla": "https://www.cipla.com/sites/default/files/Cipla-AR-2024-25.pdf",
    "drreddys": "https://www.nseindia.com/companies-listing/corporate-filings-announcements?symbol=DRREDDY&tabIndex=equity",
    "hcltech": "https://www.hcltech.com/hcl-annual-report-2026/approach-to-value-creation",
    "hindalco": "https://www.hindalco.com/investors/annual-general-meeting-2026",
    "hindustanunilever": "https://www.hul.co.in/files/annual-report-2024-25.pdf",
    "infosys": "https://www.infosys.com/investors/reports-filings/annual-report/annual/documents/infosys-ar-26.pdf",
    "itc": "https://itcportal.com/investors.html",
    "jswsteel": "https://www.jswsteel.in/investors-overview",
    "mahindra": "https://www.mahindra.com/annual-report-FY2026/index.html",
    "marutisuzuki": "https://www.bseindia.com/xml-data/corpfiling/AttachLive/bf787958-642f-4d2e-8b39-4642011531dc.pdf",
    "nestle": "https://nsearchives.nseindia.com/corporate/NESTLEIND1_03062025001441_NestleIndiaNoticeAGMAR202425signed.pdf",
    "ntpc": "https://www.ntpc.co.in/sites/default/files/annual-report/complete-reports/Annual%20Report%202024-25_1.pdf",
    "powergrid": "https://www.powergrid.in/sites/default/files/annual_reports/Powergrid_Annual_Report_2024-25-1.pdf",
    "shreecement": "https://www.nseindia.com/get-quote/equity/SHREECEM/SHREE-CEMENT-LIMITED",
    "sunpharma": "https://sunpharma.com/wp-content/uploads/2025/07/Sunpharmaannualreport202425.pdf",
    "tatasteel": "https://www.tatasteel.com/media/23970/brsr-fy2024-25.pdf",
    "tcs": "https://www.tcs.com/who-we-are/investor-relations/financial-statements",
    "ultratechcement": "https://www.ultratechcement.com/investors/financials",
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pdf_creation_date(path):
    value = pymupdf.open(path).metadata.get("creationDate", "")
    match = re.match(r"D:(\d{4})(\d{2})(\d{2})", value)
    return "-".join(match.groups()) if match else ""


existing_annotators = {}
if os.path.isfile("data/companies.csv"):
    existing = pd.read_csv("data/companies.csv", dtype=str).fillna("")
    existing_annotators = dict(zip(existing.company_id, existing.annotator))

rows = []
for path in sorted(glob.glob("raw_pdfs/*.pdf")):
    filename = os.path.basename(path)
    match = re.fullmatch(r"(.+)_FY(\d{2})(\d{2})\.pdf", filename)
    if not match:
        raise ValueError(f"Unexpected PDF filename: {filename}")
    company_id, start_year, end_year = match.groups()
    rows.append({
        "company_id": company_id,
        "sector": SECTOR[company_id],
        "fy": f"FY20{start_year}-{end_year}",
        "pdf_filename": filename,
        "source_url": SOURCE_URL[company_id],
        "source_accessed_date": date.today().isoformat(),
        "pdf_metadata_created": pdf_creation_date(path),
        "pdf_sha256": sha256(path),
        "pdf_bytes": os.path.getsize(path),
        "annotator": existing_annotators.get(company_id, ""),
    })

df = pd.DataFrame(rows)
assert len(df) == 20, f"expected 20 companies, found {len(df)}"
assert df.source_url.ne("").all()

ids = sorted(df.company_id)
random.Random(42).shuffle(ids)
folds = {company_id: index // 5 for index, company_id in enumerate(ids)}
df["fold"] = df.company_id.map(folds)
df = df[["company_id", "sector", "fy", "pdf_filename", "source_url",
         "source_accessed_date", "pdf_metadata_created", "pdf_sha256",
         "pdf_bytes", "fold", "annotator"]]

df.to_csv("data/companies.csv", index=False)
with open("data/folds.json", "w", encoding="utf-8") as stream:
    json.dump(folds, stream, indent=2)

print(df[["company_id", "sector", "fy", "fold"]].sort_values("fold").to_string(index=False))
print("\nFY counts:\n", df.fy.value_counts())
print("\nFold sizes:\n", df.fold.value_counts().sort_index())
print("\nWrote data/companies.csv with source URLs and file hashes.")
