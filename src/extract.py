"""Extract selectable PDF text one page at a time for the BRSR retrieval corpus.

Usage (from the project root):
    python src/extract.py
"""
import argparse
import json
import re
from pathlib import Path

import pymupdf


def company_id_from_filename(filename):
    match = re.fullmatch(r"(.+)_FY\d{4}\.pdf", filename)
    if not match:
        raise ValueError(f"Unexpected report filename: {filename}")
    return match.group(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", default="raw_pdfs")
    parser.add_argument("--output", default="data/corpus/pages.jsonl")
    args = parser.parse_args()

    pdfs = sorted(Path(args.pdf_dir).glob("*.pdf"))
    if len(pdfs) != 20:
        raise ValueError(f"Expected 20 PDFs in {args.pdf_dir}; found {len(pdfs)}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows, empty_pages = 0, 0
    with output.open("w", encoding="utf-8") as stream:
        for pdf_path in pdfs:
            company_id = company_id_from_filename(pdf_path.name)
            with pymupdf.open(pdf_path) as document:
                for page_number, page in enumerate(document, start=1):
                    text = page.get_text("text").strip()
                    if not text:
                        empty_pages += 1
                    stream.write(json.dumps({
                        "company_id": company_id,
                        "pdf_filename": pdf_path.name,
                        "page": page_number,
                        "text": text,
                    }, ensure_ascii=False) + "\n")
                    rows += 1

    print(f"Wrote {rows} page records to {output}; empty text pages: {empty_pages}.")


if __name__ == "__main__":
    main()
