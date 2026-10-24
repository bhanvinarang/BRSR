import pymupdf, glob

for f in sorted(glob.glob("raw_pdfs/*.pdf")):
    doc = pymupdf.open(f)
    chars = sum(len(p.get_text()) for p in doc)
    print(f"{f}: {len(doc)} pages, {chars} chars, {chars/len(doc):.0f} chars/page")