"""Create a clearly labelled, non-blinded AI source-page audit of the review sample.

This is a QA aid, not an independent human inter-annotator review. It must not be
used to claim inter-annotator agreement in the paper.

Usage:
    python src/run_ai_review_audit.py
"""
import hashlib
import re

import pandas as pd
import pymupdf


MASTER = "data/annotations.csv"
PACKET = "data/review_packet_blinded.csv"
OUT = "data/review_responses_ai_audit.csv"


def compact(value):
    """Comparison form robust to commas, spaces, line breaks, and CO₂ glyphs."""
    value = str(value).lower().replace("₂", "2")
    return re.sub(r"[^a-z0-9]", "", value)


master = pd.read_csv(MASTER, dtype=str).fillna("")
packet = pd.read_csv(PACKET, dtype=str).fillna("")
audit = packet.merge(
    master[["row_id", "status", "gold_value", "gold_unit", "page", "notes", "label_type"]],
    on="row_id", how="left", validate="one_to_one",
)
if len(audit) != 64 or audit.status.eq("").any():
    raise ValueError("Review packet must contain 64 rows from the master annotations.")

docs = {}
page_text, value_found, page_hash = [], [], []
for row in audit.itertuples(index=False):
    if row.pdf_filename not in docs:
        docs[row.pdf_filename] = pymupdf.open(f"raw_pdfs/{row.pdf_filename}")
    text = docs[row.pdf_filename][int(row.page) - 1].get_text()
    page_text.append(text)
    page_hash.append(hashlib.sha256(text.encode("utf-8")).hexdigest())
    if row.status == "not_disclosed":
        # Absence/blank decisions need semantic inspection; do not pretend this
        # text match proves an abstention label.
        value_found.append("")
    else:
        value_found.append(compact(row.gold_value) in compact(text))

for doc in docs.values():
    doc.close()

audit["reviewer"] = "Codex_AI_nonblinded_source_audit"
audit["review_status"] = audit.status
audit["review_value"] = audit.gold_value
audit["review_unit"] = audit.gold_unit
audit["review_page"] = audit.page
audit["review_notes"] = audit.notes
audit["review_method"] = (
    "Non-blinded AI source-page audit. This is not an independent human review "
    "and must not be used as inter-annotator agreement."
)
audit["annotated_page_text_sha256"] = page_hash
audit["value_found_on_annotated_page"] = value_found

def semantic_verdict(row):
    if row.status != "not_disclosed":
        return "value_visible_on_annotated_page"
    if row.label_type in {"printed_blank", "row_absent"}:
        return "confirmed_blank_or_absent_optional_row"
    if row.label_type == "parts_only":
        return "policy_sensitive_components_only"
    return "not_disclosed_requires_human_adjudication"


audit["ai_semantic_verdict"] = audit.apply(semantic_verdict, axis=1)
audit["human_adjudication_required"] = (
    audit.ai_semantic_verdict.isin({
        "policy_sensitive_components_only",
        "not_disclosed_requires_human_adjudication",
    }) | ~audit.value_found_on_annotated_page.eq(True) & audit.status.ne("not_disclosed")
)

audit.to_csv(OUT, index=False)
direct = audit[audit.status != "not_disclosed"]
print(f"Wrote {OUT} ({len(audit)} rows).")
print(f"Direct value check: {direct.value_found_on_annotated_page.sum()}/{len(direct)} values found on the annotated page.")
print(f"Human adjudication required: {audit.human_adjudication_required.sum()}/{len(audit)} rows.")
