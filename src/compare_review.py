"""Compare a completed independent-review sheet against the master annotations.

Usage:
    python src/compare_review.py data/review_responses.csv
"""
import sys

import pandas as pd


responses_path = sys.argv[1] if len(sys.argv) > 1 else "data/review_responses.csv"
master = pd.read_csv("data/annotations.csv", dtype=str).fillna("")
review = pd.read_csv(responses_path, dtype=str).fillna("")

required = {"row_id", "reviewer", "review_status", "review_value", "review_unit", "review_page"}
missing = required - set(review.columns)
if missing:
    raise ValueError(f"Review file is missing columns: {sorted(missing)}")
if len(review) != 64 or review.row_id.duplicated().any():
    raise ValueError("The review file must contain exactly 64 distinct rows.")
if review.reviewer.eq("").any() or review.review_status.eq("").any():
    raise ValueError("Reviewer and review_status must be completed for every sampled row.")
if not review.review_status.isin({"answered", "zero", "not_disclosed"}).all():
    raise ValueError("review_status must be answered, zero, or not_disclosed.")

columns = ["row_id", "status", "gold_value", "gold_unit", "page", "label_type"]
out = review.merge(master[columns], on="row_id", how="left", validate="one_to_one")
if out.status.isna().any():
    raise ValueError("A review row_id does not appear in the master annotations.")

def normalize(value):
    return "".join(str(value).lower().split()).replace(",", "")

out["status_agree"] = out.review_status.eq(out.status)
out["value_agree"] = out.apply(
    lambda row: row.status == "not_disclosed" or normalize(row.review_value) == normalize(row.gold_value), axis=1
)
out["unit_agree"] = out.apply(
    lambda row: row.status == "not_disclosed" or normalize(row.review_unit) == normalize(row.gold_unit), axis=1
)
out["page_agree"] = out.review_page.eq(out.page)
out["overall_agree"] = out.status_agree & out.value_agree & out.unit_agree
out.to_csv("results/review_comparison.csv", index=False)

print(f"Status agreement: {out.status_agree.mean():.1%} ({out.status_agree.sum()}/{len(out)})")
print(f"Value/unit agreement: {out.overall_agree.mean():.1%} ({out.overall_agree.sum()}/{len(out)})")
print("Wrote results/review_comparison.csv; adjudicate every disagreement before reporting agreement.")
