"""Create the blinded 20% independent-review packet for BRSR-QA.

Usage:
    python src/make_review_sample.py

The reviewer receives no original status, value, unit, page, or annotation note.
"""
import pandas as pd


SEED = 20261026
SAMPLE_SIZE = 64  # exactly 20% of the 320 annotated rows

master = pd.read_csv("data/annotations.csv", dtype=str).fillna("")
if len(master) != 320:
    raise ValueError(f"Expected 320 master rows; found {len(master)}")

sample = master.sample(n=SAMPLE_SIZE, random_state=SEED).sort_values("row_id")
packet = sample[["row_id", "company_id", "fy", "pdf_filename", "field_id",
                 "field_name", "question"]].copy()
packet["review_instructions"] = (
    "Independently locate the current-year disclosure. Do not consult the original annotation. "
    "Record not_disclosed only when the requested total is not printed."
)
packet.to_csv("data/review_packet_blinded.csv", index=False)

responses = packet.copy()
responses["reviewer"] = ""
responses["review_status"] = ""  # answered, zero, or not_disclosed
responses["review_value"] = ""
responses["review_unit"] = ""
responses["review_page"] = ""
responses["review_notes"] = ""
responses.to_csv("data/review_responses.csv", index=False)

print(f"Wrote a blinded {len(packet)}-row review packet using seed {SEED}.")
print("Give data/review_responses.csv to a reviewer; retain data/annotations.csv privately.")
