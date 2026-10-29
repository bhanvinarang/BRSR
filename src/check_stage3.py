"""Sanity numbers for the metrics: oracle, always-wrong and always-abstain predictors.

The oracle replies with the gold answer but only sees the retrieved context, so its UAR is the floor
that retrieval imposes on every system.   python src/check_stage3.py
"""
import json
from pathlib import Path

import pandas as pd

from metrics import context_text, score_prediction, summarize
from parse_output import parse_model_output
from prompts import gold_to_target

ROOT = Path(__file__).resolve().parent.parent


def main(rows_csv=ROOT / "results" / "retrieval_recall_final_rows.csv"):
    annotations = pd.read_csv(ROOT / "data" / "annotations.csv", dtype=str).fillna("")
    contexts = {r.row_id: context_text(json.loads(r.retrieved_chunks))
                for r in pd.read_csv(rows_csv, dtype=str).itertuples()}
    predictors = {
        "oracle (gold answer)": lambda row: json.dumps(gold_to_target(row)),
        "always a wrong number": lambda row: '{"answer": "12345", "unit": "GJ"}' if row["field_id"] != "F12" else '{"answer": "No"}',
        "always not disclosed": lambda row: '{"answer": "not disclosed"}',
    }
    for name, make in predictors.items():
        scored = [{**row, **score_prediction(row, parse_model_output(make(row)), contexts[row["row_id"]])}
                  for row in annotations.to_dict("records")]
        s = summarize(scored)
        print(f"{name:24s} accuracy {s['accuracy']:.3f} (n={s['n_accuracy']})  FAR {s['far']:.3f} (n={s['n_far']})  "
              f"UAR {s['uar']:.3f} (n={s['n_uar']})  parse errors {s['parse_error_rate']:.3f}")


if __name__ == "__main__":
    main()
