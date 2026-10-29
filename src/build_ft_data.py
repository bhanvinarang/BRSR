"""Build QLoRA training data: one example per question of the 15 training companies in each fold.

The target is the gold answer when it can be read from the retrieved context, otherwise
{"answer": "not disclosed"}, so the model learns to abstain when evidence is missing.
"Readable from the context" means:
  * numeric answers: the gold number is printed in a retrieved chunk (short numbers of fewer than
    four digits must also come from the gold page, to avoid coincidental matches);
  * Yes/No and zero answers: the gold page is among the retrieved pages;
  * rows labelled not_disclosed (including PARTS_ONLY): always "not disclosed".
Test companies never appear: each fold's file contains only companies from the other three folds.

    python src/build_ft_data.py        ->  data/ft/fold{0..3}_train.jsonl, data/ft/stats.json
"""
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from metrics import context_text, value_in_context
from prompts import build_messages, gold_to_target, target_json

ROOT = Path(__file__).resolve().parent.parent
K = 5


def answer_is_readable(row, chunks):
    pages = {c["page"] for c in chunks}
    page_hit = str(row["page"]).isdigit() and int(row["page"]) in pages
    if row["status"] == "not_disclosed":
        return True
    if row["status"] == "zero" or row["field_id"] == "F12":
        return page_hit
    digits = "".join(ch for ch in str(row["gold_value"]) if ch.isdigit())
    present = value_in_context(row["gold_value"], context_text(chunks))
    return present and (len(digits) >= 4 or "." in str(row["gold_value"]) or page_hit)


def main():
    annotations = pd.read_csv(ROOT / "data" / "annotations.csv", dtype=str).fillna("")
    folds = json.loads((ROOT / "data" / "folds.json").read_text(encoding="utf-8"))
    frozen = {}
    for line in open(ROOT / "data" / "retrieval_top5.jsonl", encoding="utf-8"):
        rec = json.loads(line)
        frozen[rec["row_id"]] = rec["retrieved_chunks"][:K]

    out_dir = ROOT / "data" / "ft"
    out_dir.mkdir(exist_ok=True)
    stats = {}
    for fold in range(4):
        test_companies = {c for c, f in folds.items() if int(f) == fold}
        train = annotations[~annotations.company_id.isin(test_companies)]
        assert test_companies.isdisjoint(set(train.company_id)), "test company leaked into training"
        assert train.company_id.nunique() == 15 and len(train) == 240
        counts = Counter()
        with (out_dir / f"fold{fold}_train.jsonl").open("w", encoding="utf-8") as stream:
            for row in train.to_dict("records"):
                chunks = frozen[row["row_id"]]
                readable = answer_is_readable(row, chunks)
                target = gold_to_target(row) if readable else {"answer": "not disclosed"}
                relabelled = (not readable) and row["status"] != "not_disclosed"
                counts["relabelled_by_retrieval_miss"] += relabelled
                counts["target_not_disclosed" if target["answer"] == "not disclosed" else "target_value"] += 1
                stream.write(json.dumps({
                    "row_id": row["row_id"], "company_id": row["company_id"], "fold": fold,
                    "field_id": row["field_id"], "relabelled": bool(relabelled),
                    "messages": build_messages(row["question"], chunks),
                    "target": target_json(target),
                }, ensure_ascii=False) + "\n")
        stats[fold] = {"examples": len(train), "train_companies": sorted(train.company_id.unique()),
                       "test_companies": sorted(test_companies), **counts}
        print(f"fold {fold}: {len(train)} examples | value targets {counts['target_value']} | "
              f"not-disclosed targets {counts['target_not_disclosed']} | "
              f"answerable rows relabelled because retrieval missed them: {counts['relabelled_by_retrieval_miss']}")
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
