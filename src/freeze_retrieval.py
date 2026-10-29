"""Freeze the verified retrieval output so every system reads identical context.

Reads results/retrieval_recall_final_rows.csv (written by evaluate_retrieval.py with the selected
hybrid settings, top-5 chunks per question) and writes data/retrieval_top5.jsonl.  The GPU machine then
needs neither FAISS nor the embedding model.  Top-k for k < 5 is simply the first k chunks.

    python src/freeze_retrieval.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def main():
    rows = pd.read_csv(ROOT / "results" / "retrieval_recall_final_rows.csv", dtype=str)
    annotations = pd.read_csv(ROOT / "data" / "annotations.csv", dtype=str).fillna("")
    assert len(rows) == len(annotations) == 320
    assert list(rows.row_id) == list(annotations.row_id), "row order differs from annotations.csv"
    out = ROOT / "data" / "retrieval_top5.jsonl"
    with out.open("w", encoding="utf-8") as stream:
        for row, ann in zip(rows.itertuples(index=False), annotations.itertuples(index=False)):
            chunks = json.loads(row.retrieved_chunks)
            assert len(chunks) == 5, (row.row_id, len(chunks))
            assert all(c["chunk_id"].startswith(ann.company_id) for c in chunks), "chunk from another company"
            stream.write(json.dumps({
                "row_id": ann.row_id, "company_id": ann.company_id, "fold": int(ann.fold),
                "field_id": ann.field_id, "retrieved_chunks": chunks,
            }, ensure_ascii=False) + "\n")
    print(f"Wrote {out} ({len(rows)} questions, 5 chunks each)")


if __name__ == "__main__":
    main()
