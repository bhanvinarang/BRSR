"""Measure gold-page retrieval recall for every BRSR-QA question.

Usage (from the project root, after extraction and indexing):
    python src/evaluate_retrieval.py
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from retrieve import CompanyRetriever, FIELD_TERMS, QUERY_PREFIX


def recall_summary(rows, k_values):
    output = {}
    for group_name, group in rows.groupby("evaluation_group", dropna=False):
        output[group_name] = {f"recall_at_{k}": float(group[f"recall_at_{k}"].mean()) for k in k_values}
        output[group_name]["n"] = int(len(group))
    return output


def _digits(value):
    return re.sub(r"[^0-9.]", "", str(value).replace(",", ""))


def value_in_text(text, gold_value):
    """True if the gold number is printed in ``text`` (Indian/Western comma grouping ignored)."""
    target = _digits(gold_value)
    flat = re.sub(r"(?<=\d),(?=\d)", "", text)
    return re.search(r"(?<![\d.])" + re.escape(target) + r"(?!\d)", flat) is not None


def value_scoreable(row):
    """Answered numeric rows whose value is long enough to match without chance hits."""
    target = _digits(row.gold_value)
    return (row.status == "answered" and row.field_id != "F12" and str(row.page).isdigit()
            and (len(target) >= 4 or "." in target))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default="data/annotations.csv")
    parser.add_argument("--index-dir", default="data/index")
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--k", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument(
        "--unique-pages", action="store_true",
        help="Use one highest-scoring chunk per page before filling the top-k evidence budget.",
    )
    parser.add_argument("--strategy", choices=["dense", "hybrid"], default="hybrid")
    parser.add_argument("--lexical-weight", type=float, default=1.5)
    parser.add_argument(
        "--legacy-lexical", action="store_true",
        help="BM25 on question + FIELD_TERMS (earlier baseline) instead of the BRSR template wording.",
    )
    parser.add_argument("--candidate-k", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--query-vectors", nargs="+",
        help="Precomputed .npy query-vector files in annotation order; avoids re-embedding during evaluation.",
    )
    parser.add_argument("--output", default="results/retrieval_recall_rows.csv")
    parser.add_argument("--summary", default="results/retrieval_recall_summary.json")
    args = parser.parse_args()
    k_values = sorted(set(args.k))
    if not k_values or k_values[0] < 1:
        raise ValueError("--k values must be positive")

    annotations = pd.read_csv(args.annotations, dtype=str).fillna("")
    if len(annotations) != 320:
        raise ValueError(f"Expected 320 annotations; found {len(annotations)}")
    retriever = CompanyRetriever(
        args.index_dir, args.model, args.device, load_model=not bool(args.query_vectors)
    )
    max_k = max(k_values)
    query_texts = []
    for row in annotations.itertuples(index=False):
        expanded = row.question
        if row.field_id in FIELD_TERMS:
            expanded = f"{row.question} BRSR disclosure table {FIELD_TERMS[row.field_id]}"
        query_texts.append(QUERY_PREFIX + expanded)
    if args.query_vectors:
        query_vectors = np.concatenate([np.load(path) for path in args.query_vectors])
        if len(query_vectors) != len(annotations):
            raise ValueError("Precomputed query-vector count does not match annotations.")
    else:
        query_vectors = retriever.model.encode(
            query_texts, batch_size=args.batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        ).astype("float32")
    records = []
    for row, query_vector in zip(annotations.itertuples(index=False), query_vectors):
        retrieved = retriever.retrieve(
            row.company_id, row.question, max_k, unique_pages=args.unique_pages,
            candidate_k=args.candidate_k, field_id=row.field_id, strategy=args.strategy,
            query_embedding=query_vector, lexical_weight=args.lexical_weight,
            lexical_template=not args.legacy_lexical,
        )
        pages = [item["page"] for item in retrieved]
        evidence = [{
            "chunk_id": item["chunk_id"], "page": item["page"],
            "score": round(item["score"], 6), "text": item["text"],
        } for item in retrieved]
        has_gold_page = str(row.page).isdigit()
        gold_page = int(row.page) if has_gold_page else None
        record = {
            "row_id": row.row_id,
            "company_id": row.company_id,
            "fold": row.fold,
            "field_id": row.field_id,
            "status": row.status,
            "label_type": row.label_type,
            "gold_page": gold_page,
            "has_gold_page": has_gold_page,
            "retrieved_pages_at_max_k": json.dumps(pages),
            "retrieved_chunks": json.dumps(evidence, ensure_ascii=False),
        }
        scoreable = value_scoreable(row)
        for k in k_values:
            record[f"recall_at_{k}"] = int(gold_page in pages[:k]) if has_gold_page else None
            record[f"value_recall_at_{k}"] = (
                int(any(value_in_text(item["text"], row.gold_value) for item in retrieved[:k]))
                if scoreable else None
            )
        records.append(record)

    output = pd.DataFrame(records)
    output["evaluation_group"] = "all"
    output.loc[output.status.isin(["answered", "zero"]), "evaluation_group"] = "answerable"
    output.loc[output.status.eq("not_disclosed"), "evaluation_group"] = "not_disclosed"
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    scored = output[output.has_gold_page]
    summary = {
        "unique_pages": args.unique_pages,
        "strategy": args.strategy,
        "candidate_k": args.candidate_k,
        "scored_rows": int(len(scored)),
        "missing_gold_page_rows": int((~output.has_gold_page).sum()),
        "recall": recall_summary(scored, k_values),
        "main_accuracy_eligible": recall_summary(
            scored[scored.label_type != "parts_only"], k_values
        ),
        "lexical_weight": args.lexical_weight,
        "lexical_template": not args.legacy_lexical,
        # Share of answered numeric rows whose gold value is printed inside the top-k chunks
        # (page recall alone can overstate what the model actually sees).
        "value_recall": {
            f"value_recall_at_{k}": float(output[f"value_recall_at_{k}"].dropna().mean()) for k in k_values
        },
        "value_recall_n": int(output[f"value_recall_at_{k_values[0]}"].notna().sum()),
    }
    with Path(args.summary).open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {len(output)} retrieval rows to {args.output}.")


if __name__ == "__main__":
    main()
