"""Embed a slice of field-expanded retrieval questions for resumable CPU evaluation.

Run eight slices (40 rows each), then pass the resulting files to
``evaluate_retrieval.py --query-vectors`` in the same order.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from retrieve import FIELD_TERMS, QUERY_PREFIX


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default="data/annotations.csv")
    parser.add_argument("--start", required=True, type=int)
    parser.add_argument("--end", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    annotations = pd.read_csv(args.annotations, dtype=str).fillna("")
    subset = annotations.iloc[args.start:args.end]
    if subset.empty:
        raise ValueError("The requested slice is empty.")
    queries = [
        QUERY_PREFIX + f"{row.question} BRSR disclosure table {FIELD_TERMS.get(row.field_id, '')}"
        for row in subset.itertuples(index=False)
    ]
    model = SentenceTransformer(args.model, device=args.device)
    vectors = model.encode(queries, batch_size=16, convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=False).astype("float32")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, vectors)
    print(f"Wrote {len(vectors)} vectors to {args.output}.")


if __name__ == "__main__":
    main()
