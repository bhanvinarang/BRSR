"""Chunk extracted BRSR pages and build one normalized FAISS index per company.

Usage (from the project root):
    python src/chunk_index.py
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def load_pages(path):
    grouped = defaultdict(list)
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["text"]:
                grouped[row["company_id"]].append(row)
    return grouped


def chunk_text(text, chunk_size, overlap):
    """Split within a page, preferring whitespace boundaries where practical."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    start, chunks = 0, []
    while start < len(text):
        stop = min(start + chunk_size, len(text))
        if stop < len(text):
            boundary = text.rfind(" ", start + chunk_size // 2, stop)
            if boundary > start:
                stop = boundary
        piece = text[start:stop].strip()
        if piece:
            chunks.append((start, piece))
        if stop >= len(text):
            break
        start = max(stop - overlap, start + 1)
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", default="data/corpus/pages.jsonl")
    parser.add_argument("--index-dir", default="data/index")
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--chunk-size", type=int, default=1500)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--companies", nargs="+", help="Optional company IDs to build; useful for resumable CPU runs.")
    args = parser.parse_args()

    grouped = load_pages(args.pages)
    if len(grouped) != 20:
        raise ValueError(f"Expected pages for 20 companies; found {len(grouped)}")

    output = Path(args.index_dir)
    output.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(args.model, device=args.device)
    manifest_path = output / "manifest.json"
    manifest = {
        "embedding_model": args.model,
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "companies": {},
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("embedding_model", "chunk_size", "overlap"):
            if manifest[key] != getattr(args, key if key != "embedding_model" else "model"):
                raise ValueError("Existing index manifest uses different retrieval settings; remove data/index before rebuilding.")

    company_ids = args.companies or sorted(grouped)
    unknown = set(company_ids) - set(grouped)
    if unknown:
        raise ValueError(f"Unknown company IDs: {sorted(unknown)}")

    for company_id in company_ids:
        chunks = []
        for page in sorted(grouped[company_id], key=lambda item: item["page"]):
            for offset, text in chunk_text(page["text"], args.chunk_size, args.overlap):
                chunks.append({
                    "chunk_id": f"{company_id}_p{page['page']}_o{offset}",
                    "company_id": company_id,
                    "pdf_filename": page["pdf_filename"],
                    "page": page["page"],
                    "char_offset": offset,
                    "text": text,
                })
        if not chunks:
            raise ValueError(f"No text chunks generated for {company_id}")

        vectors = model.encode(
            [chunk["text"] for chunk in chunks],
            batch_size=args.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(np.ascontiguousarray(vectors))
        faiss.write_index(index, str(output / f"{company_id}.faiss"))
        with (output / f"{company_id}.chunks.jsonl").open("w", encoding="utf-8") as stream:
            for chunk in chunks:
                stream.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        manifest["companies"][company_id] = {
            "chunk_count": len(chunks),
            "dimension": int(vectors.shape[1]),
            "pdf_filename": chunks[0]["pdf_filename"],
        }
        print(f"{company_id}: {len(chunks)} chunks")

    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    print(f"Manifest now contains {len(manifest['companies'])} FAISS indexes in {output}.")


if __name__ == "__main__":
    main()
