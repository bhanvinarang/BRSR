"""Offline retrieval tuning without a GPU, FAISS or the embedding model.

Chunk vectors are read straight from the flat FAISS files and the field-expanded
query vectors saved by embed_retrieval_queries.py are reused, so only the lexical
side and the fusion can vary here.  Gold pages come from annotations.csv, so any
setting chosen with this script is selected on retrieval recall, not model accuracy.

Usage (project root):  python src/tune_retrieval_offline.py
"""
import json
import math
import re
import struct
import sys
from collections import Counter
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def _load_field_terms():
    """Read FIELD_TERMS from retrieve.py without importing it (it needs faiss)."""
    import ast
    tree = ast.parse((Path(__file__).resolve().parent / "retrieve.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "FIELD_TERMS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError("FIELD_TERMS not found in retrieve.py")


FIELD_TERMS = _load_field_terms()

STOP = set("what is are the of by in a an and or for to total does do how many much".split())


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def read_flat_ip(path):
    raw = Path(path).read_bytes()
    assert raw[:4] == b"IxFI", "only IndexFlatIP files are supported"
    d, n = struct.unpack("<iq", raw[4:16])
    count = struct.unpack("<Q", raw[37:45])[0]
    assert count == d * n
    return np.frombuffer(raw, dtype="<f4", count=count, offset=45).reshape(n, d)


class Company:
    def __init__(self, cid):
        idx = ROOT / "data" / "index"
        self.chunks = [json.loads(l) for l in open(idx / f"{cid}.chunks.jsonl", encoding="utf-8")]
        self.vecs = read_flat_ip(idx / f"{cid}.faiss")
        self.pages = np.array([c["page"] for c in self.chunks])
        self.tok = [tokenize(c["text"]) for c in self.chunks]
        self.df = Counter(t for ts in self.tok for t in set(ts))
        self.avg = sum(map(len, self.tok)) / len(self.tok)
        self.tf = [Counter(ts) for ts in self.tok]
        self.page_ids = sorted(set(self.pages.tolist()))


def bm25(company, terms, k1=1.2, b=0.75, weights=None):
    n = len(company.tok)
    out = np.zeros(n)
    for term in terms:
        df = company.df.get(term, 0)
        if not df:
            continue
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        w = 1.0 if weights is None else weights.get(term, 1.0)
        for i, tf in enumerate(company.tf):
            f = tf.get(term, 0)
            if f:
                out[i] += w * idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(company.tok[i]) / company.avg))
    return out


def rank_pages_rrf(company, dense, lex, k, depth=50, rrf_k=60, wd=1.0, wl=1.0):
    d_pos = np.argsort(-dense)[:depth]
    l_pos = np.argsort(-lex)[:depth]
    dr = {int(p): r for r, p in enumerate(d_pos, 1)}
    lr = {int(p): r for r, p in enumerate(l_pos, 1)}
    cand = set(dr) | set(lr)
    score = {p: wd / (rrf_k + dr.get(p, 10_000)) + wl / (rrf_k + lr.get(p, 10_000)) for p in cand}
    pages, seen = [], set()
    for p in sorted(cand, key=lambda x: -score[x]):
        pg = int(company.pages[p])
        if pg in seen:
            continue
        seen.add(pg)
        pages.append(pg)
        if len(pages) >= k:
            break
    return pages


def load():
    ann = pd.read_csv(ROOT / "data" / "annotations.csv", dtype=str).fillna("")
    qv = np.concatenate([np.load(p) for p in sorted(glob(str(ROOT / "results" / "query_vectors_*.npy")))])
    assert len(qv) == len(ann) == 320
    companies = {c: Company(c) for c in ann.company_id.unique()}
    return ann, qv, companies


def evaluate(ann, rows_pages, ks=(1, 3, 5)):
    res = {}
    for name, mask in {
        "answerable": ann.status.isin(["answered", "zero"]).values,
        "not_disc(elig)": (ann.status.eq("not_disclosed") & ann.label_type.ne("parts_only")).values,
    }.items():
        mask = mask & ann.page.str.isdigit().values
        res[name] = {}
        for k in ks:
            hits = [int(int(ann.page[i]) in rows_pages[i][:k]) for i in np.where(mask)[0]]
            res[name][k] = float(np.mean(hits))
        res[name]["n"] = int(mask.sum())
    return res


def run(ann, qv, companies, lex_query, k=5, **kw):
    rows = []
    for i, row in enumerate(ann.itertuples(index=False)):
        c = companies[row.company_id]
        dense = c.vecs @ qv[i]
        terms = lex_query(row)
        bm_kw = {x: kw[x] for x in ("k1", "b") if x in kw}
        lex = bm25(c, terms, **bm_kw)
        rrf_kw = {x: kw[x] for x in ("depth", "rrf_k", "wd", "wl") if x in kw}
        rows.append(rank_pages_rrf(c, dense, lex, k, **rrf_kw))
    return rows


def lex_baseline(row):
    return tokenize(f"{row.question} BRSR disclosure table {FIELD_TERMS[row.field_id]}")


if __name__ == "__main__":
    ann, qv, companies = load()
    pages = run(ann, qv, companies, lex_baseline)
    print(json.dumps(evaluate(ann, pages), indent=1))
