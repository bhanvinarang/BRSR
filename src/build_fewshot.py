"""Pick three short solved examples per test fold, from companies outside that fold only.

Each example is a ~400-character excerpt of the report page around the gold value (or around an
"NA" optional row), the original question, and the target JSON.  Output: data/fewshot_examples.json
(read it before use; the selection is seeded and deterministic).

    python src/build_fewshot.py
"""
import json
import random
import re
from pathlib import Path

import pandas as pd

from prompts import gold_to_target

ROOT = Path(__file__).resolve().parent.parent
STANDARD_UNITS = {"GJ", "tCO2e", "kL", "MT"}


def load_pages():
    pages = {}
    for line in open(ROOT / "data" / "corpus" / "pages.jsonl", encoding="utf-8"):
        rec = json.loads(line)
        pages[(rec["company_id"], rec["page"])] = rec["text"]
    return pages


def excerpt(text, start, end, before=300, after=100):
    lo, hi = max(0, start - before), min(len(text), end + after)
    snippet = text[lo:hi]
    if lo > 0 and "\n" in snippet:
        snippet = snippet.split("\n", 1)[1]            # drop the partial first line
    if hi < len(text) and "\n" in snippet:
        snippet = snippet.rsplit("\n", 1)[0]           # drop the partial last line
    snippet = re.sub(r"[ \t]+", " ", snippet)
    return re.sub(r"\n\s*\n+", "\n", snippet).strip()


def numeric_candidates(rows, pages, fields):
    out = []
    for r in rows.itertuples(index=False):
        if r.status != "answered" or r.field_id not in fields or not str(r.page).isdigit():
            continue
        if fields == {"F10"} and False:
            continue
        if r.field_id not in {"F10", "F11"} and r.gold_unit not in STANDARD_UNITS:
            continue
        text = pages.get((r.company_id, int(r.page)), "")
        i = text.find(r.gold_value)
        if i < 0 or text.count(r.gold_value) > 2:
            continue
        ex = excerpt(text, i, i + len(r.gold_value))
        if r.gold_value in ex and 150 <= len(ex) <= 520:
            out.append((r, ex))
    return out


def blank_candidates(rows, pages):
    out = []
    for r in rows.itertuples(index=False):
        if r.label_type != "printed_blank" or not str(r.page).isdigit():
            continue
        text = pages.get((r.company_id, int(r.page)), "")
        for m in re.finditer(r"\(optional\)[^\n]*(?:\n[^\n]*){0,2}?\n\s*(NA|N/A|-|–)\s*\n", text, re.I):
            ex = excerpt(text, m.start(), m.end(), before=260, after=60)
            if 120 <= len(ex) <= 520:
                out.append((r, ex))
                break
    return out


def main():
    annotations = pd.read_csv(ROOT / "data" / "annotations.csv", dtype=str).fillna("")
    folds = json.loads((ROOT / "data" / "folds.json").read_text(encoding="utf-8"))
    pages = load_pages()
    result = {}
    for fold in range(4):
        train = annotations[annotations.company_id.map(lambda c: int(folds[c]) != fold)]
        rng = random.Random(1000 + fold)
        picks = []
        used = set()
        for kind, fields in (("energy_emissions", {"F01", "F03", "F04"}), ("blank", None), ("count", {"F10", "F11"})):
            cands = blank_candidates(train, pages) if kind == "blank" else numeric_candidates(train, pages, fields)
            cands = [c for c in cands if c[0].company_id not in used]
            if not cands:
                raise RuntimeError(f"no {kind} candidates for fold {fold}")
            row, ex = rng.choice(cands)
            used.add(row.company_id)
            picks.append({
                "company_id": row.company_id, "row_id": row.row_id, "field_id": row.field_id,
                "question": row.question, "context": ex, "target": gold_to_target(row._asdict()),
            })
        order = list(range(len(picks)))
        rng.shuffle(order)
        result[str(fold)] = [picks[i] for i in order]
    out = ROOT / "data" / "fewshot_examples.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
