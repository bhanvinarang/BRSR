"""Stage 7: error analysis.  Tags wrong answers into five categories and builds tagging sheets for four people.

    python src/error_analysis.py            (from the BRSR folder)

Categories (as in the pipeline):  1 Retrieval miss | 2 Extraction error | 3 Unit/scale error |
                                  4 Invented number | 5 Wrongly "not disclosed"
The automatic tags are heuristics used to pre-sort and to count; the paper's per-category figures should come from
the human tags in the sampled sheets.  The tagging sheets given to people do NOT show the automatic tag.

Rules (applied to every accuracy-eligible question the system got wrong):
  retrieval miss          the gold answer is not in the retrieved text (value not printed there, gold page not retrieved)
  wrongly not disclosed   the answer was in the retrieved text, but the system said "not disclosed"
  unit/scale error        a number with the gold digits (or a power-of-ten / kWh-GJ multiple of the gold number)
  invented number         the system's number is not printed anywhere in the retrieved text
  extraction error        any other wrong value taken from the retrieved text (wrong row/column/year, unparseable output),
                          and numbers given for a question whose truth is "not disclosed" that are printed in the text
Questions whose truth is "not disclosed" and that were answered with a number are 'invented' if the number is not in the
retrieved text, otherwise 'extraction'.
"""
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import _flag, context_text, evaluate_jsonl, parse_number, value_in_context  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "error_analysis"
SYSTEMS = ["S1", "S2", "S3"]
SEED, PER_SYSTEM = 42, 20
CATEGORIES = ["Retrieval miss", "Extraction error", "Unit/scale error", "Invented number", 'Wrongly "not disclosed"']
RATIOS = [3.6, 1 / 3.6, 3.6e3, 1 / 3.6e3, 3.6e6, 1 / 3.6e6, 277.7778, 1 / 277.7778]


def scale_confusion(pred_answer, gold_value):
    a, scale_a, _ = parse_number(pred_answer)
    g, scale_g, _ = parse_number(gold_value)
    if a is None or g in (None, 0) or a == 0:
        return False
    a, g = a * scale_a, g * scale_g
    digits = lambda x: "".join(ch for ch in f"{abs(x):.6g}" if ch.isdigit()).strip("0")[:4]
    if digits(a) and digits(a) == digits(g):
        return True
    ratio = abs(a / g)
    if any(abs(ratio / r - 1) < 0.02 for r in RATIOS):
        return True
    exp = math.log10(ratio)
    return abs(exp - round(exp)) < 0.01 and round(exp) != 0


def subtag(row, parsed, tag):
    """Finer split of the big 'Extraction error' bucket (reported alongside, not instead of, the five categories)."""
    if tag != "Extraction error":
        return ""
    if row["status"] == "not_disclosed":
        return "number given although the true answer is not disclosed"
    if parsed["kind"] == "parse_error":
        return "unparseable output"
    unit = (parsed.get("unit") or "").strip().lower()
    if row["field_id"] == "F11" and ("%" in unit or "percent" in unit or "%" in str(parsed.get("answer"))):
        return "percentage given instead of a count"
    return "wrong number for an answerable question"


def classify(row, rec):
    parsed = rec["parsed_answer"]
    chunks = rec["retrieved_chunks"]
    context = context_text(chunks)
    pages = {int(c["page"]) for c in chunks}
    gold_page = int(row["page"]) if str(row["page"]).isdigit() else None
    page_hit = gold_page in pages if gold_page is not None else False
    if row["status"] == "not_disclosed":
        return "Invented number" if row["grounded"] is False else "Extraction error", page_hit, False
    if row["field_id"] == "F12" or row["status"] == "zero":
        in_context = page_hit
    else:
        in_context = value_in_context(row["gold_value"], context) or page_hit
    value_hit = value_in_context(row["gold_value"], context) if row["field_id"] != "F12" else page_hit
    if not in_context:
        return "Retrieval miss", page_hit, value_hit
    if parsed["kind"] == "not_disclosed":
        return 'Wrongly "not disclosed"', page_hit, value_hit
    if parsed["kind"] == "parse_error":
        return "Extraction error", page_hit, value_hit
    if row["field_id"] != "F12" and scale_confusion(parsed["answer"], row["gold_value"]):
        return "Unit/scale error", page_hit, value_hit
    if row["grounded"] is False:
        return "Invented number", page_hit, value_hit
    return "Extraction error", page_hit, value_hit


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    annotations = pd.read_csv(ROOT / "data" / "annotations.csv", dtype=str).fillna("")
    wrong = []
    for system in SYSTEMS:
        path = ROOT / "results" / f"predictions_{system}_k5.jsonl"
        recs = {r["question_id"]: r for r in map(json.loads, open(path, encoding="utf-8"))}
        for row in evaluate_jsonl(path, annotations, system=system):
            if not _flag(row["acc_main_eligible"]) or row["correct"]:
                continue
            rec = recs[row["row_id"]]
            tag, page_hit, value_hit = classify(row, rec)
            parsed = rec["parsed_answer"]
            wrong.append({
                "system": system, "row_id": row["row_id"], "company": row["company_id"], "fold": row["fold"],
                "field_id": row["field_id"], "field": row["field_name"], "question": row["question"],
                "gold_status": row["status"], "gold_value": row["gold_value"], "gold_unit": row["gold_unit"],
                "gold_page": row["page"], "gold_annotator": row["annotator"],
                "system_kind": parsed["kind"], "system_answer": parsed.get("answer"), "system_unit": parsed.get("unit"),
                "raw_output": rec["raw_output"], "retrieved_pages": " ".join(str(c["page"]) for c in rec["retrieved_chunks"]),
                "gold_page_retrieved": page_hit, "gold_value_in_context": value_hit, "auto_tag": tag,
                "auto_subtag": subtag(row, parsed, tag)})
    df = pd.DataFrame(wrong)
    df.to_csv(OUT / "all_wrong_answers_auto_tagged.csv", index=False)

    md = ["# Stage 7: error analysis", "", "## Automatic tag counts over ALL wrong answers (heuristic)", "",
          "| Category | " + " | ".join(SYSTEMS) + " |", "|---|" + "---|" * len(SYSTEMS)]
    counts = {s: Counter(df[df.system == s].auto_tag) for s in SYSTEMS}
    for cat in CATEGORIES:
        md.append(f"| {cat} | " + " | ".join(str(counts[s][cat]) for s in SYSTEMS) + " |")
    md.append("| **total wrong** | " + " | ".join(f"**{sum(counts[s].values())}**" for s in SYSTEMS) + " |")

    md += ["", "## Split of the 'Extraction error' bucket (automatic)", "",
           "| Sub-type | " + " | ".join(SYSTEMS) + " |", "|---|" + "---|" * len(SYSTEMS)]
    for sub in sorted(set(df.auto_subtag) - {""}):
        md.append(f"| {sub} | " + " | ".join(str(int(((df.system == s_) & (df.auto_subtag == sub)).sum())) for s_ in SYSTEMS) + " |")

    # --- sample 20 wrong answers per system, assign each to someone other than the gold annotator
    rng = random.Random(SEED)
    sample = pd.concat([df[df.system == s].sample(n=min(PER_SYSTEM, (df.system == s).sum()), random_state=SEED)
                        for s in SYSTEMS]).reset_index(drop=True)
    # Two taggers, A and B.  A is assumed to be annotator P1 and B annotator P2, so nobody tags a row they
    # annotated themselves (rows annotated by P3 can go to either).  Swap the file names if it is the other way round.
    people = ["A", "B"]
    own = {"P1": "A", "P2": "B"}
    load = Counter()
    order = list(sample.index)
    rng.shuffle(order)
    assigned = {}
    for i in order:
        allowed = [p for p in people if p != own.get(sample.loc[i, "gold_annotator"])] or people
        pick = min(allowed, key=lambda p: (load[p], rng.random()))
        assigned[i] = pick
        load[pick] += 1
    sample["sample_id"] = [f"E{n + 1:02d}" for n in range(len(sample))]
    sample["assigned_to"] = pd.Series(assigned)
    sample["human_tag"] = ""
    sample["human_notes"] = ""
    sample.to_csv(OUT / "error_sample_all_with_auto_tags.csv", index=False)
    blind_cols = ["sample_id", "assigned_to", "system", "company", "field_id", "field", "question", "gold_status",
                  "gold_value", "gold_unit", "gold_page", "system_kind", "system_answer", "system_unit", "raw_output",
                  "retrieved_pages", "gold_page_retrieved", "gold_value_in_context", "human_tag", "human_notes"]
    for person in people:
        sample[sample.assigned_to == person][blind_cols].to_csv(OUT / f"tagging_sheet_{person}.csv", index=False)

    md += ["", f"## Sample for human tagging: {PER_SYSTEM} wrong answers per system, split across {len(people)} people", "",
           "Assignments (nobody tags a row they annotated): " + ", ".join(f"{p}: {load[p]}" for p in people), "",
           "Valid values for `human_tag`: " + " | ".join(CATEGORIES), "",
           "## Automatic tags within the 20-per-system sample (people will confirm or correct these)", "",
           "| Category | " + " | ".join(SYSTEMS) + " |", "|---|" + "---|" * len(SYSTEMS)]
    sc = {s: Counter(sample[sample.system == s].auto_tag) for s in SYSTEMS}
    for cat in CATEGORIES:
        md.append(f"| {cat} | " + " | ".join(str(sc[s][cat]) for s in SYSTEMS) + " |")

    # --- illustrative examples (automatic tags, to be confirmed by people)
    md += ["", "## Candidate illustrative examples per category (confirm by hand before using)", ""]
    for cat in CATEGORIES:
        hits = df[df.auto_tag == cat]
        md.append(f"### {cat} ({len(hits)} wrong answers)")
        for _, r in hits.sample(n=min(2, len(hits)), random_state=SEED).iterrows():
            md.append(f"- {r.system} / {r.row_id}: \"{r.question}\" gold {r.gold_value} {r.gold_unit} (page {r.gold_page}); "
                      f"system said {r.system_answer} {r.system_unit or ''}; retrieved pages {r.retrieved_pages}")
        md.append("")

    # --- retrieval context for the error counts
    md += ["## Where retrieval stood on the wrong answers", "", "| System | wrong | gold page not retrieved | gold value not printed in retrieved text |",
           "|---|---|---|---|"]
    for s in SYSTEMS:
        part = df[(df.system == s) & (df.gold_status != "not_disclosed")]
        md.append(f"| {s} | {len(df[df.system == s])} | {int((~part.gold_page_retrieved.astype(bool)).sum())} of {len(part)} answerable | "
                  f"{int((~part.gold_value_in_context.astype(bool)).sum())} of {len(part)} answerable |")
    md.append("")
    (OUT / "error_analysis_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
