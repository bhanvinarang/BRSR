"""Prompt templates for the three systems (S1 zero-shot, S2 few-shot, S3 fine-tuned).

All three use the same instruction and the same output format; S2 only adds three solved examples
(drawn from training companies) and S3 uses the S1 prompt with a fine-tuned model.

    messages = build_messages(question, chunks)                      # S1 / S3
    messages = build_messages(question, chunks, examples=examples)   # S2
    tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

``chunks`` is the list of dicts returned by CompanyRetriever.retrieve() (needs 'page' and 'text').
"""
import json
from pathlib import Path

# Verbatim wording from the paper (Section IV-A).
INSTRUCTION_MINIMAL = (
    "Answer using only the context. Reply in JSON as {\"answer\": ..., \"unit\": ...}. "
    "If the context does not contain the answer, reply {\"answer\": \"not disclosed\"}."
)

# Default: the same instruction plus the annotation rules the gold labels follow.  If you use this,
# paste it into Section IV-A in place of the one-sentence prompt.
INSTRUCTION_GUIDED = (
    "You extract disclosures from an Indian company's Business Responsibility and Sustainability "
    "Report (BRSR). Answer using only the context. "
    "Reply with JSON only, in the form {\"answer\": ..., \"unit\": ...}. "
    "Copy the number exactly as printed for the current financial year (the first value column, "
    "not the previous year) and give the unit as printed. "
    "For yes/no questions answer \"Yes\" or \"No\". "
    "A printed 0 or \"Nil\" is a value. "
    "If the context does not contain the answer (the row is missing, or the value is blank, \"NA\" "
    "or \"-\"), or only the components of a total are printed, reply {\"answer\": \"not disclosed\"}. "
    "Do not add up or calculate values."
)

INSTRUCTIONS = {"minimal": INSTRUCTION_MINIMAL, "guided": INSTRUCTION_GUIDED}
DEFAULT_VARIANT = "guided"

_NO_UNIT = {"", "not stated", "number", "yes/no"}


def format_context(chunks):
    """Render retrieved chunks, best first, each tagged with its report page."""
    return "\n\n".join(f"[Page {c['page']}]\n{c['text'].strip()}" for c in chunks)


def gold_to_target(row):
    """The reply the model should give for an annotation row (used for few-shot examples and S3 data)."""
    if row["status"] == "not_disclosed":
        return {"answer": "not disclosed"}
    if row["field_id"] == "F12":
        return {"answer": "Yes" if str(row["gold_value"]).strip().lower().startswith("y") else "No"}
    unit = None if str(row["gold_unit"]).strip().lower() in _NO_UNIT else row["gold_unit"]
    return {"answer": row["gold_value"], "unit": unit}


def target_json(target):
    return json.dumps(target, ensure_ascii=False)


def _example_block(index, example):
    return (
        f"Example {index}\n"
        f"Context:\n{example['context'].strip()}\n"
        f"Question: {example['question']}\n"
        f"Answer: {target_json(example['target'])}"
    )


def build_user_message(question, chunks, examples=None):
    parts = []
    if examples:
        parts.append("Here are solved examples of the task.\n\n" +
                     "\n\n".join(_example_block(i, ex) for i, ex in enumerate(examples, 1)) +
                     "\n\nNow answer the new question in the same way.")
    parts.append(f"Context:\n{format_context(chunks)}")
    parts.append(f"Question: {question}")
    parts.append("Answer (JSON only):")
    return "\n\n".join(parts)


def build_messages(question, chunks, examples=None, variant=DEFAULT_VARIANT):
    return [
        {"role": "system", "content": INSTRUCTIONS[variant]},
        {"role": "user", "content": build_user_message(question, chunks, examples)},
    ]


def load_fewshot(fold, path=None):
    """Few-shot examples for a test fold; they come only from companies outside that fold."""
    path = Path(path) if path else Path(__file__).resolve().parent.parent / "data" / "fewshot_examples.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    examples = data[str(fold)]
    folds = json.loads((path.parent / "folds.json").read_text(encoding="utf-8"))
    leaked = [e["company_id"] for e in examples if int(folds[e["company_id"]]) == int(fold)]
    assert not leaked, f"few-shot example from a test-fold company: {leaked}"
    return examples
