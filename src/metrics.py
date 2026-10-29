"""Scoring for BRSR-QA: accuracy, false-answer rate (FAR), unsupported-answer rate (UAR).

Conventions (these are the ones to state in the paper, Section V-B):
  * A numeric answer is correct if it is within 1% (relative) of the gold value after both are
    converted to a base unit (GJ, tCO2e, kL, metric tonnes).  Indian digit grouping (11,16,91,853),
    decimals and scale words (million, crore, lakh) are understood.
  * If the model's unit is missing or unrecognised, the number is accepted when it matches the gold
    value either in the unit printed in the report or in the base unit (strict_units=True turns this off).
  * Employee counts (F10, F11) and the optional intensity rows (F13-F16) are compared as plain numbers;
    units are not checked.  Yes/No (F12) and "not disclosed" must match exactly.
  * Accuracy is computed over the main-accuracy rows (acc_main_eligible: 279 rows, PARTS_ONLY excluded).
  * FAR = share of far_eligible gold-"not disclosed" rows (77, PARTS_ONLY excluded) where the system
    returned a value.  Unparseable replies are not counted as false answers; they are reported as the
    parse-error rate instead.
  * UAR = share of the system's numeric answers that do not appear anywhere in the retrieved context.
    Answers with 3+ significant digits are matched on those digits (grouping, decimal point and
    power-of-ten unit scaling are ignored); shorter numbers must match exactly.  Computed over every
    numeric answer the system gave, whatever the gold label.  Because retrieval can miss the value, even a
    perfect extractor has a non-zero UAR (see oracle_uar in check_stage3.py); compare systems against it.
"""
import json
import math
import re
from collections import defaultdict

import numpy as np

from parse_output import parse_model_output

TOLERANCE = 0.01

SCALE_WORDS = {
    "thousand": 1e3, "k": 1e3, "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5,
    "million": 1e6, "mn": 1e6, "mln": 1e6, "crore": 1e7, "crores": 1e7, "cr": 1e7,
    "billion": 1e9, "bn": 1e9,
}
_PREFIX_WORDS = {"kilo": 1e3, "thousand": 1e3, "million": 1e6, "mn": 1e6, "billion": 1e9,
                 "lakh": 1e5, "crore": 1e7}

FIELD_DIM = {
    "F01": "energy", "F02": "energy",
    "F03": "mass", "F04": "mass", "F05": "mass",
    "F06": "water", "F07": "water",
    "F08": "mass", "F09": "mass",
    "F10": "count", "F11": "count", "F12": "yesno",
    "F13": "intensity", "F14": "intensity", "F15": "intensity", "F16": "intensity",
}

# alias -> factor to the base unit of its dimension (energy GJ, mass tonnes, water kL)
_UNITS = {
    "energy": {
        "gj": 1, "gigajoule": 1, "gigajoules": 1, "giga joule": 1, "giga joules": 1,
        "mj": 1e-3, "megajoule": 1e-3, "megajoules": 1e-3, "tj": 1e3, "terajoule": 1e3,
        "terajoules": 1e3, "pj": 1e6, "petajoule": 1e6, "petajoules": 1e6,
        "kwh": 0.0036, "mwh": 3.6, "gwh": 3600, "kj": 1e-6, "joule": 1e-9, "joules": 1e-9,
    },
    "mass": {
        "tco2e": 1, "t co2e": 1, "tco2": 1, "tco2 eq": 1, "tco2eq": 1, "tonnes of co2 equivalent": 1,
        "tonnes co2 equivalent": 1, "tonnes of co2e": 1, "tonnes co2e": 1, "tonne co2e": 1,
        "metric tonnes of co2 equivalent": 1, "metric tonnes co2e": 1, "tonnes of co2": 1,
        "mtco2e": 1e6, "ktco2e": 1e3, "kg co2e": 1e-3,
        "mt": 1, "metric tonnes": 1, "metric tonne": 1, "metric tons": 1, "tonnes": 1, "tonne": 1,
        "tons": 1, "ton": 1, "t": 1, "kg": 1e-3, "kgs": 1e-3, "kilograms": 1e-3, "kt": 1e3, "mmt": 1e6,
    },
    "water": {
        "kl": 1, "kilolitre": 1, "kilolitres": 1, "kiloliter": 1, "kiloliters": 1, "m3": 1,
        "cubic metres": 1, "cubic meters": 1, "cubic metre": 1, "ml": 1e3, "megalitre": 1e3,
        "megalitres": 1e3, "megaliters": 1e3, "litres": 1e-3, "litre": 1e-3, "liters": 1e-3,
        "liter": 1e-3, "l": 1e-3,
    },
}


class Unit:
    def __init__(self, dimension, factor, has_prefix):
        self.dimension, self.factor, self.has_prefix = dimension, factor, has_prefix

    def __repr__(self):
        return f"Unit({self.dimension}, {self.factor:g}, prefix={self.has_prefix})"


def _norm_unit_text(text):
    text = str(text).lower().replace("₂", "2").replace("³", "3")
    text = re.sub(r"[().,*^#]", " ", text)
    text = re.sub(r"\bco\s*2\b", "co2", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_unit(text):
    """Return a Unit for strings such as 'Million GJ', 'kilo tonnes of CO2 equivalent', 'KL'."""
    if text is None:
        return None
    norm = _norm_unit_text(text)
    if not norm or "/" in norm:
        return None
    prefix, has_prefix = 1.0, False
    for _ in range(2):
        head, _, rest = norm.partition(" ")
        if head in _PREFIX_WORDS and rest:
            prefix *= _PREFIX_WORDS[head]
            has_prefix = True
            norm = rest
    for dimension, table in _UNITS.items():
        if norm in table:
            return Unit(dimension, table[norm] * prefix, has_prefix)
    return None


_NUM_RE = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?|\.\d+)")


def parse_number(value):
    """Parse a number from text or a numeric type.  Returns (number | None, scale_word_factor, remainder)."""
    if value is None or isinstance(value, bool):
        return None, 1.0, ""
    if isinstance(value, (int, float)):
        return (float(value), 1.0, "") if math.isfinite(value) else (None, 1.0, "")
    text = str(value).strip().lower().replace("₹", " ").replace("rs.", " ").replace("inr", " ")
    if text in {"nil", "zero"}:
        return 0.0, 1.0, ""
    match = _NUM_RE.search(text)
    if not match:
        return None, 1.0, ""
    token = match.group(1).replace(",", "")
    try:
        number = float(token)
    except ValueError:
        return None, 1.0, ""
    if match.start() > 0 and text[match.start() - 1] == "-" and (match.start() == 1 or text[match.start() - 2] == " "):
        number = -number
    rest = text[match.end():].strip()
    scale = 1.0
    scale_match = re.match(r"(thousand|lakhs?|lacs?|million|mn|mln|crores?|cr|billion|bn)\b\.?\s*(.*)$", rest)
    if scale_match:
        scale = SCALE_WORDS[scale_match.group(1)]
        rest = scale_match.group(2)
    return number, scale, rest.strip(" ,;:")


def parse_yes_no(value):
    if value is None:
        return None
    text = str(value).strip().strip(".!\"'").lower()
    if text in {"yes", "y", "true", "available", "exists"}:
        return "yes"
    if text in {"no", "n", "false"}:
        return "no"
    return None


def _close(candidate, gold, tol):
    if gold == 0:
        return abs(candidate) <= 1e-9
    return abs(candidate - gold) <= tol * abs(gold)


def _compatible(dim_a, dim_b):
    return dim_a == dim_b


def numeric_match(answer, unit, gold_value, gold_unit, field_id, tol=TOLERANCE, strict_units=False):
    """True if ``answer``/``unit`` equals the gold value within ``tol`` after unit conversion."""
    gold, _, _ = parse_number(gold_value)
    if gold is None:
        raise ValueError(f"Unparseable gold value: {gold_value!r}")
    number, text_scale, remainder = parse_number(answer)
    if number is None:
        return False
    dim = FIELD_DIM.get(field_id, "mass")
    if dim in {"count", "intensity"}:
        return _close(number * text_scale, gold, tol)

    gold_unit_obj = parse_unit(gold_unit)
    if gold_unit_obj is None:
        raise ValueError(f"Unrecognised gold unit: {gold_unit!r} ({field_id})")
    gold_base = gold * gold_unit_obj.factor

    pred_unit = parse_unit(unit) or parse_unit(remainder)
    if pred_unit is None:
        if strict_units:
            return False
        candidates = [number * text_scale * gold_unit_obj.factor, number * text_scale]
    else:
        if not _compatible(pred_unit.dimension, gold_unit_obj.dimension):
            return False
        scale = 1.0 if pred_unit.has_prefix else text_scale
        candidates = [number * scale * pred_unit.factor]
    return any(_close(c, gold_base, tol) for c in candidates)


def significant_digits(text):
    digits = re.sub(r"[,.]", "", str(text))
    digits = re.sub(r"\D", "", digits)
    return digits.strip("0")


def context_numbers(context):
    """(significant-digit strings, exact float values) of every number printed in ``context``."""
    flat = re.sub(r"(?<=\d),(?=\d)", "", context or "")
    tokens = re.findall(r"\d+(?:\.\d+)?", flat)
    return {significant_digits(t) for t in tokens} - {""}, {float(t) for t in tokens}


def _rounded_variants(token):
    """The printed number ``token`` (e.g. '916465.19') rounded or truncated to fewer decimals."""
    from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
    if "." not in token:
        return {}
    value = Decimal(token)
    decimals = len(token.split(".")[1])
    out = {}
    for places in range(decimals):
        quantum = Decimal(1).scaleb(-places)
        for mode in (ROUND_HALF_UP, ROUND_DOWN):
            out.setdefault(places, set()).add(value.quantize(quantum, rounding=mode))
    return out


def _rounded_match(clean, context):
    """True if ``clean`` is a printed context number rounded/truncated to ``clean``'s own decimals."""
    from decimal import Decimal
    places = len(clean.split(".")[1]) if "." in clean else 0
    target = Decimal(clean)
    flat = re.sub(r"(?<=\d),(?=\d)", "", context or "")
    for token in set(re.findall(r"\d+\.\d+", flat)):
        if target in _rounded_variants(token).get(places, ()):
            return True
    return False


def is_grounded(answer, context):
    """True if the answer's number is printed in ``context``.

    Numbers with at least three significant digits are matched on those digits, so grouping style,
    decimal-point position and power-of-ten unit scaling (254 vs 254,000,000) do not matter.  Such a
    number is also grounded when it is a printed number rounded or truncated to fewer decimals
    (916,465 for a printed 916,465.19).  Shorter numbers must match exactly, otherwise 0.5 would be
    'found' in 500.
    """
    number, _, _ = parse_number(answer)
    if number is None:
        return None
    if number == 0:
        return True
    sigs, values = context_numbers(context)
    tokens = re.findall(r"\d[\d,]*(?:\.\d+)?", str(answer)) or [f"{number:.10f}".rstrip("0").rstrip(".")]
    for token in tokens:
        clean = token.replace(",", "")
        digits = significant_digits(clean)
        if len(digits) >= 3:
            if digits in sigs or _rounded_match(clean, context):
                return True
        elif float(clean) in values:
            return True
    return False


def value_in_context(gold_value, context):
    """True if the gold number is printed in ``context`` exactly (comma grouping ignored)."""
    target = re.sub(r"[^0-9.]", "", str(gold_value).replace(",", ""))
    if not target:
        return False
    flat = re.sub(r"(?<=\d),(?=\d)", "", context or "")
    return re.search(r"(?<![\d.])" + re.escape(target) + r"(?!\d)", flat) is not None


def context_text(retrieved):
    """Concatenate retrieved chunks (list of dicts with 'text', or one string)."""
    if isinstance(retrieved, str):
        return retrieved
    return "\n".join(item["text"] if isinstance(item, dict) else str(item) for item in retrieved or [])


def score_prediction(row, parsed, context="", tol=TOLERANCE, strict_units=False):
    """Score one parsed prediction against one annotation row (mapping with status, field_id, ...)."""
    status = row["status"]
    kind = parsed["kind"]
    field_id = row["field_id"]
    correct = False
    if status == "not_disclosed":
        correct = kind == "not_disclosed"
    elif kind == "value":
        if field_id == "F12":
            correct = parse_yes_no(parsed["answer"]) == parse_yes_no(row["gold_value"])
        else:
            correct = numeric_match(parsed["answer"], parsed["unit"], row["gold_value"], row["gold_unit"],
                                    field_id, tol, strict_units)
    grounded = None
    if kind == "value" and field_id != "F12":
        grounded = is_grounded(parsed["answer"], context)
    return {
        "correct": bool(correct),
        "pred_kind": kind,
        "false_answer": bool(status == "not_disclosed" and kind == "value"),
        "grounded": grounded,
        "format_ok": bool(parsed.get("format_ok")),
    }


def _flag(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def summarize(scored_rows):
    """scored_rows: list of dicts holding annotation fields plus the output of score_prediction()."""
    acc_rows = [r for r in scored_rows if _flag(r.get("acc_main_eligible", True))]
    far_rows = [r for r in scored_rows if _flag(r.get("far_eligible", False))]
    value_rows = [r for r in scored_rows if r["grounded"] is not None]
    answerable = [r for r in acc_rows if r["status"] in {"answered", "zero"}]
    unanswerable = [r for r in acc_rows if r["status"] == "not_disclosed"]

    def mean(rows, key):
        return float(np.mean([bool(r[key]) for r in rows])) if rows else float("nan")

    by_field = defaultdict(list)
    for r in acc_rows:
        by_field[r["field_id"]].append(r["correct"])
    by_type = defaultdict(list)
    for r in far_rows:
        by_type[r.get("label_type", "?")].append(r["false_answer"])
    return {
        "n_scored": len(scored_rows),
        "n_accuracy": len(acc_rows),
        "accuracy": mean(acc_rows, "correct"),
        "accuracy_answerable": mean(answerable, "correct"),
        "accuracy_not_disclosed": mean(unanswerable, "correct"),
        "n_far": len(far_rows),
        "far": mean(far_rows, "false_answer"),
        "far_by_type": {k: float(np.mean(v)) for k, v in sorted(by_type.items())},
        "n_uar": len(value_rows),
        "uar": float(np.mean([not r["grounded"] for r in value_rows])) if value_rows else float("nan"),
        "parse_error_rate": float(np.mean([r["pred_kind"] == "parse_error" for r in scored_rows])),
        "format_failure_rate": float(np.mean([not r["format_ok"] for r in scored_rows])),
        "accuracy_by_field": {k: float(np.mean(v)) for k, v in sorted(by_field.items())},
    }


def evaluate_jsonl(predictions_path, annotations, system=None, strict_units=False):
    """Score a predictions JSONL (system, fold, question_id, retrieved_chunks, raw_output[, parsed_answer])."""
    rows_by_id = {r["row_id"]: r for r in annotations.to_dict("records")}
    scored = []
    with open(predictions_path, encoding="utf-8") as stream:
        for line in stream:
            rec = json.loads(line)
            if system and rec.get("system") != system:
                continue
            row = rows_by_id[rec["question_id"]]
            parsed = rec.get("parsed_answer") or parse_model_output(rec.get("raw_output"))
            result = score_prediction(row, parsed, context_text(rec.get("retrieved_chunks")), strict_units=strict_units)
            scored.append({**row, **result, "system": rec.get("system"), "fold": rec.get("fold", row.get("fold"))})
    return scored
