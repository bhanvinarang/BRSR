"""Robust parsing of a model's reply into a structured answer.

The prompt asks for {"answer": ..., "unit": ...} or {"answer": "not disclosed"}, but a 1.5B model
sometimes adds prose, code fences, single quotes or trailing commas.  parse_model_output() accepts
all of those and reports whether the reply followed the requested format (``format_ok``), so the
paper can report a format-failure rate separately from accuracy.

Returned dict:
    kind       "value" | "not_disclosed" | "parse_error"
    answer     the raw answer (str / number) or None
    unit       the raw unit string or None
    format_ok  True only when a JSON object with an "answer" key was found
"""
import ast
import json
import re

_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.S)
_NOT_DISCLOSED = re.compile(
    r"^(not[\s_\-]*disclosed|undisclosed|not\s+(available|provided|reported|mentioned|stated|found|specified)"
    r"|n/?a|na|none|null|unknown|no\s+data|-+|—|–|)$",
    re.I,
)
_EMPTY_UNIT = {"", "null", "none", "n/a", "na", "-", "not stated"}


def _strip_fences(text):
    match = _FENCE.search(text)
    return match.group(1) if match else text


def _json_candidates(text):
    """Yield every dict that can be decoded starting at a '{' in ``text``."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        chunk = text[match.start():]
        try:
            value, _ = decoder.raw_decode(chunk)
            if isinstance(value, dict):
                yield value
                continue
        except ValueError:
            pass
        end = chunk.find("}")
        if end == -1:
            continue
        snippet = chunk[: end + 1]
        repaired = re.sub(r",\s*}", "}", snippet)
        for loader in (json.loads, ast.literal_eval):
            try:
                value = loader(repaired)
            except (ValueError, SyntaxError):
                continue
            if isinstance(value, dict):
                yield value
                break


def _pick(dicts):
    dicts = list(dicts)
    for d in dicts:
        keys = {str(k).lower(): k for k in d}
        if "answer" in keys:
            return {str(k).lower(): v for k, v in d.items()}
    return None


def _classify(answer, unit, format_ok):
    if isinstance(answer, bool):
        answer = "Yes" if answer else "No"
    if isinstance(unit, str) and unit.strip().lower() in _EMPTY_UNIT:
        unit = None
    if isinstance(unit, (int, float)):
        unit = str(unit)
    if answer is None:
        return {"kind": "not_disclosed", "answer": None, "unit": unit, "format_ok": format_ok}
    if isinstance(answer, (list, dict)):
        return {"kind": "parse_error", "answer": answer, "unit": unit, "format_ok": False}
    if isinstance(answer, str):
        cleaned = answer.strip().strip("\"'").strip()
        if _NOT_DISCLOSED.match(cleaned) or re.search(r"not[\s_\-]*disclosed", cleaned, re.I):
            return {"kind": "not_disclosed", "answer": cleaned, "unit": unit, "format_ok": format_ok}
        answer = cleaned
    return {"kind": "value", "answer": answer, "unit": unit, "format_ok": format_ok}


def parse_model_output(raw):
    text = "" if raw is None else str(raw)
    body = _strip_fences(text).strip()
    parsed = _pick(_json_candidates(body))
    if parsed is not None:
        return _classify(parsed.get("answer"), parsed.get("unit"), True)

    # Fallbacks: "answer": ... without braces, a bare "not disclosed", or a bare short number.
    match = re.search(r"[\"']?answer[\"']?\s*[:=]\s*(\"([^\"]*)\"|'([^']*)'|[^,}\n]+)", body, re.I)
    if match:
        answer = next(g for g in (match.group(2), match.group(3), match.group(1)) if g is not None)
        unit_match = re.search(r"[\"']?unit[\"']?\s*[:=]\s*(\"([^\"]*)\"|'([^']*)'|[^,}\n]+)", body, re.I)
        unit = None
        if unit_match:
            unit = next(g for g in (unit_match.group(2), unit_match.group(3), unit_match.group(1)) if g is not None)
        return _classify(answer.strip(), unit, False)
    if re.search(r"not[\s_\-]*disclosed", body, re.I):
        return {"kind": "not_disclosed", "answer": None, "unit": None, "format_ok": False}
    if body and len(body) <= 40 and re.search(r"\d", body):
        return {"kind": "value", "answer": body, "unit": None, "format_ok": False}
    if re.fullmatch(r"(?i)\s*(yes|no)\W*", body):
        return {"kind": "value", "answer": body.strip(" .!"), "unit": None, "format_ok": False}
    return {"kind": "parse_error", "answer": None, "unit": None, "format_ok": False}
