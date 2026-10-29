"""Hand-made tests for the parser and metrics, plus a gold-answer self-consistency check.

Run from the project root:   python tests/test_stage3.py      (or: pytest tests)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from metrics import (is_grounded, numeric_match, parse_number, parse_unit, parse_yes_no,  # noqa: E402
                     score_prediction, summarize)
from parse_output import parse_model_output  # noqa: E402
from prompts import build_messages, gold_to_target, load_fewshot  # noqa: E402


# ---------------------------------------------------------------- parser
def test_parse_clean_json():
    p = parse_model_output('{"answer": "20,55,520", "unit": "GJ"}')
    assert p == {"kind": "value", "answer": "20,55,520", "unit": "GJ", "format_ok": True}


def test_parse_fenced_with_prose():
    p = parse_model_output('Sure! Here you go:\n```json\n{"answer": "36,726", "unit": null}\n```\nHope it helps.')
    assert p["kind"] == "value" and p["answer"] == "36,726" and p["unit"] is None and p["format_ok"]


def test_parse_single_quotes_and_trailing_comma():
    assert parse_model_output("{'answer': '8,944', 'unit': 'tCO2e'}")["answer"] == "8,944"
    assert parse_model_output('{"answer": "8,944", "unit": "tCO2e",}')["answer"] == "8,944"


def test_parse_not_disclosed_variants():
    for raw in ['{"answer": "not disclosed"}', '{"answer": "Not Disclosed."}', '{"answer": null}',
                '{"answer": "NA"}', '{"answer": "-"}', '{"answer": ""}', 'not disclosed']:
        assert parse_model_output(raw)["kind"] == "not_disclosed", raw


def test_parse_numeric_json_and_nil():
    assert parse_model_output('{"answer": 1234.5, "unit": "kL"}')["answer"] == 1234.5
    assert parse_model_output('{"answer": "Nil"}')["kind"] == "value"        # zero is a value
    assert parse_number("Nil")[0] == 0.0


def test_parse_garbage_and_fallback():
    assert parse_model_output("I could not find anything relevant, sorry")["kind"] == "parse_error"
    assert parse_model_output("")["kind"] == "parse_error"
    fallback = parse_model_output('answer: "6,36,833", unit: "number"')
    assert fallback["kind"] == "value" and not fallback["format_ok"]


# ---------------------------------------------------------------- numbers and units
def test_indian_and_western_grouping():
    assert parse_number("11,16,91,853")[0] == 111691853
    assert parse_number("1,11,707.5")[0] == 111707.5
    assert parse_number("6,02,58,642.49")[0] == 60258642.49
    assert parse_number("1,116,91,853.0")[0] == 111691853.0


def test_scale_words():
    assert parse_number("2.5 crore")[:2] == (2.5, 1e7)
    assert parse_number("3 lakh")[:2] == (3.0, 1e5)
    assert parse_number("254 million GJ")[:3] == (254.0, 1e6, "gj")


def test_unit_parsing():
    assert parse_unit("Million GJ").factor == 1e6
    assert parse_unit("kilo tonnes of CO2 equivalent").factor == 1e3
    assert parse_unit("Million Litres").factor == 1e3          # kL per million litres
    assert parse_unit("tCO₂e").dimension == "mass"
    assert parse_unit("GJ/MVA") is None


def test_tolerance_one_percent():
    assert numeric_match("1,009", "number", "1,000", "number", "F10")
    assert not numeric_match("1,011", "number", "1,000", "number", "F10")


def test_energy_scale_conversion():
    gold = ("254", "Million GJ", "F01")
    assert numeric_match("254000000", "GJ", *gold)
    assert numeric_match("254", "Million GJ", *gold)
    assert numeric_match("254 million", "GJ", *gold)
    assert not numeric_match("254", "GJ", *gold)            # a stated unit that disagrees is wrong
    assert numeric_match("254", None, *gold)                # unit omitted: accepted (lenient)
    assert not numeric_match("254", None, *gold, strict_units=True)
    assert numeric_match("2500", "GJ", "2.5", "TJ", "F01")


def test_water_and_wrong_dimension():
    assert numeric_match("389691000", "KL", "3,89,691", "Million Litres", "F06")
    assert not numeric_match("389691", "kL", "254", "Million GJ", "F01")


def test_zero_and_yes_no_and_intensity():
    assert numeric_match("0", "kL", "0", "kL", "F07") and numeric_match("Nil", None, "0", "kL", "F07")
    assert not numeric_match("5", "kL", "0", "kL", "F07")
    assert parse_yes_no("Yes.") == "yes" and parse_yes_no("no") == "no"
    assert numeric_match("2.99", None, "2.99", "GJ/MVA", "F13")
    assert not numeric_match("3.5", None, "2.99", "GJ/MVA", "F13")


# ---------------------------------------------------------------- scoring, FAR, UAR
ROW_NUM = {"status": "answered", "field_id": "F01", "gold_value": "20,55,520", "gold_unit": "GJ"}
ROW_ND = {"status": "not_disclosed", "field_id": "F15", "gold_value": "", "gold_unit": ""}
ROW_ZERO = {"status": "zero", "field_id": "F07", "gold_value": "0", "gold_unit": "kL"}


def test_score_correct_wrong_and_abstain():
    ctx = "Total energy consumed (A+B+C+D+E+F) GJ 20,55,520 20,18,011"
    assert score_prediction(ROW_NUM, parse_model_output('{"answer":"2055520","unit":"GJ"}'), ctx)["correct"]
    assert not score_prediction(ROW_NUM, parse_model_output('{"answer":"2018011","unit":"GJ"}'), ctx)["correct"]
    assert not score_prediction(ROW_NUM, parse_model_output('{"answer":"not disclosed"}'), ctx)["correct"]
    assert not score_prediction(ROW_ZERO, parse_model_output('{"answer":"not disclosed"}'), "")["correct"]


def test_false_answer_only_when_value_returned():
    assert score_prediction(ROW_ND, parse_model_output('{"answer":"0.75"}'), "x 0.75")["false_answer"]
    assert not score_prediction(ROW_ND, parse_model_output('{"answer":"not disclosed"}'), "")["false_answer"]
    err = score_prediction(ROW_ND, parse_model_output("???"), "")
    assert not err["false_answer"] and not err["correct"] and err["pred_kind"] == "parse_error"


def test_groundedness():
    ctx = "Total energy consumed 20,55,520 GJ and Scope 1 emissions 1,045 kilo tonnes"
    assert is_grounded("2055520", ctx) is True
    assert is_grounded("20,55,520", ctx) is True
    assert is_grounded("1,045", ctx) is True
    assert is_grounded("1045000", ctx) is True               # same digits, unit-scaled
    assert is_grounded("99,999", ctx) is False
    assert is_grounded("not a number", ctx) is None


def test_grounded_when_decimals_dropped():
    from metrics import is_grounded
    ctx = "Total energy 916,465.19 GJ; other 12084.58 and 500.4"
    assert is_grounded("916,465", ctx) and is_grounded("12,085", ctx) and is_grounded("12084.6", ctx)
    assert not is_grounded("916,466", ctx) and not is_grounded("912345", ctx)
    assert not is_grounded("0.5", ctx) and not is_grounded("5", ctx)


def test_summarize_small_example():
    ctx = "value 500 here"
    rows = []
    cases = [  # (row, reply)
        ({**ROW_NUM, "gold_value": "500", "acc_main_eligible": "True", "far_eligible": "False"}, '{"answer":"500","unit":"GJ"}'),
        ({**ROW_NUM, "gold_value": "700", "acc_main_eligible": "True", "far_eligible": "False"}, '{"answer":"600","unit":"GJ"}'),
        ({**ROW_ND, "acc_main_eligible": "True", "far_eligible": "True", "label_type": "row_absent"}, '{"answer":"0.5"}'),
        ({**ROW_ND, "acc_main_eligible": "True", "far_eligible": "True", "label_type": "row_absent"}, '{"answer":"not disclosed"}'),
    ]
    for row, reply in cases:
        rows.append({**row, **score_prediction(row, parse_model_output(reply), ctx)})
    s = summarize(rows)
    assert s["accuracy"] == 0.5                      # 1 of 2 numeric correct... plus 1 of 2 ND correct = 2/4
    assert s["far"] == 0.5 and s["n_far"] == 2
    assert abs(s["uar"] - 2 / 3) < 1e-9              # answers 500 (grounded), 600 and 0.5 (not)


# ---------------------------------------------------------------- prompts
def test_prompt_and_fewshot_have_no_test_company():
    chunks = [{"page": 3, "text": "Total employees (D + E) 6,36,833"}]
    msgs = build_messages("How many employees?", chunks)
    assert msgs[0]["role"] == "system" and "[Page 3]" in msgs[1]["content"]
    folds = json.loads((ROOT / "data" / "folds.json").read_text())
    for fold in range(4):
        examples = load_fewshot(fold)
        assert len(examples) == 3
        assert all(int(folds[e["company_id"]]) != fold for e in examples)
        assert "Example 3" in build_messages("q", chunks, examples=examples)[1]["content"]
    assert gold_to_target({"status": "not_disclosed", "field_id": "F13", "gold_value": "", "gold_unit": ""}) == {"answer": "not disclosed"}


# ---------------------------------------------------------------- gold self-consistency
def test_gold_answers_score_perfectly():
    import pandas as pd
    ann = pd.read_csv(ROOT / "data" / "annotations.csv", dtype=str).fillna("")
    scored = []
    for row in ann.to_dict("records"):
        target = gold_to_target(row)
        scored.append({**row, **score_prediction(row, parse_model_output(json.dumps(target)), "")})
    s = summarize(scored)
    assert s["n_accuracy"] == 279 and s["accuracy"] == 1.0 and s["n_far"] == 77 and s["far"] == 0.0
    wrong = [{**r, **score_prediction(r, parse_model_output('{"answer":"not disclosed"}'), "")} for r in ann.to_dict("records")]
    assert summarize(wrong)["far"] == 0.0 and abs(summarize(wrong)["accuracy"] - 77 / 279) < 1e-9


if __name__ == "__main__":
    failures = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:                      # noqa: BLE001
            failures += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} tests passed")
    sys.exit(1 if failures else 0)
