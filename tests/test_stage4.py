"""Checks for the Stage 4 data and plumbing that need no GPU.   python tests/test_stage4.py"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from run_inference import check_adapter_meta  # noqa: E402
from train_qlora import read_examples  # noqa: E402

FOLDS = json.loads((ROOT / "data" / "folds.json").read_text())
ANN = pd.read_csv(ROOT / "data" / "annotations.csv", dtype=str).fillna("")


def test_frozen_retrieval_complete():
    recs = [json.loads(l) for l in open(ROOT / "data" / "retrieval_top5.jsonl", encoding="utf-8")]
    assert [r["row_id"] for r in recs] == list(ANN.row_id)
    assert all(len(r["retrieved_chunks"]) == 5 for r in recs)
    assert all(c["chunk_id"].startswith(r["company_id"]) for r in recs for c in r["retrieved_chunks"])


def test_ft_data_never_contains_test_companies():
    for fold in range(4):
        examples, train, test = read_examples(ROOT / "data" / "ft" / f"fold{fold}_train.jsonl", fold, FOLDS)
        assert len(examples) == 240 and len(train) == 15 and len(test) == 5
        assert set(train).isdisjoint(test)
        rows = {e["row_id"] for e in examples}
        assert rows.isdisjoint(set(ANN[ANN.company_id.isin(test)].row_id))


def test_ft_targets_follow_the_rule():
    examples = [json.loads(l) for l in open(ROOT / "data" / "ft" / "fold0_train.jsonl", encoding="utf-8")]
    gold = ANN.set_index("row_id")
    for e in examples:
        row = gold.loc[e["row_id"]]
        target = json.loads(e["target"])
        if row.status == "not_disclosed":
            assert target == {"answer": "not disclosed"}
        elif not e["relabelled"]:
            assert target["answer"] == (row.gold_value if row.field_id != "F12" else "Yes")
        else:
            assert target == {"answer": "not disclosed"}          # retrieval missed the answer
    assert 5 <= sum(e["relabelled"] for e in examples) <= 40
    assert all(e["messages"][0]["role"] == "system" and "Question:" in e["messages"][1]["content"] for e in examples)


def test_adapter_leak_check_fires():
    good = {"fold": 0, "train_companies": [c for c, f in FOLDS.items() if int(f) != 0]}
    check_adapter_meta(good, 0, FOLDS)
    bad = {"fold": 0, "train_companies": [c for c, f in FOLDS.items()]}
    for meta, fold in ((bad, 0), (good, 1)):
        try:
            check_adapter_meta(meta, fold, FOLDS)
        except AssertionError:
            continue
        raise AssertionError("leak was not caught")


def test_dry_run_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        for system in ("S1", "S2", "S3"):
            out = Path(tmp) / f"{system}.jsonl"
            run = subprocess.run([sys.executable, str(ROOT / "src" / "run_inference.py"), "--system", system,
                                  "--dry-run", "--limit", "16", "--out", str(out)], capture_output=True, text=True, cwd=ROOT)
            assert run.returncode == 0, run.stderr[-800:]
            recs = [json.loads(l) for l in open(out, encoding="utf-8")]
            assert len(recs) == 64 and {r["fold"] for r in recs} == {0, 1, 2, 3}
            assert all(k in recs[0] for k in ("system", "fold", "question_id", "retrieved_chunks", "raw_output", "parsed_answer"))
            assert all(len(r["retrieved_chunks"]) == 5 for r in recs)
            # resume: a second run adds nothing
            subprocess.run([sys.executable, str(ROOT / "src" / "run_inference.py"), "--system", system, "--dry-run",
                            "--limit", "16", "--out", str(out)], capture_output=True, text=True, cwd=ROOT, check=True)
            assert len(open(out, encoding="utf-8").readlines()) == 64


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
