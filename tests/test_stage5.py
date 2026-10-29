"""Stage 5 plumbing test (no GPU): the energy script runs every block and writes a complete CSV."""
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_dry_run_covers_every_block_and_rotates_order():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "energy_runs_k5.csv"
        cmd = [sys.executable, str(ROOT / "src" / "run_energy.py"), "--dry-run", "--per-fold", "3",
               "--warmup", "1", "--out", str(out)]
        done = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        assert done.returncode == 0, done.stderr[-500:]
        rows = list(csv.DictReader(open(out)))
        assert len(rows) == 3 * 3 * 4                                   # systems x repeats x folds
        first_per_repeat = {}
        for r in rows:
            first_per_repeat.setdefault(int(r["repeat"]), r["system"])
        assert first_per_repeat == {1: "S1", 2: "S2", 3: "S3"}           # system order rotates
        assert all(int(r["n_questions"]) == 3 for r in rows)
        assert (Path(tmp) / "energy_summary_k5.csv").exists()
        again = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)   # resume adds nothing
        assert again.returncode == 0 and len(list(csv.DictReader(open(out)))) == 36


if __name__ == "__main__":
    test_dry_run_covers_every_block_and_rotates_order()
    print("PASS  test_dry_run_covers_every_block_and_rotates_order\n\n1/1 tests passed")
