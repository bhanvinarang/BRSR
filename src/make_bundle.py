"""Zip only what the GPU machine needs (no PDFs, venv, indexes or page corpus).

    python src/make_bundle.py     ->  stage4_bundle.zip   (upload it to Google Drive)
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ["prompts.py", "parse_output.py", "metrics.py", "build_ft_data.py", "train_qlora.py",
       "run_inference.py", "run_energy.py", "check_stage3.py"]
DATA = ["annotations.csv", "folds.json", "fewshot_examples.json", "retrieval_top5.jsonl"]


def main():
    out = ROOT / "stage4_bundle.zip"
    files = [ROOT / "src" / n for n in SRC] + [ROOT / "data" / n for n in DATA]
    files += sorted((ROOT / "data" / "ft").glob("*")) + sorted((ROOT / "tests").glob("test_*.py"))
    files += [ROOT / "requirements_gpu.txt", ROOT / "notebooks" / "stage4_colab.ipynb", ROOT / "notebooks" / "stage5_colab.ipynb"]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            bundle.write(path, path.relative_to(ROOT).as_posix())
        for name in ("results", "adapters"):
            bundle.writestr(f"{name}/.keep", "")
    print(f"{out} ({out.stat().st_size / 1e6:.1f} MB, {len(files)} files)")


if __name__ == "__main__":
    main()
