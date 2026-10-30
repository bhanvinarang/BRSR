# BRSR-QA

BRSR-QA is a research pipeline for answering questions about disclosures in Indian
companies' Business Responsibility and Sustainability Reports (BRSRs). It combines
PDF text extraction, company-specific retrieval, prompted language-model inference,
and optional QLoRA fine-tuning. The repository contains the dataset annotations,
experiment code, notebooks, and analysis outputs.

The evaluation set covers 20 reports and 320 questions (16 fields per company).
The reports span FY 2024-25 and FY 2025-26. Of the 320 annotations, 191 are
answered, 11 are reported zeros, and 118 are not disclosed. The main accuracy
set excludes the 41 `PARTS_ONLY` rows (279 rows); the false-answer-rate
denominator is 77 not-disclosed rows after excluding those same rows.

## Systems

The inference scripts compare three systems using the same frozen retrieved
evidence:

- **S1 — zero-shot:** the base instruction-tuned model with no examples.
- **S2 — few-shot:** the same base model with three fold-safe solved examples.
- **S3 — fine-tuned:** the S1 prompt with a fold-specific QLoRA adapter.

All systems use greedy decoding. The configured base model is
`Qwen/Qwen2.5-1.5B-Instruct`. S3 trains and evaluates by company-level folds:
each test fold contains five companies, and its adapter is checked to ensure it
was not trained on those companies.

Retrieval uses `BAAI/bge-small-en-v1.5` and FAISS dense search combined with BM25
using reciprocal-rank fusion. The selected configuration retrieves the top five
chunks using BRSR template wording for the lexical query; there is no one-chunk-
per-page restriction. Retrieval was selected using annotated gold-page recall,
not model accuracy. See [retrieval selection notes](docs/retrieval_selection_notes.md)
for configuration comparisons and caveats.

## Repository layout

```text
data/        Annotations, company/report provenance, folds, and review files
docs/        Dataset corrections and retrieval-selection notes
notebooks/   Google Colab workflows for inference/training and energy evaluation
results/     Prediction, retrieval, energy, statistics, and error-analysis outputs
src/         Dataset, extraction, retrieval, inference, training, and analysis scripts
tests/       CPU-capable plumbing and regression checks
```

Generated PDF-derived text, indexes, fine-tuning data, model adapters, and raw
PDFs are intentionally excluded from version control; see `.gitignore`.

## Requirements and setup

Python 3.10 or later is recommended. A GPU is not needed for data preparation,
PDF extraction, indexing, retrieval evaluation, and dry-run checks.

```bash
git clone <repository-url>
cd BRSR

python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The retrieval dependencies include PyMuPDF, pandas, NumPy, Sentence
Transformers, FAISS CPU, scikit-learn, and statsmodels. The separate
`requirements_gpu.txt` is for the GPU experiment environment and includes
Transformers, PEFT, bitsandbytes, Accelerate, and CodeCarbon.

### Source reports

The report PDFs are **not included**. If you have lawful access to the reports,
place the 20 files in `raw_pdfs/` using names that end in `_FY2425.pdf` or
`_FY2526.pdf` (for example, `company_FY2425.pdf`). `src/extract.py` expects
exactly 20 PDFs. Check the source URL and SHA-256 provenance in
[`data/companies.csv`](data/companies.csv); do not redistribute report PDFs
unless their terms explicitly permit it.

## Rebuild the retrieval inputs

Run commands from the repository root with the project environment activated.
The clean annotations, company folds, few-shot examples, frozen retrieval
contexts, and report-derived corpus/index are inputs to later stages.

```bash
# Optional: validate the raw annotation export and regenerate the clean master CSV
python src/prepare_dataset.py data/annotations_raw.csv

# Extract report pages, then build one dense index per company
python src/extract.py
python src/chunk_index.py

# Evaluate the selected hybrid retrieval settings and save the top-five contexts
python src/evaluate_retrieval.py \
  --k 5 \
  --output results/retrieval_recall_final_rows.csv \
  --summary results/retrieval_recall_final_summary.json
python src/freeze_retrieval.py

# Build fold-safe few-shot examples and QLoRA training examples
python src/build_fewshot.py
python src/build_ft_data.py
```

By default, extraction writes `data/corpus/pages.jsonl` and indexing writes to
`data/index/`. The retrieval-evaluation command above writes the file expected
by `src/freeze_retrieval.py`, which creates `data/retrieval_top5.jsonl`.
`src/build_fewshot.py` writes `data/fewshot_examples.json`; `src/build_ft_data.py`
writes training data under `data/ft/`.

## Run and evaluate the systems

### CPU plumbing checks

The inference dry run uses a stub generator, not the language model. It checks
that prompts, retrieval contexts, output parsing, and resumable output work
without downloading model weights or requiring a GPU.

```bash
python src/run_inference.py --system S1 --dry-run --limit 8
python tests/test_stage3.py
python tests/test_stage4.py
python tests/test_stage5.py
```

Stage 4 and 5 tests use the frozen retrieval and fold-specific training inputs
described above. They exercise dry-run paths and do not measure real model
quality, latency, or energy consumption.

### GPU inference and fine-tuning

For real inference and QLoRA training, use a CUDA GPU (the notebooks target a
Google Colab T4). Ensure the model weights are accessible in your Hugging Face
environment. The GPU environment can be installed with:

```bash
python -m pip install -r requirements_gpu.txt
```

Run inference on all four folds:

```bash
python src/run_inference.py --system S1
python src/run_inference.py --system S2
```

For S3, train one adapter per fold and then run inference:

```bash
python src/train_qlora.py --fold 0
python src/train_qlora.py --fold 1
python src/train_qlora.py --fold 2
python src/train_qlora.py --fold 3
python src/run_inference.py --system S3
```

Inference outputs are written to `results/predictions_<SYSTEM>_k<K>.jsonl`
unless `--out` is specified. The scripts resume by skipping questions already
present in the output file. For a smaller smoke run, add `--limit 8`; for S3,
adapters for each requested fold must already exist under `adapters/fold<N>/`.

To generate evaluation summaries and statistics after producing prediction
files for S1/S2/S3 at k=5 **and** running the energy/latency experiment:

```bash
python src/stats.py
```

### Colab bundle and energy/latency

The GPU notebooks are
[`notebooks/stage4_colab.ipynb`](notebooks/stage4_colab.ipynb) and
[`notebooks/stage5_colab.ipynb`](notebooks/stage5_colab.ipynb). The Stage 4
notebook runs S1/S2/S3 and trains the fold adapters; Stage 5 measures energy
and latency. Both notebooks document their GPU setup and resumable workflow.

After building `data/ft/` and `data/retrieval_top5.jsonl`, create the upload
bundle:

```bash
python src/make_bundle.py
```

Upload `stage4_bundle.zip` to the root of Google Drive and follow the notebook
instructions. The bundle contains the GPU-stage code and required data, but
not the source PDFs, page corpus, FAISS indexes, or Python environment.

To run energy/latency measurements directly from a configured GPU environment:

```bash
python src/run_energy.py
python src/run_energy.py --summarize-only
```

The default experiment measures all three systems over three repeats and all
320 questions. Measurements are hardware-dependent; use the same GPU, session,
and precision when comparing systems. `--dry-run` checks only the measurement
plumbing and does not provide meaningful performance measurements.

## Dataset, review, and paper notes

- The annotations contain 320 questions across 20 companies and 16 fields.
  The label definitions and paper counts are documented in
  [dataset corrections](docs/paper_dataset_corrections.md).
- Company/report provenance, source URLs, and SHA-256 hashes are recorded in
  `data/companies.csv`; fixed company-level folds are in `data/folds.json`.
- A deterministic blinded review sample is prepared with
  `python src/make_review_sample.py`. Do not claim inter-annotator agreement
  until an independent reviewer completes `data/review_responses.csv` and all
  disagreements are adjudicated.
- The repository includes review and analysis files. Check their contents and
  the applicable data permissions before sharing them outside the project.
- No license file is currently provided. Add or consult the project's license
  and data-use terms before redistributing code or data.

## Citation

If you use this project in research, cite the associated BRSR-QA paper when its
final bibliographic details are available. For reproducibility, report the
dataset version, fold, prompt variant, retrieval configuration, model revision,
and hardware used.
