# Retrieval configuration: what changed and why

Selected default (src/retrieve.py): hybrid retrieval = BGE-small dense search + BM25, merged by
reciprocal-rank fusion (rrf_k = 60, dense weight 1.0, lexical weight 1.5); the BM25 query is the
BRSR template row wording for the field (LEXICAL_TERMS, written from the public SEBI template,
not from annotations); top-5 chunks, no one-chunk-per-page restriction. Dense query vectors are
unchanged (question + FIELD_TERMS), so results/query_vectors_*.npy are still valid.

Measured on all 20 companies (gold page = annotated page; value recall = gold number printed in the
retrieved chunks; 202 answerable rows, 74 eligible not-disclosed rows with a page, 154 numeric rows).

| Configuration (top-5)                         | Page recall (answerable) | Page recall (not disclosed) | Value in context | Avg context |
|-----------------------------------------------|-------------------------:|----------------------------:|-----------------:|------------:|
| Dense, earlier baseline                        | 79.2% | 82.4% | n/a   | n/a |
| Hybrid, earlier (question+FIELD_TERMS, 1 chunk/page) | 95.0% | 93.2% | 79.9% | ~7.0k chars |
| Selected hybrid (template wording, weight 1.5) | 97.5% | 98.6% | 90.9% | ~6.9k chars |

Robustness: the selected setting was also chosen when each fold was held out in turn (grid of 2 lexical
queries x 3 lexical weights picked on the other three folds), and it beat the earlier hybrid on every fold.

Caveats to state in the paper:
- Retrieval settings were chosen using recall against annotated gold pages (all 20 companies), not model accuracy.
- About a dozen variants were compared on 202 answerable rows, so differences of 1-2 rows are noise; only the larger gains are meaningful.
- Page recall overstates what the model sees: the value-in-context measure is the stricter one.
- Remaining misses are mostly employee counts (F10/F11) printed on early Section A pages and optional-intensity rows (F13-F16).
- Rerun the official evaluation with the project venv: python src/evaluate_retrieval.py --query-vectors results/query_vectors_*.npy
  (offline numpy check already matches the earlier summaries exactly when the earlier flags are used).

Paper edits: Section IV-B -> hybrid dense+BM25 with reciprocal rank fusion, template-wording keyword query, k = 5
(no page de-duplication); Section IV-C / V -> note that retrieval was selected on gold-page recall; Section VI-C ->
keep the k sweep (1, 3, 5) for S1.
