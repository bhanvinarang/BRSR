# Dataset corrections for the BRSR-QA paper

These replacements are generated from `data/annotations.csv` after the UltraTech
filename correction and raw-CSV cleanup. Apply them to the editable paper source;
the supplied PDF cannot be safely edited as if it were the source manuscript.

## Abstract and Section III-A

Keep the total statement: 320 questions, with 191 answered, 11 reported zeros,
and 118 not disclosed (36.9%).

Replace the financial-year statement with:

> Ten reports cover FY 2024-25 and ten cover FY 2025-26.

## Table II

| Label | Count | Share |
| --- | ---: | ---: |
| Answered | 191 | 59.7% |
| Zero (reported 0 or “Nil”) | 11 | 3.4% |
| Not disclosed: value printed as blank, “NA” or dash | 33 | 10.3% |
| Not disclosed: optional row absent from the report | 39 | 12.2% |
| Not disclosed: other | 5 | 1.6% |
| Not disclosed: components only (`PARTS_ONLY`) | 41 | 12.8% |

State that the false-answer-rate denominator is 77: the 118 not-disclosed rows
minus the 41 `PARTS_ONLY` rows. State that the main accuracy set contains 279
rows, excluding `PARTS_ONLY` rows.

## Section III-B

Do not claim inter-annotator agreement until a genuinely independent reviewer has
completed `data/review_responses.csv` and every disagreement has been adjudicated.
The deterministic 64-row (20%) blinded review sample is produced by
`src/make_review_sample.py` using seed `20261026`.

## Data and code availability

Describe the released data as annotations, company identifiers, folds, report
provenance, source URLs, and SHA-256 hashes. Do not redistribute the source PDFs
unless their terms explicitly allow redistribution.
