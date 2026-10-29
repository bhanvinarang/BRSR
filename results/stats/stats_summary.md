# Stage 6 results

## Table II: accuracy, FAR, UAR (k=5, 95% bootstrap CI over questions)

| System | Accuracy % | FAR % | UAR % | Acc. answerable % | Acc. not-disclosed % |
|---|---|---|---|---|---|
| S1 RAG zero-shot | 53.0 [47.0, 58.8] | 81.8 [72.7, 90.9] | 0.4 [0.0, 1.1] | 66.3 | 18.2 |
| S2 RAG few-shot | 64.2 [58.1, 69.9] | 35.1 [24.7, 45.5] | 0.8 [0.0, 2.1] | 63.9 | 64.9 |
| S3 RAG + QLoRA | 83.5 [78.9, 87.5] | 3.9 [0.0, 9.1] | 1.1 [0.0, 2.8] | 78.7 | 96.1 |

Accuracy over n=279 eligible questions, FAR over n=77, UAR over each system's numeric answers (S1 275, S2 239, S3 180).

Company-level (cluster) bootstrap intervals, accuracy and FAR:

- S1: accuracy 53.0 [46.4, 59.7], FAR 81.8 [71.2, 91.0]
- S2: accuracy 64.2 [58.8, 69.9], FAR 35.1 [22.7, 47.6]
- S3: accuracy 83.5 [77.7, 88.6], FAR 3.9 [0.0, 7.9]

## Paired comparisons (exact McNemar; Holm correction over the three pairs per metric)

| Metric | A vs B | n | only A | only B | B - A (pp) [95% CI] | p | p (Holm) |
|---|---|---|---|---|---|---|---|
| accuracy (correct) | S1 vs S2 | 279 | 23 | 54 | +11.1 [+5.0, +17.6] | 0.0005 | 0.0005 |
| accuracy (correct) | S1 vs S3 | 279 | 10 | 95 | +30.5 [+24.0, +36.6] | 0.0000 | 0.0000 |
| accuracy (correct) | S2 vs S3 | 279 | 7 | 61 | +19.4 [+14.0, +24.7] | 0.0000 | 0.0000 |
| FAR (false answer) | S1 vs S2 | 77 | 36 | 0 | -46.8 [-57.2, -36.4] | 0.0000 | 0.0000 |
| FAR (false answer) | S1 vs S3 | 77 | 60 | 0 | -77.9 [-87.0, -67.5] | 0.0000 | 0.0000 |
| FAR (false answer) | S2 vs S3 | 77 | 24 | 0 | -31.2 [-41.6, -20.8] | 0.0000 | 0.0000 |

'only A' counts questions where A had the event (correct / false answer) and B did not.

## Per-field accuracy (%)

| Field | Name | n | S1 | S2 | S3 |
|---|---|---|---|---|---|
| F01 | Total energy consumed | 20 | 65 | 80 | 90 |
| F03 | Scope 1 emissions | 19 | 89 | 89 | 89 |
| F04 | Scope 2 emissions | 20 | 70 | 50 | 90 |
| F05 | Scope 3 emissions | 19 | 84 | 84 | 74 |
| F06 | Total water withdrawn | 20 | 65 | 70 | 85 |
| F07 | Total water discharged | 20 | 45 | 50 | 50 |
| F08 | Total waste generated | 20 | 75 | 80 | 85 |
| F09 | Hazardous waste generated | 1 | 0 | 0 | 0 |
| F10 | Total employees | 20 | 75 | 65 | 80 |
| F11 | Women employees | 20 | 5 | 5 | 75 |
| F12 | Employee grievance mechanism | 20 | 100 | 100 | 100 |
| F13 | Energy intensity (optional) | 20 | 15 | 40 | 90 |
| F14 | Water intensity (optional) | 20 | 25 | 55 | 75 |
| F15 | Waste intensity (optional) | 20 | 30 | 65 | 95 |
| F16 | Scope 1 and Scope 2 emission intensity (optional) | 20 | 5 | 70 | 95 |

## k sweep, S1 only

| k | Accuracy % [CI] | FAR % | UAR % | Wh/query | Latency ms |
|---|---|---|---|---|---|
| 1 | 47.7 [41.9, 53.8] | 62.3 [50.6, 72.7] | 1.8 | 0.0194 | 885 |
| 3 | 52.0 [45.9, 57.4] | 81.8 [72.7, 89.6] | 0.4 | 0.0277 | 1266 |
| 5 | 53.0 [47.0, 58.8] | 81.8 [72.7, 90.9] | 0.4 | 0.0385 | 1767 |

## Table III: inference energy and latency (k=5, Tesla T4, fp16, greedy, 3 repeats)

| System | Wh/query (CodeCarbon) | Wh/query (GPU only, NVML) | Latency ms/query | Prompt tokens | Answer tokens |
|---|---|---|---|---|---|
| S1 RAG zero-shot | 0.0385 ± 0.0000 | 0.0326 ± 0.0000 | 1767 ± 0.7 | 2280 | 20.2 |
| S2 RAG few-shot | 0.0465 ± 0.0001 | 0.0394 ± 0.0001 | 2144 ± 3.3 | 2883 | 18.1 |
| S3 RAG + QLoRA | 0.0338 ± 0.0002 | 0.0286 ± 0.0001 | 1555 ± 6.6 | 2280 | 14.6 |

Training energy (QLoRA, one adapter per fold, CodeCarbon):

| Fold | Energy Wh | Minutes |
|---|---|---|
| 0 | 70.9 | 54.9 |
| 1 | 71.6 | 55.4 |
| 2 | 69.7 | 53.9 |
| 3 | 63.7 | 48.7 |
| total | 275.9 | 213.0 |

Break-even (queries after which S3's lower inference energy pays back all four training runs):
- vs S1: saving 0.0047 Wh/query -> about 58,920 queries
- vs S2: saving 0.0127 Wh/query -> about 21,692 queries
