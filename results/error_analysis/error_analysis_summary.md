# Stage 7: error analysis

## Automatic tag counts over ALL wrong answers (heuristic)

| Category | S1 | S2 | S3 |
|---|---|---|---|
| Retrieval miss | 3 | 3 | 3 |
| Extraction error | 123 | 88 | 29 |
| Unit/scale error | 2 | 3 | 3 |
| Invented number | 1 | 1 | 0 |
| Wrongly "not disclosed" | 2 | 5 | 11 |
| **total wrong** | **131** | **100** | **46** |

## Split of the 'Extraction error' bucket (automatic)

| Sub-type | S1 | S2 | S3 |
|---|---|---|---|
| number given although the true answer is not disclosed | 63 | 26 | 3 |
| percentage given instead of a count | 8 | 11 | 0 |
| wrong number for an answerable question | 52 | 51 | 26 |

## Sample for human tagging: 20 wrong answers per system, split across 2 people

Assignments (nobody tags a row they annotated): A: 30, B: 30

Valid values for `human_tag`: Retrieval miss | Extraction error | Unit/scale error | Invented number | Wrongly "not disclosed"

## Automatic tags within the 20-per-system sample (people will confirm or correct these)

| Category | S1 | S2 | S3 |
|---|---|---|---|
| Retrieval miss | 1 | 1 | 1 |
| Extraction error | 19 | 17 | 13 |
| Unit/scale error | 0 | 1 | 1 |
| Invented number | 0 | 0 | 0 |
| Wrongly "not disclosed" | 0 | 1 | 5 |

## Candidate illustrative examples per category (confirm by hand before using)

### Retrieval miss (9 wrong answers)
- S3 / nestle_F14: "What water intensity does Nestle India report in the optional row “Water intensity (optional) – the relevant metric may be selected by the entity”?" gold 4.88 kL/ton (page 53); system said not disclosed ; retrieved pages 72 49 48 74 48
- S1 / nestle_F14: "What water intensity does Nestle India report in the optional row “Water intensity (optional) – the relevant metric may be selected by the entity”?" gold 4.88 kL/ton (page 53); system said 4.40 kilolitres/ton; retrieved pages 72 49 48 74 48

### Extraction error (240 wrong answers)
- S1 / ultratechcement_F13: "What energy intensity does UltraTech Cement report in the optional row “Energy intensity (optional) – the relevant metric may be selected by the entity”?" gold   (page 15); system said 470.23 kJ/rupee kJ/rupee; retrieved pages 15 36 18 39 16
- S1 / hcltech_F16: "What Scope 1 and Scope 2 emission intensity does HCLTech report in the optional row “Total Scope 1 and Scope 2 emission intensity (optional) – the relevant metric may be selected by the entity”?" gold   (page 56); system said 0.076 million ₹; retrieved pages 62 56 76 73 56

### Unit/scale error (8 wrong answers)
- S1 / sunpharma_F01: "What is the total energy consumed by Sun Pharmaceutical Industries?" gold 29,07,742 GJ (page 27); system said 2,907,742 Joules; retrieved pages 27 1 29 1 31
- S3 / ntpc_F01: "What is the total energy consumed by NTPC?" gold 36,08,634 TJ (page 16); system said 360,863 TJ; retrieved pages 16 18 9 16 21

### Invented number (2 wrong answers)
- S2 / cipla_F10: "What is the total number of employees of Cipla?" gold   (page 3); system said 3,607 employees; retrieved pages 8 3 3 4 13
- S1 / drreddys_F11: "What is the number of women employees of Dr. Reddy's Laboratories?" gold 8,759 number (page 6); system said 34 %; retrieved pages 6 7 6 44 38

### Wrongly "not disclosed" (18 wrong answers)
- S1 / tatasteel_F07: "What is the total water discharged by Tata Steel?" gold 2,63,435 Million Litres (page 46); system said not disclosed ; retrieved pages 46 52 46 46 10
- S1 / tatasteel_F11: "What is the number of women employees of Tata Steel?" gold 7,626 number (page 6); system said not disclosed ; retrieved pages 6 6 7 31 41

## Where retrieval stood on the wrong answers

| System | wrong | gold page not retrieved | gold value not printed in retrieved text |
|---|---|---|---|
| S1 | 131 | 5 of 68 answerable | 8 of 68 answerable |
| S2 | 100 | 5 of 73 answerable | 9 of 73 answerable |
| S3 | 46 | 5 of 43 answerable | 10 of 43 answerable |
