# Retrieval Evaluation

Run on 2026-07-03T08:24:45+00:00  
Embedding model: `all-MiniLM-L6-v2`  
Queries scored (excludes should_refuse): 12

## Overall metrics

| Metric | k=5 | k=10 |
|---|---|---|
| Precision | 0.250 | 0.142 |
| Recall    | 0.806    | 0.917    |
| MRR       | 0.609 | — |

## Per-category breakdown

| Category | n | Precision@5 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|---|
| checkbox_table | 3 | 0.133 | 0.667 | 1.000 | 0.704 |
| domain_terminology | 3 | 0.400 | 0.889 | 1.000 | 0.750 |
| parameter_table | 3 | 0.333 | 1.000 | 1.000 | 0.733 |
| prose | 3 | 0.133 | 0.667 | 0.667 | 0.250 |

## Should-refuse queries (guardrail check)

These have no labeled relevant chunks. Retrieval will still return top-K chunks; 
the actual refusal behavior is tested by the generation layer, not this harness.

- **q13**: What is the average cost of a condominium in Singapore?
- **q14**: Who is the current CEO of URA?
- **q15**: What is the plot ratio for my plot of land at 123 Sengkang Drive?

## Per-query detail

### q01 (parameter_table)
*What is the road buffer for an expressway?*
- First relevant chunk rank: `1`
- Precision@5: 0.600, Recall@5: 1.000
- Relevant IDs: `['Summary-Commercial_p0_t0_r3', 'Summary-B1_p0_t0_r3', 'Summary-B2_p0_t0_r3']`

### q02 (parameter_table)
*What is the minimum unit size for B1 industrial developments?*
- First relevant chunk rank: `1`
- Precision@5: 0.200, Recall@5: 1.000
- Relevant IDs: `['Summary-B1_p1_t0_r6']`

### q03 (parameter_table)
*What is the minimum floor-to-floor height in B1 industrial buildings?*
- First relevant chunk rank: `5`
- Precision@5: 0.200, Recall@5: 1.000
- Relevant IDs: `['Summary-B1_p0_t0_r10']`

### q04 (checkbox_table)
*Is a void deck included as GFA?*
- First relevant chunk rank: `1`
- Precision@5: 0.200, Recall@5: 1.000
- Relevant IDs: `['Summary_GFA_p2_t0_r18']`

### q05 (checkbox_table)
*Are bay windows counted toward GFA?*
- First relevant chunk rank: `9`
- Precision@5: 0.000, Recall@5: 0.000
- Relevant IDs: `['Summary_GFA_p0_t3_r4']`

### q06 (checkbox_table)
*Is a roof cover excluded from GFA calculations?*
- First relevant chunk rank: `1`
- Precision@5: 0.200, Recall@5: 1.000
- Relevant IDs: `['Summary_GFA_p2_t0_r1']`

### q07 (domain_terminology)
*What is the maximum plot ratio for HDB residential estates?*
- First relevant chunk rank: `1`
- Precision@5: 0.600, Recall@5: 1.000
- Relevant IDs: `['Summary-PW_p0_t0_r4', 'Summary-CCI_p0_t0_r4', 'Summary-EI_p0_t0_r4']`

### q08 (domain_terminology)
*What is the GPR for landed housing fringe areas?*
- First relevant chunk rank: `4`
- Precision@5: 0.400, Recall@5: 0.667
- Relevant IDs: `['Summary-CCI_p0_t0_r3', 'Summary-EI_p0_t0_r3', 'Summary-PW_p0_t0_r3']`

### q09 (domain_terminology)
*What zoning covers religious buildings like temples and mosques?*
- First relevant chunk rank: `1`
- Precision@5: 0.200, Recall@5: 1.000
- Relevant IDs: `['MP25WrittenStatement_p16_t0_r1']`

### q10 (prose)
*What is the definition of plot ratio in the Master Plan?*
- First relevant chunk rank: `not in top-10`
- Precision@5: 0.000, Recall@5: 0.000
- Relevant IDs: `['MP25WrittenStatement_p5_para0']`

### q11 (prose)
*Which act defines what counts as a national park?*
- First relevant chunk rank: `4`
- Precision@5: 0.200, Recall@5: 1.000
- Relevant IDs: `['MP25WrittenStatement_p5_para0']`

### q12 (prose)
*When was the first Master Plan for Singapore approved?*
- First relevant chunk rank: `2`
- Precision@5: 0.200, Recall@5: 1.000
- Relevant IDs: `['MP25WrittenStatement_p2_para0']`
