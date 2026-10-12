# Developability screening report: HIC retention time

- Antibodies screened: **43**
- Model: `desc_all` + `svr_rbf`
- Endpoint: HIC retention time (min); lower is better
- Risk threshold: 11.7 min (Jain et al. 2017 warning flag)
- Prediction intervals: +/-1.23 min at 80% nominal coverage

## Risk calls

| call | n | meaning |
|---|---|---|
| high_risk | 1 | whole interval on the unfavourable side |
| uncertain | 23 | interval straddles the threshold |
| low_risk | 19 | whole interval on the favourable side |

## Applicability domain

| band | n |
|---|---|
| in_domain | 31 |
| borderline | 12 |
| extrapolation | 0 |

## Ranked candidates

| rank | antibody | predicted min | interval | risk | applicability | CDR liabilities |
|---|---|---|---|---|---|---|
| 1 | jain2024_hic_0009 | 9.01 | [7.78, 10.23] | low_risk | in_domain | 0 |
| 2 | jain2024_hic_0007 | 9.40 | [8.17, 10.63] | low_risk | in_domain | 2 |
| 3 | jain2024_hic_0025 | 9.51 | [8.28, 10.73] | low_risk | borderline | 5 |
| 4 | jain2024_hic_0034 | 9.75 | [8.52, 10.97] | low_risk | borderline | 2 |
| 5 | jain2024_hic_0006 | 9.79 | [8.56, 11.01] | low_risk | in_domain | 2 |
| 6 | jain2024_hic_0042 | 9.82 | [8.59, 11.04] | low_risk | borderline | 2 |
| 7 | jain2024_hic_0033 | 9.98 | [8.75, 11.20] | low_risk | in_domain | 6 |
| 8 | jain2024_hic_0010 | 9.98 | [8.76, 11.21] | low_risk | in_domain | 4 |
| 9 | jain2024_hic_0027 | 10.02 | [8.80, 11.25] | low_risk | in_domain | 7 |
| 10 | jain2024_hic_0008 | 10.09 | [8.87, 11.32] | low_risk | in_domain | 3 |
| 11 | jain2024_hic_0032 | 10.10 | [8.87, 11.32] | low_risk | in_domain | 4 |
| 12 | jain2024_hic_0022 | 10.11 | [8.88, 11.33] | low_risk | in_domain | 3 |
| 13 | jain2024_hic_0035 | 10.13 | [8.90, 11.35] | low_risk | borderline | 9 |
| 14 | jain2024_hic_0001 | 10.14 | [8.91, 11.36] | low_risk | in_domain | 3 |
| 15 | jain2024_hic_0028 | 10.20 | [8.97, 11.42] | low_risk | in_domain | 3 |
| 16 | jain2024_hic_0036 | 10.24 | [9.02, 11.47] | low_risk | borderline | 4 |
| 17 | jain2024_hic_0030 | 10.28 | [9.05, 11.50] | low_risk | in_domain | 3 |
| 18 | jain2024_hic_0029 | 10.29 | [9.06, 11.51] | low_risk | in_domain | 9 |
| 19 | jain2024_hic_0031 | 10.32 | [9.09, 11.54] | low_risk | in_domain | 9 |
| 20 | jain2024_hic_0000 | 10.51 | [9.28, 11.73] | uncertain | in_domain | 4 |
| 21 | jain2024_hic_0005 | 10.55 | [9.33, 11.78] | uncertain | in_domain | 6 |
| 22 | jain2024_hic_0016 | 10.61 | [9.38, 11.83] | uncertain | in_domain | 4 |
| 23 | jain2024_hic_0003 | 10.65 | [9.43, 11.88] | uncertain | borderline | 1 |
| 24 | jain2024_hic_0023 | 10.68 | [9.45, 11.90] | uncertain | borderline | 6 |
| 25 | jain2024_hic_0011 | 10.83 | [9.61, 12.06] | uncertain | in_domain | 6 |
| 26 | jain2024_hic_0020 | 10.87 | [9.65, 12.10] | uncertain | in_domain | 2 |
| 27 | jain2024_hic_0004 | 10.93 | [9.71, 12.16] | uncertain | in_domain | 2 |
| 28 | jain2024_hic_0026 | 10.93 | [9.71, 12.16] | uncertain | in_domain | 5 |
| 29 | jain2024_hic_0024 | 10.96 | [9.74, 12.19] | uncertain | borderline | 6 |
| 30 | jain2024_hic_0002 | 11.00 | [9.77, 12.22] | uncertain | in_domain | 2 |
| 31 | jain2024_hic_0037 | 11.03 | [9.80, 12.25] | uncertain | in_domain | 5 |
| 32 | jain2024_hic_0017 | 11.10 | [9.87, 12.32] | uncertain | in_domain | 5 |
| 33 | jain2024_hic_0019 | 11.29 | [10.07, 12.52] | uncertain | in_domain | 2 |
| 34 | jain2024_hic_0013 | 11.40 | [10.17, 12.63] | uncertain | in_domain | 4 |
| 35 | jain2024_hic_0012 | 11.45 | [10.23, 12.68] | uncertain | borderline | 5 |
| 36 | jain2024_hic_0018 | 11.50 | [10.28, 12.73] | uncertain | in_domain | 4 |
| 37 | jain2024_hic_0021 | 11.50 | [10.28, 12.73] | uncertain | in_domain | 2 |
| 38 | jain2024_hic_0014 | 11.74 | [10.51, 12.96] | uncertain | in_domain | 4 |
| 39 | jain2024_hic_0015 | 11.88 | [10.66, 13.11] | uncertain | borderline | 6 |
| 40 | jain2024_hic_0040 | 11.93 | [10.71, 13.16] | uncertain | borderline | 2 |
| 41 | jain2024_hic_0039 | 12.45 | [11.22, 13.67] | uncertain | borderline | 7 |
| 42 | jain2024_hic_0038 | 12.70 | [11.48, 13.93] | uncertain | in_domain | 5 |
| 43 | jain2024_hic_0041 | 13.41 | [12.18, 14.63] | high_risk | in_domain | 3 |

## How to read this

The model ranks candidates by predicted assay value. It does not replace the assay. An `extrapolation` applicability band means the antibody is less similar to the training set than anything the model was validated on, so its prediction should be treated as unknown rather than as evidence of safety. `uncertain` risk calls are exactly that: the prediction interval covers the threshold, so the model cannot separate the two outcomes for that antibody.

Liability motif counts are independent of the model. They are rule-based sequence annotations (glycosylation sequons, deamidation and isomerisation dipeptides, oxidation-prone residues) and are reported so that a candidate with a clean predicted value but an exposed CDR liability is not overlooked.