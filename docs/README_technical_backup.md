# Predicting Antibody Developability from Sequence

A reproducible, sequence-only pipeline for predicting **HIC retention time** —
a well-defined antibody developability endpoint — from paired heavy and light
chain variable domains, with held-out external validation and interpretable
sequence determinants.

Built on the [FLAb](https://github.com/Graylab/FLAb) antibody fitness dataset.

---

## What this does

| | |
|---|---|
| **Primary endpoint** | HIC retention time (min), Jain et al. 2017 clinical-stage panel (n=137) |
| **External validation** | Jain et al. 2024 germline-Fv panel (n=43), same assay platform, zero sequence overlap |
| **Stress test** | Jetha et al. 2018 integrin mutant series (n=97), different scale, censored |
| **Secondary endpoints** | AC-SINS, Fab Tm, PSR — modelled independently, never pooled |
| **Representations** | physicochemical descriptors vs **AbLang2** (antibody PLM) vs **ESM-2 35M** (general PLM) |
| **Outputs** | trained models, metric tables, figures, a screening CLI, and a generated report |

The project deliberately models **one assay at a time**. Developability is not
a single quantity: on the very same 137 antibodies, the four endpoints
correlate only weakly with each other, so a model trained on their union would
be fitting an artefact rather than a property.

---

## Quick start

### Requirements

- Windows 10/11 (the setup script is PowerShell; the Python code is
  platform-independent)
- Python 3.11–3.13 installed per-user
- ~4 GB free disk, ~8 GB RAM
- No GPU needed — with a few hundred antibodies it buys nothing
- Internet access on the first run (clones FLAb, downloads model weights)

### 1. Build the environment

```bash
powershell -ExecutionPolicy Bypass -File .\scripts\00_setup_env.ps1
```

Creates `.venv`, installs all dependencies, writes `requirements-lock.txt`,
and runs an import smoke test.

> If `python` on your PATH resolves to an interpreter bundled with another
> application (Inkscape ships one, for instance), the script ignores it and
> finds the real per-user CPython itself.

### 2. Run the whole study

```bash
powershell -ExecutionPolicy Bypass -File .\scripts\run_all.ps1
```

**Runtime:** roughly 2–3 hours on a 12-core laptop, CPU only, plus about
10 minutes on the first run to download the AbLang2 (166 MB) and ESM-2 35M
weights.

Useful flags:

```bash
# fast smoke test: small model zoo, 2 CV repeats
powershell -ExecutionPolicy Bypass -File .\scripts\run_all.ps1 -Quick -Repeats 2

# one endpoint only
powershell -ExecutionPolicy Bypass -File .\scripts\run_all.ps1 -Endpoint hic

# reuse cached embeddings
powershell -ExecutionPolicy Bypass -File .\scripts\run_all.ps1 -SkipFeatures
```

### 3. Verify the install (optional, ~30 s)

```bash
.\.venv\Scripts\python.exe -m tests.smoke
```

Thirteen checks covering module imports, charge and hydrophobicity physics,
liability regexes, metric formulas, conformal coverage, Benjamini-Hochberg
monotonicity, table formatting, every model at both low and high
dimensionality, CDR clustering, and the guarantee that no antibody leaks
between a train and test fold. `run_all.ps1` runs these automatically before
starting any compute.

### 4. Screen new candidates

```bash
.\.venv\Scripts\python.exe -m src.tool.screen --input my_candidates.csv --endpoint hic --output screened.csv --report screened.md
```

Input CSV needs `heavy_chain` and `light_chain` columns (`antibody_id`
optional). Output gives, per antibody: predicted retention time, a conformal
prediction interval, a risk call against the published 11.7-min threshold, an
applicability-domain band, and a chemical liability report.

---

## Running steps individually

Each stage is a module and can be run on its own. They must run in this order
the first time; afterwards any stage can be re-run alone.

```bash
# set once per shell
$env:PYTHONPATH = "."; $env:PYTHONIOENCODING = "utf-8"

.\.venv\Scripts\python.exe -m src.data.acquire        # pinned FLAb clone + manifest
.\.venv\Scripts\python.exe -m src.data.curate         # clean, dedup, risk labels
.\.venv\Scripts\python.exe -m src.features.numbering  # IMGT numbering, CDRs, germlines
.\.venv\Scripts\python.exe -m src.features.descriptors
.\.venv\Scripts\python.exe -m src.data.splits         # CDR-clustered folds
.\.venv\Scripts\python.exe -m src.features.build      # AbLang2 + ESM-2 embeddings
.\.venv\Scripts\python.exe -m src.analysis.eda
.\.venv\Scripts\python.exe -m src.models.experiment
.\.venv\Scripts\python.exe -m src.analysis.motifs
.\.venv\Scripts\python.exe -m src.report.build_report
```

`PYTHONIOENCODING=utf-8` is not optional on Windows: several FLAb assay column
names contain `∆`, `λ` and `°`, which crash the default cp1252 console codec.

---

## Method summary

**Endpoint choice.** HIC retention time: a single unambiguous assay readout,
mechanistically tied to surface hydrophobicity, and the only FLAb
developability endpoint available in three independent cohorts.

**Curation.** Measurements are read from the *named raw assay column*, never
from FLAb's derived `fitness` column — `fitness` is a verbatim copy of the raw
value and does **not** encode direction of merit, so code that assumes "higher
fitness is better" silently inverts the lower-is-better endpoints.

**Leakage control.** Every antibody appearing in both a training and an
external cohort is removed before modelling. (For HIC there are none: the three
cohorts share no antibody, no heavy chain and no light chain.)

**Splitting.** Antibodies are clustered by **CDR identity** — whole-chain
identity is useless because all antibodies share 70–85% framework — using
single-linkage agglomerative clustering, then split with target-stratified
grouped K-fold, repeated 10×.

**Evaluation.** Nested CV with grouped inner grid search; out-of-fold
predictions pooled before computing metrics (averaging per-fold R² on ~27
points is biased); bootstrap CIs on every headline number; a label-permutation
test and a full-refit permutation test; paired bootstrap for model comparison;
split-conformal prediction intervals with *measured* coverage.

**Interpretation.** Univariate associations, per-CDR composition, IMGT
position-wise enrichment and model attribution — all Benjamini-Hochberg
corrected, with "survives FDR" distinguished from "nominally significant".

Full detail in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) and
[`report/FINAL_REPORT.md`](report/FINAL_REPORT.md).

---

## Repository layout

```
scripts/
  00_setup_env.ps1          environment build + dependency lock
  run_all.ps1               full pipeline (pre-flight checks -> ... -> tool demo)
  demo_screen.py            end-to-end screening demo on the held-out cohort

tests/
  smoke.py                  fast pre-flight checks: imports, metric formulas,
                            conformal coverage, fold integrity, model zoo

src/
  config.py                 endpoints, split protocol, seeds, paths
  data/
    acquire.py              pinned sparse clone of FLAb + SHA-256 manifest
    curate.py               cleaning, dedup, risk labels, leakage control
    splits.py               CDR clustering, grouped folds, applicability domain
  features/
    numbering.py            AntPack IMGT numbering, CDRs, germlines, liabilities
    descriptors.py          physicochemical, CDR-restricted and TAP-style features
    plm.py                  AbLang2 and ESM-2 embeddings (disk-cached)
    build.py                feature-set assembly
  models/
    zoo.py                  estimators + hyper-parameter grids
    train.py                nested CV, external evaluation, permutation tests
    metrics.py              metrics, bootstrap CIs, conformal intervals
    experiment.py           the sweep and all result artefacts
  analysis/
    eda.py                  provenance and distribution figures
    motifs.py               liability / composition / positional / attribution
    plotting.py             shared colourblind-safe figure style
  tool/
    screen.py               command-line screening tool
  report/
    build_report.py         generates the report from result artefacts

data/
  raw/FLAb/                 pinned upstream checkout (not committed)
  processed/                curated datasets, annotations, splits, manifest
  cache/                    PLM embeddings and identity matrices

results/
  tables/                   every metric table as CSV
  figures/                  every figure as 300-dpi PNG
  predictions/              out-of-fold and external predictions
  summary_*.json            machine-readable headline numbers

models_store/               serialised final models (joblib)
docs/                       literature review, methodology
report/FINAL_REPORT.md      the generated report
```

---

## Key dependencies

| package | purpose | note |
|---|---|---|
| `ablang2` 0.2.1 | antibody-specific PLM | weights from Zenodo on first use |
| `antpack` 0.3.8.6.2 | IMGT numbering, CDRs, germlines, liabilities | pinned: ≥0.3.9 needs an interactive licence key; 0.5 is yanked |
| `transformers` / `fair-esm` | ESM-2 35M baseline | HF route with fair-esm fallback |
| `torch` (CPU) | PLM inference | CPU build on purpose — a GPU buys nothing at this scale |
| `scikit-learn`, `lightgbm`, `xgboost` | models | |
| `biopython` | instability index, molecular weight, pairwise alignment | |
| `shap` | model attribution | falls back to permutation importance if unavailable |

**ANARCI is deliberately not used.** It is the field-standard numbering tool
but shells out to HMMER, which has no supported Windows build and cannot be
pip-installed. AntPack is a pure pip install with one dependency, and was
verified against published trastuzumab CDRs before being trusted.

Exact pinned versions: `requirements-lock.txt`.

---

## Reproducibility

- **Single seed** (`config.SEED`) drives splits, model init, bootstrap and
  permutations.
- **Data pinned** to a FLAb commit hash, with SHA-256 of every consumed file in
  `data/processed/data_manifest.json`.
- **Environment pinned** in `requirements-lock.txt`.
- **Report generated** from result artefacts — no number in it is typed by
  hand, so it cannot drift from the run that produced it.

---

## Limitations and responsible use

This is a **triage and ranking aid, not a safety assessment**.

- A favourable prediction means the antibody resembles training antibodies that
  behaved well on one assay. It is not evidence of developability.
- Antibodies flagged `extrapolation` by the applicability-domain check should be
  read as **unknown**, not safe.
- Absolute predictions in minutes are only meaningful on the same HIC platform
  (column chemistry, salt system and gradient all shift retention time). For any
  other platform, use the ranking only.
- Sequence-only features cannot distinguish buried from exposed hydrophobic
  residues, which is precisely what HIC responds to. This is the main reason
  structure-based models outperform on this endpoint.
- Training data is clinical-stage antibodies, an already heavily filtered
  population. Performance on naive repertoire antibodies is untested.

Full discussion in `report/FINAL_REPORT.md` §7–8.

---

## Data licensing and attribution

FLAb is redistributed by the [Gray Lab](https://github.com/Graylab/FLAb); the
underlying measurements belong to the cited primary publications (Jain et al.
2017 PNAS; Jain et al. 2024 mAbs; Jetha et al. 2018 mAbs). This project uses
them for non-commercial academic coursework. Check the upstream repository and
the primary papers for licence terms before any other use.

Model weights: AbLang2 (OPIG, Oxford) and ESM-2 (Meta AI) are used under their
respective licences.
