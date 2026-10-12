# Deliverables map

Where each item requested in the project brief lives in this repository.
Paths are relative to the project root. Items marked *(generated)* are
produced by `scripts/run_all.ps1` and are not meaningful until it has run.

---

## Technical

| Requested | Delivered as |
|---|---|
| Curated / preprocessed dataset | `data/processed/dataset_{hic,acsins,tm,psr}.csv` — one row per antibody per cohort, with sequences, measurement, units, cohort role, censoring flag and risk labels |
| Documented split-generation procedure | `src/data/splits.py` (code) + `docs/METHODOLOGY.md` §4 (rationale) + `data/processed/splits_*.json` (the exact fold indices actually used) |
| Data-processing pipeline | `src/data/acquire.py` → `curate.py` → `splits.py` |
| Feature-engineering pipeline | `src/features/numbering.py`, `descriptors.py`, `plm.py`, `build.py` |
| Trained baseline model | `baseline_trivial` feature set (chain lengths + mean hydropathy) — reported in every comparison table as the floor any real model must clear |
| Trained final models | `models_store/final_{endpoint}.joblib` *(generated)* — includes the fitted pipeline, the feature-set name and column order, the conformal half-width, and the risk threshold |
| Evaluation results | `results/tables/cv_regression_*.csv`, `cv_classification_*.csv`, `external_*.csv`, `model_comparison_*.csv` *(generated)* |
| Validation | `results/tables/external_*.csv` — held-out Jain 2024 cohort plus the out-of-distribution Jetha stress test, with the evaluation role of each stated explicitly |
| Error analysis | `src/analysis/error_analysis.py`; `results/tables/error_*_worst_predictions.csv`, `results/figures/error_*_residuals.png` *(generated)* |
| Uncertainty quantification | Split-conformal prediction intervals (`src/models/metrics.py`), with **measured** coverage per cohort and per applicability band rather than an assumed guarantee |
| Applicability-domain analysis | `data/processed/applicability_*.json`, `results/tables/error_*_ad_bands.csv`, `results/figures/error_*_applicability.png` *(generated)* |
| Ablation | `results/tables/ablation_*.csv` — leave-one-family-out and each-family-alone over six descriptor families, plus the whole feature-set sweep |

## Research

| Requested | Delivered as |
|---|---|
| Focused literature review | `docs/LITERATURE_REVIEW.md` — 9 sections, ~30 references with DOIs, separating published values from inferred ones |
| Problem-specific assumptions | `docs/METHODOLOGY.md` — every design decision with its justification, including the three that were **changed after looking at the data** |
| Research findings, figures, tables | `results/figures/` (300-dpi PNG), `results/tables/` (CSV) *(generated)* |
| Reproducible interpretation | `src/analysis/motifs.py` — liability motifs, per-CDR composition, IMGT position-wise enrichment and model attribution, all Benjamini-Hochberg corrected |
| Research report | `report/FINAL_REPORT.md` *(generated)* — 11 sections, every number read from the result artefacts |
| Publication-ready material | Report §7 positions the result against published comparators with their protocols; §8 states the limitations that would have to be addressed first |

## Software / tool

| Requested | Delivered as |
|---|---|
| Sequence-based screening tool | `src/tool/screen.py` — CLI taking a CSV of heavy/light chains, returning predictions, conformal intervals, a risk call, an applicability-domain band and a liability report |
| Worked demonstration | `scripts/demo_screen.py` — runs the tool on the held-out cohort, where ground truth exists, so the demo can be checked rather than merely displayed |
| Source code | `src/` — 17 modules |
| Pinned dependencies | `requirements-lock.txt` (exact versions) + `scripts/00_setup_env.ps1` (the install that produced it) |
| Model / checkpoint files | `models_store/final_*.joblib` *(generated)* |
| Reproducibility instructions | `README.md` §Quick start; `report/FINAL_REPORT.md` §10 |

## Documentation

| Requested | Delivered as |
|---|---|
| Technical documentation | `README.md`, module docstrings throughout `src/` |
| Methodology documentation | `docs/METHODOLOGY.md` |
| Data provenance | `data/processed/data_manifest.json` — SHA-256 of every consumed file, the pinned FLAb commit, source publication and assay column for each cohort; `data/processed/curation_report.json` — what was dropped and why |
| Limitations | `report/FINAL_REPORT.md` §8; `README.md` §Limitations |
| Uncertainty notes | `report/FINAL_REPORT.md` §3.3 (intervals and their two caveats), §3.5 (measured conformal coverage), §5.2 (coverage by applicability band) |
| Responsible-use notes | `report/FINAL_REPORT.md` §9; `README.md` §Limitations and responsible use; the screening tool prints the same caveats in its own report |
| README with exact execution instructions | `README.md` |

## Quality controls not explicitly requested

| Control | Where |
|---|---|
| Pre-flight test suite | `tests/smoke.py` — 13 checks (imports, charge physics, hydrophobicity scales, liability regexes, metric formulas, conformal coverage, BH monotonicity, table formatting, every model at low and high dimensionality, CDR clustering, fold integrity). Run automatically by `run_all.ps1` before any compute |
| Leakage control | Cross-cohort overlap removal in `curate.py`, verified at three levels (exact pair, heavy-only, light-only) and reported in `curation_report.json` |
| Independent liability implementation | `src/features/numbering.py` carries our own regex set alongside AntPack's curated scanner, so the two can be cross-checked |
| Adversarial code audit | Five independent reviewers (leakage, metrics/statistics, domain correctness, runtime behaviour, reporting integrity) ran against this codebase; the defects they confirmed are fixed and the fixes are documented in `docs/METHODOLOGY.md` and in code comments at each site |

---

## Reading order for a reviewer

1. `README.md` — what it does and how to run it
2. `report/FINAL_REPORT.md` — the result, with every number sourced
3. `docs/METHODOLOGY.md` — why it is built this way, including what changed after seeing the data
4. `docs/LITERATURE_REVIEW.md` — the prior work the result is positioned against
5. `src/config.py` — the single place where endpoints, thresholds and the evaluation protocol are defined
