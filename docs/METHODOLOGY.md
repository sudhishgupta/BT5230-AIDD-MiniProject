# Methodology and design decisions

This document records *why* the pipeline is built the way it is, including the
decisions that were changed after looking at the data. It is written so that a
reviewer can attack the design on its merits rather than having to reverse
engineer it from the code.

---

## 1. Endpoint selection

### 1.1 The constraint

The project brief is explicit: *"Select one clearly defined developability
endpoint for the primary task. Avoid mixing measurements obtained under
incompatible experimental conditions."*

This is not a formality. FLAb aggregates assays that measure different physics
in incompatible units — AC-SINS in nm of plasmon shift, PSR as a unitless 0–1
score, Tm in °C, HIC in minutes of retention. Concatenating them into one
regression target would produce a model that mostly learns *which assay a row
came from*.

We tested the weaker version of this claim empirically. On the identical 137
antibodies, the four endpoints correlate only weakly with one another
(`results/figures/eda_endpoint_correlations.png`). Knowing an antibody's HIC
behaviour tells you little about its thermostability or polyreactivity.
"Developability" is a label for a family of distinct failure modes, not a
quantity.

### 1.2 Why HIC retention time

Three reasons:

1. **It is a single, unambiguous readout.** One column, one gradient, one
   number in minutes. No composite scoring, no derived index.
2. **It is mechanistically interpretable.** HIC separates by solvent-exposed
   apolar surface under high kosmotropic salt. Long retention means a
   hydrophobic surface, which is a direct, physical developability liability
   (aggregation propensity, difficult purification).
3. **It is the only FLAb developability endpoint with three independent
   cohorts.** This is what makes genuine external validation possible rather
   than a reshuffle of the training set.

### 1.3 Secondary endpoints

AC-SINS, Fab Tm and PSR are run through the identical pipeline but **never
pooled** — each has its own training cohort, external cohort, risk threshold
and model. They test whether the *method* generalises across endpoints. They do
not enlarge the training set.

---

## 2. Data decisions

### 2.1 Reading the raw column, not `fitness`

FLAb provides a `fitness` column alongside the raw measurement. We verified on
every row of every file we use that `fitness` is a **verbatim copy** of the raw
value. Critically, it does **not** encode direction of merit: the
thermostability file (higher is better) and the HIC file (lower is better) are
stored identically.

Any pipeline that treats `fitness` as "higher is better" silently inverts three
of our four endpoints and reports backwards correlations that look like
failures of the model rather than failures of the data loader. We therefore
read the named raw assay column and apply direction explicitly from
`config.Endpoint.favourable_direction`.

The 2024 files have no `fitness` column at all, which would also break any code
that assumes it exists.

### 2.2 Demoting Jetha 2018 from validation set to stress test

**This was changed after auditing the data, and the change matters.**

The initial design treated `jetha2019homology_RT.csv` as a second external
validation cohort — it reports HIC retention time in minutes, like the others.
Auditing the file disqualified it on three independent grounds:

| problem | evidence |
|---|---|
| Not a diverse cohort | All 97 heavy chains are 120 aa with pairwise Hamming distance ≤ 3; 90 of 97 rows share one *identical* light chain. It is a point-mutant library of a single parent Fv. |
| Right-censored | 30 of 97 values (30.9%) sit at exactly 30.00 min — the column run-time ceiling. A third of the labels are inequalities, not measurements. |
| Different absolute scale | Median 27.59 min against 9.9 min for both Jain cohorts. Different column and gradient. |

The decisive demonstration: applying the published 11.7-min risk threshold to
this cohort flags **97 of 97** antibodies. The scales are simply not
transferable.

It is retained in the pipeline as an explicitly labelled out-of-distribution
**stress test** — useful for showing what happens outside the applicability
domain — and no validation claim is drawn from it. Its Spearman is also
reported on the uncensored subset alone, because a correlation dominated by one
tie block of 30 identical values is not measuring what it appears to.

A corollary worth recording: FLAb's own documented few-shot protocol uses a
random 80/10/10 split. On *this* dataset a random split leaks almost
completely, since near-neighbours of every test point are in training.

### 2.3 Censoring in the training endpoint

Two of the 137 training antibodies sit at exactly 25.0 min, having exceeded the
maximum measurement time. Prior work on this panel excluded them. We **retain**
them but flag them: they are the most hydrophobic antibodies in the set, and
removing the hardest examples would flatter the model. The flag allows the
headline result to be recomputed without them as a sensitivity check.

### 2.4 Risk threshold: published, not distributional

The binary label uses the **published** Jain et al. 2017 warning threshold of
**> 11.7 min**, defined there as the worst decile of the 48 *approved*
antibodies (not of all 137).

This is better than a quantile of our own training distribution for a specific
reason: a quantile guarantees the same positive rate in every cohort by
construction, which makes a cross-cohort classification comparison vacuous. A
fixed published threshold has a stable clinical meaning and transfers unchanged.

On our curated training panel it flags 15 of 137 (10.9%), reproducing the
intended ~10% rate — a useful consistency check on the curation. The
distributional cuts are retained as `risk_label_q80` / `risk_label_q90`
columns for sensitivity analysis.

---

## 3. Features

### 3.1 Numbering: AntPack, not ANARCI

ANARCI is the field standard but shells out to `hmmscan`; HMMER has no
supported Windows build and is not pip-installable. The alternatives were
surveyed:

| tool | verdict |
|---|---|
| `anarci` | pure-Python wrapper, needs external HMMER binary. Unusable on Windows. |
| **`antpack` 0.3.8.6.2** | **chosen.** One dependency (numpy), ships a cp313 Windows wheel, no licence key, IMGT/Kabat/Martin/Aho. |
| `antpack` ≥ 0.3.9 | needs an interactive `AntPack-setup` licence key — breaks unattended builds. |
| `antpack` 0.5 | yanked on PyPI. |
| `anarcii` | works, but pulls in torch and requires Python ≥ 3.11. |
| `abnumber` | defaults to the ANARCI backend; fails on Windows unless `use_anarcii=True`. |
| `riot-na` | pins `scikit-bio==0.6.2` (sdist-only, no Windows) and `biopython==1.84` (no cp313 wheel). Hard fail. |

AntPack was verified against published trastuzumab CDRs before being trusted.
All 277 antibodies numbered successfully.

**One antibody produced a framework warning** — `WDQGT` in FR4 instead of the
canonical `WGQGT`. The first implementation treated any AntPack message as a
failure and silently dropped that antibody's CDR features. Inspection showed
the alignment was entirely correct (numbering covered every residue, 95.5%
identity, sensible regions). The `ok` criterion was changed to require
structural soundness — recognised chain type, numbering covering the whole
sequence, plausible identity — and to keep the message as a warning. Discarding
real antibodies for having unusual frameworks is precisely the wrong behaviour
in a developability study, where unusual is the interesting case.

AntPack also supplies two things we did not plan for but use:
`LiabilitySearchTool` (a curated, severity-annotated, CDR-aware liability
scanner) and `VJGeneTool` (germline V/J assignment).

### 3.2 Descriptors

Three hydrophobicity scales are computed rather than one — Kyte-Doolittle,
Eisenberg and Black-Mould — specifically because the literature disagrees about
which is appropriate for HIC. Including all three turns a hidden assumption
into a measured result, and the measurement is reported.

The maximum windowed hydrophobicity is an explicit, labelled approximation: the
quantity that physically matters is a *solvent-exposed* hydrophobic patch, which
is three-dimensional. A sequence-only descriptor cannot distinguish a buried
leucine from an exposed one. This is stated as a limitation rather than papered
over, and it is the clearest reason structure-based models outperform here.

Charge is computed at pH 5.5 and 7.4 — a typical high-concentration formulation
buffer and physiological pH respectively.

The Fv charge-symmetry term (product of VH and VL net charge) is the
sequence-only analogue of TAP's structural SFvCSP.

### 3.3 Language models matched for width

AbLang2-paired emits 480 dimensions. ESM-2 35M emits 480 per chain. This
matching is deliberate: a comparison between a 480-d antibody model and, say, a
1280-d general model would confound "antibody-specific pre-training" with
"more capacity".

Heavy and light chains are embedded **separately** for ESM-2 and concatenated.
ESM-2 has no paired-chain separator token in its vocabulary, so joining the
chains with an artificial linker would put it out of distribution. AbLang2
genuinely models the pairing (it has a `|` separator token), and that is exactly
the difference the comparison is designed to detect.

**Pooling detail that matters:** ESM-2's mean pooling excludes the `<cls>` and
`<eos>` tokens explicitly, not just padding. Those two states carry no residue
identity but do vary with sequence length, so averaging over the raw attention
mask injects a length-correlated offset into every embedding — a spurious
feature when the target itself correlates with chain length.

AbLang2's `seqcoding` mode is used as the authors defined it (a length-masked
mean over the formatted `<H>|<L>` string). `align=True` is never used, because
it imports ANARCI.

---

## 4. Evaluation

### 4.1 Why cluster on CDR identity

Every antibody variable domain shares 70–85% framework identity with every
other. Whole-chain identity therefore cannot distinguish "near-duplicate" from
"unrelated" — everything looks similar. The discriminating signal is in the CDR
loops.

We align the six IMGT CDRs **region by region** (CDR-H1 against CDR-H1, not
against a concatenation) and compute identity as total matches over the summed
longer-of-pair lengths. Region-wise alignment stops a long CDR-H3 from
dominating and stops CDR-L1 of one antibody being aligned against CDR-H2 of
another.

Clustering is **single-linkage**, because the required property is transitive:
if A is a near-duplicate of B and B of C, all three must stay on the same side
of the split.

### 4.2 What the clustering actually found

Mean pairwise CDR identity in the training panel is **0.35**, and at the 0.80
cut 137 antibodies form 132 clusters whose largest member count is 2.

**The Jain panel is genuinely non-redundant.** Clustered and random splits
should therefore give nearly identical answers here — and the reported ablation
confirms they do.

This is worth stating plainly rather than hiding: on this dataset the usual
near-duplicate inflation is not the problem. We still use the clustered
protocol, on the principle that a protocol should be correct by construction
rather than because it was checked to be unnecessary — and the same check on
the Jetha cohort would show a very large gap.

### 4.3 Pool, do not average

With 137 antibodies and 5 folds, each test fold holds ~27 points. R² and
Spearman on 27 points are extremely noisy, and *averaging* per-fold R² is
biased, because each fold's variance denominator is computed on its own small
sample. We therefore pool out-of-fold predictions across each repeat and
compute one metric on all 137. The spread across the 10 repeats is reported as
the stability of the estimate.

### 4.4 Hyper-parameter selection criterion

Inner selection uses **negative MAE**, not RMSE. The HIC distribution has a long
right tail (median 9.9 min, max 25 min); RMSE would let a handful of extreme
antibodies decide the hyper-parameters. All four regression metrics are reported
regardless, so the choice is visible rather than buried.

### 4.5 Two permutation tests

- A **correlation-only** test permutes labels against fixed predictions. Cheap;
  tests whether the final ranking association could arise by chance.
- A **full-refit** test re-runs the entire nested cross-validation — feature
  set, model family, grid search and all — on permuted labels. Expensive, so it
  is run only for the headline configuration, but it is the honest test,
  because it also prices in the selection bias introduced by the grid search.

### 4.6 Paired comparison between models

Models are compared by **paired bootstrap on the same antibodies**. Resampling
jointly and taking the difference each time cancels the shared "which
antibodies did we happen to draw" variance that dominates two independent
marginal intervals. This is why two heavily overlapping CIs can still
correspond to a reliable difference, and why comparing CI overlap by eye is the
wrong test.

### 4.7 Multiple comparisons

The sweep compares 7 feature sets × up to 7 model families. The headline
configuration is chosen on cross-validated performance, which introduces
selection optimism even under nested CV. The guard is the external cohort,
which is never touched during model selection. All univariate and motif
analyses are Benjamini-Hochberg corrected, and "survives FDR" is reported
separately from "nominally significant".

### 4.8 Conformal prediction

Split-conformal calibration on out-of-fold residuals gives prediction intervals
with a finite-sample coverage guarantee under exchangeability only — no
distributional assumption, no asymptotics.

Exchangeability is exactly what a cohort shift breaks. We therefore **measure**
coverage on each external cohort and report it, rather than claiming the
guarantee transfers.

---

## 5. Interpretation

Four complementary analyses, because any single one is easy to fool at n=137:

1. **Chemical liability motifs** — do the canonical liabilities actually
   associate with this endpoint?
2. **Per-CDR amino-acid composition** — which residues, in which loop?
3. **IMGT position-wise enrichment** — with all chains aligned to a common
   numbering, which specific positions differ between the risky and safe tails?
4. **Model attribution** — what does the fitted model lean on?

Attribution is computed on the **descriptor** model rather than the overall best
model, even when a language model wins on accuracy. Ranking 480 anonymous
embedding dimensions produces something nobody can act on; "CDR aromatic
fraction" is a statement a protein engineer can test at the bench. The report
says which model won on accuracy and uses the descriptor model only to explain
the signal.

Position-level findings are framed as hypothesis generation. With several
hundred tests and a few dozen antibodies carrying any given residue at any
given position, FDR control bounds the expected false-discovery rate but does
not make any single position a validated determinant.

---

## 6. What would improve this most

In rough order of expected payoff:

1. **Structure.** The single largest known gap. Sequence features cannot resolve
   solvent exposure, and published work shows structure-derived hydrophobic
   patch descriptors substantially outperform sequence-only features on exactly
   this endpoint. A structure-prediction step followed by true patch descriptors
   is the obvious next move — with the caveat that published work also shows
   those descriptors are unstable across structure-prediction methods.
2. **More data on one platform.** 137 antibodies is the binding constraint.
   Recent high-throughput platforms have measured several hundred to several
   thousand antibodies on consistent assays; those would change what is
   achievable.
3. **Censored-aware loss.** A Tobit-style likelihood would use the censored
   observations correctly instead of treating an inequality as a point value.
4. **Fine-tuning rather than frozen embeddings.** We use AbLang2 and ESM-2 as
   frozen feature extractors. At n=137 fine-tuning would almost certainly
   overfit, but on a larger panel it is the natural extension.
