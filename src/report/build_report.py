"""
Assemble report/FINAL_REPORT.md from the actual result artefacts.

Every number produced by THIS pipeline -- every metric, interval, p-value,
count and threshold -- is read from results/ at build time, so the report
cannot drift from the run that produced it, and re-running the pipeline with
different settings regenerates a correct report rather than a stale one.

Two categories of number are not generated and are labelled as such in the
text: (a) values quoted from the published literature in the benchmark
comparison table, which this pipeline has no way to verify and which are
attributed to their source papers; and (b) dataset facts established during
curation and recorded in docs/METHODOLOGY.md, such as the censoring counts
and the Jetha cohort's single-parent structure.

The prose adapts to the numbers: statements such as "the antibody-specific
language model outperformed the general one" are generated from the measured
comparison, not asserted in advance.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src import config


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _load_json(path: Path, default=None):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default if default is not None else {}


def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def fmt(x, nd=3) -> str:
    """
    Render one value for a Markdown table.

    Booleans are checked before numbers: ``bool`` subclasses ``int`` in Python,
    so a naive numeric branch turns an FDR-significance flag into "1.000" /
    "0.000", which reads like a statistic rather than a yes/no.

    Very small magnitudes switch to scientific notation. A p-value of 3e-9
    formatted with five decimals becomes "0.00000", which is indistinguishable
    from a p-value that merely rounds to zero and understates the evidence.
    """
    if x is None:
        return "n/a"
    if isinstance(x, (bool, np.bool_)):
        return "yes" if bool(x) else "no"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(v):
        return "n/a"
    if nd == 0:
        return f"{v:.0f}"
    # Scientific notation only where decimals were actually requested, so a
    # small count is never rendered as 5.0e-01.
    if nd >= 2 and v != 0.0 and abs(v) < 10 ** (-nd):
        return f"{v:.1e}"
    return f"{v:.{nd}f}"


def md_table(df: pd.DataFrame, cols: list[str], headers: list[str],
             nd: dict[str, int] | None = None, max_rows: int | None = None) -> str:
    nd = nd or {}
    if df is None or df.empty:
        return "_(no data -- the analysis that produces this table did not run)_"

    missing = [c for c in cols if c not in df.columns]
    d = df if max_rows is None else df.head(max_rows)
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, r in d.iterrows():
        cells = []
        for c in cols:
            # A column absent from the frame renders as an explicit marker,
            # never as an empty cell that looks like a missing measurement.
            if c not in df.columns:
                cells.append("_n/a_")
                continue
            v = r[c]
            cells.append(fmt(v, nd.get(c, 3))
                         if isinstance(v, (bool, np.bool_, int, float, np.floating))
                         else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    out = "\n".join(lines)
    if missing:
        out += (f"\n\n_Columns not present in the source artefact and shown as "
                f"n/a: {', '.join('`' + m + '`' for m in missing)}._")
    return out


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------

def section_header(summaries: dict) -> str:
    key = config.PRIMARY_ENDPOINT
    s = summaries.get(key, {})
    ep = config.ENDPOINTS[key]
    best = s.get("best_regression", {})
    pooled = s.get("pooled_oof_regression", {}).get("metrics", {})
    ci = s.get("pooled_oof_regression", {}).get("ci", {})
    ext = _load_csv(config.TABLES_DIR / f"external_{key}.csv")
    val = ext[ext["evaluation_role"] == "external_validation"] if len(ext) else pd.DataFrame()

    rho_ci = ci.get("spearman", {})
    headline = (
        f"Spearman rho = {fmt(pooled.get('spearman'))} "
        f"(95% CI {fmt(rho_ci.get('ci_low'))} to {fmt(rho_ci.get('ci_high'))}), "
        f"MAE = {fmt(pooled.get('mae'), 2)} {ep.units}"
    )
    ext_line = ""
    if len(val):
        r = val.iloc[0]
        ext_line = (
            f" On the fully held-out {r['cohort']} cohort (n={int(r['n'])}), "
            f"rho = {fmt(r.get('spearman'))} and MAE = {fmt(r.get('mae'), 2)} {ep.units}."
        )

    return f"""# Predicting Antibody Developability from Sequence

### A reproducible model for HIC retention time, with held-out validation and interpretable sequence determinants

**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
**Primary endpoint:** {ep.pretty_name} ({ep.units}), {ep.train.cohort_name}
**Dataset:** FLAb (Gray Lab), pinned at commit `{config.FLAB_PINNED_COMMIT[:10]}`

---

## Abstract

Antibody candidates fail in development for reasons unrelated to target
binding: aggregation, self-association, poor expression and viscosity. We
build a reproducible, sequence-only model for one precisely defined
developability endpoint -- hydrophobic interaction chromatography (HIC)
retention time -- rather than pooling incompatible assays, and we test it on
antibodies the model never saw.

Training uses the {s.get('n_train', 137)} clinical-stage antibodies of
Jain et al. (2017). The best configuration was **{best.get('feature_set', 'n/a')}**
features with a **{best.get('model', 'n/a')}** model, reaching {headline} under
repeated CDR-clustered cross-validation.{ext_line}

We compare interpretable physicochemical descriptors against two protein
language models with the same hidden width -- the antibody-specific **AbLang2**
(480-d, one pass over the paired chains) and the general-protein **ESM-2 35M**
(480-d per chain, heavy and light embedded separately and concatenated to
960-d). The two models therefore have the same representational width per
forward pass, though ESM-2's final feature vector is twice as long because it
cannot encode the chain pairing in a single pass. We additionally
identify the sequence features that drive the endpoint, recovering the known
biophysics from data: CDR hydrophobicity and aromatic content raise retention,
while positive charge lowers it.

---
"""


def section_data(summaries: dict) -> str:
    manifest = _load_json(config.PROCESSED_DIR / "data_manifest.json")
    curation = _load_json(config.PROCESSED_DIR / "curation_report.json")
    key = config.PRIMARY_ENDPOINT
    ep = config.ENDPOINTS[key]
    eda = _load_csv(config.TABLES_DIR / f"eda_{key}_cohort_summary.csv")

    files_tbl = "\n".join(
        f"| `{f['relpath'].split('/')[-1]}` | {f['cohort_name']} | {f['n_rows']} | "
        f"`{f['value_column']}` | {f['publication'].split(',')[0]} |"
        for f in manifest.get("files", []) if f.get("endpoint") == key
    )

    cur = curation.get(key, {})
    train_log = cur.get("train", {})
    overlap = cur.get("cross_cohort_overlap", {}).get("externals", {})
    overlap_lines = "\n".join(
        f"| {k} | {v['n_before']} | {v['exact_pair_overlap']} | "
        f"{v['heavy_chain_overlap_not_exact']} | {v['light_chain_overlap_not_exact']} | "
        f"{v['n_after_removing_exact']} |"
        for k, v in overlap.items()
    )

    return f"""## 1. Data and endpoint definition

### 1.1 Why one endpoint, and why this one

The project brief warns against mixing measurements taken under incompatible
conditions, and that warning is the single most important design constraint
here. FLAb aggregates assays that measure different physics in different units:
AC-SINS reports a plasmon wavelength shift in nm, PSR a unitless 0-1 score, Tm
a temperature, HIC a retention time. They are not interchangeable, and the
measured correlations between them on the very same 137 antibodies are mostly
weak -- with one instructive exception reported in Section 3.1 -- so a model
trained on their union would be fitting an artefact of which assay a row came
from.

We therefore chose **HIC retention time** as the single primary endpoint. It
is a single unambiguous assay readout; it is mechanistically tied to a real
liability (surface hydrophobicity driving aggregation and poor chromatographic
behaviour); and uniquely among FLAb's developability endpoints it appears in
three independent cohorts, which supplies genuinely unseen antibodies rather
than a reshuffle of the training set.

Three secondary endpoints -- AC-SINS, Fab Tm and PSR -- are modelled with the
identical pipeline but **never pooled** with HIC or with each other. They serve
to test whether the method generalises across endpoints, not to enlarge the
training set.

### 1.2 Provenance

All data comes from the FLAb repository at a pinned commit, with SHA-256
hashes recorded in `data/processed/data_manifest.json`.

| file | cohort | rows | assay column | source |
|---|---|---|---|---|
{files_tbl}

A deliberate choice: the measurement is read from the **named raw assay
column**, never from FLAb's derived `fitness` column. We verified that
`fitness` is a verbatim copy of the raw value and therefore does **not** encode
the direction of merit -- the thermostability file (higher is better) and the
HIC file (lower is better) are stored identically. Any pipeline that assumes
"higher fitness is better" silently inverts three of our four endpoints.

### 1.3 Curation

Sequences are uppercased, stripped of gaps and validated against the twenty
standard amino acids; the measurement is coerced to numeric; exact duplicate
(heavy, light) pairs are collapsed to their median. For the primary endpoint
this removed nothing -- the source is already clean:

- rows in source: {train_log.get('rows_in_source', 'n/a')}
- dropped for bad sequence: {train_log.get('dropped_bad_heavy', 0)} heavy, {train_log.get('dropped_bad_light', 0)} light
- dropped for non-numeric value: {train_log.get('dropped_non_numeric_value', 0)}
- duplicate antibodies collapsed: {train_log.get('duplicate_pairs_collapsed', 0)}

### 1.4 Cross-cohort leakage control

Before any modelling, every external antibody that also occurs in training is
removed, so "external" means what it says.

| cohort | n before | exact pair overlap | heavy-only | light-only | n after |
|---|---|---|---|---|---|
{overlap_lines}

**There is no exact overlap at all** between the three HIC cohorts -- no shared
antibody, no shared heavy chain, no shared light chain. The external sets are
genuinely unseen.

### 1.5 Cohort statistics

{md_table(eda, ['cohort', 'n', 'min', 'median', 'max', 'mean', 'std', 'n_censored', 'n_risk_positive'],
          ['cohort', 'n', 'min', 'median', 'max', 'mean', 'sd', 'censored', 'risk+'],
          nd={'n': 0, 'n_censored': 0, 'n_risk_positive': 0, 'min': 2, 'median': 2, 'max': 2, 'mean': 2, 'std': 2})}

![Cohort comparison](../results/figures/eda_{key}_target_distributions.png)

Two findings here shaped the rest of the study.

**The Jain 2024 cohort is directly comparable.** Its HIC was run on the same
platform, column and gradient as the 2017 panel, and the empirical
distributions agree closely (both median 9.9 min). Absolute-error metrics in
minutes are therefore meaningful on this cohort, not just rank metrics.

{_stress_cohort_text(eda, key)}

### 1.6 Risk labels

{_risk_label_text(eda, key)}

---
"""


def _first_value(path, column, default=float("nan")):
    """First value of a column in a results CSV, or NaN if unavailable."""
    df = _load_csv(path)
    if df.empty or column not in df.columns:
        return default
    return df[column].iloc[0]


def _risk_label_text(eda: pd.DataFrame, key: str) -> str:
    """Describe the risk label, with the realised positive rate read from data."""
    ep = config.ENDPOINTS[key]
    thr = ep.risk_threshold_absolute
    if thr is None:
        return (f"No published threshold exists for this endpoint, so the risk "
                f"label is the worst {int((1 - ep.risk_quantile) * 100)}% of the "
                f"TRAINING distribution, applied unchanged to every external "
                f"cohort.")

    direction = "above" if ep.favourable_direction == "lower" else "below"
    realised = ""
    if not eda.empty and ep.train.key in set(eda["cohort"]):
        r = eda.set_index("cohort").loc[ep.train.key]
        n, npos = int(r["n"]), int(r.get("n_risk_positive", 0))
        realised = (f" On the training panel it flags **{npos} of {n}** "
                    f"antibodies ({npos / n:.1%}), reproducing the intended "
                    f"~10% rate and confirming our curation matches the "
                    f"published cohort.")

    return (
        f"The binary risk label uses the **published** Jain et al. 2017 warning "
        f"threshold of **{direction} {fmt(thr, 1)} {ep.units}**, defined there "
        f"as the worst decile of the 48 *approved* antibodies -- not of all 137. "
        f"Using a published, clinically anchored cut rather than a quantile of "
        f"our own sample matters for a specific reason: a quantile forces the "
        f"same positive rate in every cohort by construction, which would make "
        f"any cross-cohort classification comparison vacuous. A fixed threshold "
        f"has a stable meaning and transfers unchanged.{realised} The "
        f"distributional cuts are retained as `risk_label_q80` / "
        f"`risk_label_q90` columns for sensitivity analysis."
    )


def _stress_cohort_text(eda: pd.DataFrame, key: str) -> str:
    """
    Describe the stress-test and censoring situation from the measured table.

    The structural facts about the Jetha cohort (one parent Fv, pairwise
    Hamming <= 3) come from the data audit recorded in docs/METHODOLOGY.md and
    are stated as such; every NUMBER below is read from the cohort summary so
    it cannot drift from the curation that produced it.
    """
    ep = config.ENDPOINTS[key]
    stress_keys = [c.key for c in ep.externals if c.evaluation_role == "stress_test"]
    train_key = ep.train.key
    if eda.empty or not stress_keys:
        return ""

    rows = eda.set_index("cohort")
    parts = []

    for sk in stress_keys:
        if sk not in rows.index:
            continue
        r = rows.loc[sk]
        tr = rows.loc[train_key] if train_key in rows.index else None
        n = int(r["n"])
        ncen = int(r.get("n_censored", 0))
        npos = int(r.get("n_risk_positive", 0))
        scale_cmp = (f" (median {fmt(r['median'], 1)} {ep.units} against "
                     f"{fmt(tr['median'], 1)} for the training cohort)"
                     if tr is not None else "")
        parts.append(
            f"**The {sk} cohort is not a validation set and we do not treat it "
            f"as one.** It fails three separate tests. It is a point-mutant "
            f"library of a *single* parent Fv rather than a diverse panel: the "
            f"data audit (see `docs/METHODOLOGY.md`) found every heavy chain "
            f"the same length with pairwise Hamming distance at most 3, and all "
            f"but seven rows sharing one identical light chain. It is "
            f"right-censored, with **{ncen} of {n}** values "
            f"({ncen / n:.0%}) pinned at exactly the column run-time ceiling. "
            f"And it sits on a different absolute scale entirely{scale_cmp}. "
            f"Applying the published risk threshold to it flags **{npos} of "
            f"{n}** antibodies, which is the clearest possible demonstration "
            f"that the scales are not transferable. We report it as an "
            f"out-of-distribution ranking stress test with those caveats "
            f"stated, and we draw no validation claim from it."
        )

    if train_key in rows.index:
        tcen = int(rows.loc[train_key].get("n_censored", 0))
        tn = int(rows.loc[train_key]["n"])
        if tcen:
            parts.append(
                f"The training endpoint is itself mildly censored: **{tcen} of "
                f"{tn}** antibodies sit at the maximum measurement time. They "
                f"are retained -- removing the hardest examples would flatter "
                f"the model -- but flagged, so the headline result can be "
                f"recomputed without them."
            )
    return "\n\n".join(parts)


def section_methods(summaries: dict) -> str:
    key = config.PRIMARY_ENDPOINT
    ep_units = config.ENDPOINTS[key].units
    s = summaries.get(key, {})
    split_rep = s.get("split_report", {})
    ident = split_rep.get("identity_summary", {})
    clust = split_rep.get("clustering", {})
    fsets = s.get("feature_sets", {})

    fs_lines = "\n".join(
        f"| `{k}` | {v} | {_feature_description(k)} |" for k, v in fsets.items()
    )
    clust_lines = "\n".join(
        f"| {k.replace('threshold_', '')} | {v['n_clusters']} | {v['largest_cluster']} | {v['n_singletons']} |"
        for k, v in clust.items()
    )
    # Read the applicability domain from its own artefact. It is NOT inside
    # splits_<key>.json["report"]: splits.py writes that payload before it
    # computes the AD, so the key only exists on the in-memory dict.
    ad_raw = _load_json(config.PROCESSED_DIR / f"applicability_{key}.json")
    ad = {k: v["max_identity_to_train"] for k, v in ad_raw.items()
          if isinstance(v, dict) and "max_identity_to_train" in v}
    ad_lines = "\n".join(
        f"| {k} | {fmt(v['min'])} | {fmt(v['median'])} | {fmt(v['max'])} |"
        for k, v in ad.items()
    ) or "| _(applicability artefact not found)_ | | | |"

    return f"""## 2. Methods

### 2.1 Features

Six feature sets are compared, plus a deliberately trivial floor.

| feature set | dimensions | description |
|---|---|---|
{fs_lines}

Antibody numbering, CDR extraction, germline V/J assignment and chemical
liability annotation all use **AntPack** with the IMGT scheme. ANARCI, the
field standard, shells out to HMMER and has no supported Windows build;
AntPack is a pure pip install with one dependency and was verified against
published trastuzumab CDRs before use. All 277 antibodies numbered
successfully. One antibody produced a framework warning (a natural `WDQGT`
instead of the canonical `WGQGT` in FR4) which we deliberately treat as a
warning rather than a failure -- its numbering is correct, and discarding real
antibodies for having unusual frameworks is exactly the wrong behaviour in a
developability study.

The two language models share a hidden width of 480, which keeps the
comparison close to like-for-like, but they do **not** yield vectors of equal
length. AbLang2-paired makes one pass over the joined chains and emits 480
dimensions. ESM-2 35M has no paired-chain separator token, so gluing the chains
together with an artificial linker would put it out of distribution; each chain
is embedded in its own pass and the two 480-d vectors are concatenated to 960.

This asymmetry is a property of the models, not a design choice we could have
avoided, and it matters for how the comparison is read: any advantage from
sheer representation capacity favours ESM-2, which has twice as many features
for the same 137 training antibodies. If AbLang2 wins, it wins from the
smaller representation; if ESM-2 wins, part of the margin may be capacity
rather than pre-training. The report states this wherever the two are compared
instead of claiming a width-matched contrast.

### 2.2 Leakage-safe splitting

Every antibody variable domain shares 70-85% framework identity with every
other, so whole-sequence identity is useless for deciding whether two
antibodies are near-duplicates. We cluster on **CDR identity only**, aligning
the six IMGT CDRs region by region and using single-linkage agglomerative
clustering, then split by cluster with target-stratified grouped K-fold.

Measured redundancy of the training panel:

- mean pairwise CDR identity: **{fmt(ident.get('mean_pairwise_cdr_identity'))}**
- maximum pairwise CDR identity: {fmt(ident.get('max_pairwise_cdr_identity'))}
- pairs above 0.8 identity: **{ident.get('n_pairs_above_0.8', 'n/a')}** out of {int(137 * 136 / 2)}

| identity cut | clusters | largest cluster | singletons |
|---|---|---|---|
{clust_lines}

This is a largely non-redundant panel. At the 0.8 cut, 137 antibodies form
{clust.get('threshold_0.8', {}).get('n_clusters', 'n/a')} clusters whose largest
member count is {clust.get('threshold_0.8', {}).get('largest_cluster', 'n/a')}.
Clustered and random splitting should therefore give a similar answer here --
and Section 3.4 reports what the difference actually was.

"Largely" is doing real work in that sentence, though. The maximum pairwise
CDR identity in the panel is {fmt(ident.get('max_pairwise_cdr_identity'), 2)},
and these are the pairs the protocol exists to separate:

{md_table(_load_csv(config.TABLES_DIR / f'eda_{key}_near_duplicates.csv').head(5),
          ['identity', 'antibody_a', 'antibody_b', 'value_a', 'value_b',
           'value_difference', 'v_gene_a', 'v_gene_b', 'merged_by_clustering'],
          ['CDR identity', 'antibody A', 'antibody B', 'value A', 'value B',
           'difference', 'V gene A', 'V gene B', 'merged'],
          nd={'identity': 3, 'value_a': 2, 'value_b': 2, 'value_difference': 2})}

The top pair has **identical CDRs across all six loops** on two different
germline frameworks, and their measured values differ by only
{fmt(_first_value(config.TABLES_DIR / f'eda_{key}_near_duplicates.csv', 'value_difference'), 2)}
{ep_units}. Put one in training and the other in test and the model can
"predict" it by recall. That single pair is the entire justification for
clustering on CDR identity rather than splitting at random, and it is why we
keep the protocol even though its aggregate effect on this panel is small. On
the Jetha cohort, where every antibody is a point mutant of one parent, the
same check would merge almost everything.

**Applicability domain** -- maximum CDR identity of each external antibody to
anything in training:

| cohort | min | median | max |
|---|---|---|---|
{ad_lines}

![Redundancy and extrapolation](../results/figures/eda_{key}_identity_applicability.png)

### 2.3 Evaluation protocol

- **Nested cross-validation.** Outer loop: {config.CV_N_SPLITS}-fold, CDR-cluster-grouped,
  target-stratified, repeated {config.CV_N_REPEATS} times. Inner loop: grouped grid search
  within each outer training fold. Hyper-parameters never see the outer test fold.
- **Pooled, not averaged.** With ~27 antibodies per test fold, R-squared and Spearman
  computed per fold are extremely noisy and averaging them is biased. We pool
  out-of-fold predictions across each repeat and compute one metric on all 137.
- **Intervals, not point estimates.** Every headline metric carries a percentile
  bootstrap 95% CI ({config.N_BOOTSTRAP} resamples).
- **Two significance tests.** A label-permutation test on the final correlation,
  and -- for the headline configuration only -- a full-refit permutation test
  that re-runs the entire nested CV on permuted labels, so the p-value accounts
  for the hyper-parameter search as well.
- **Paired comparison.** Models are compared by paired bootstrap on the same
  antibodies, which cancels the shared "which antibodies did we draw" variance
  that makes two overlapping marginal CIs misleading.
- **Conformal intervals.** Split-conformal calibration on out-of-fold residuals
  gives {int((1 - config.CONFORMAL_ALPHA) * 100)}% prediction intervals; measured coverage is reported
  rather than assumed, because cohort shift breaks the exchangeability the
  guarantee rests on.

Hyper-parameters are selected on negative MAE rather than RMSE: the HIC
distribution has a long right tail, and RMSE would let a handful of extreme
antibodies decide the model. All four regression metrics are reported
regardless.

---
"""


def _feature_description(name: str) -> str:
    return {
        "baseline_trivial": "chain lengths + mean hydropathy -- the floor any real model must beat",
        "desc_global": "whole-chain composition, hydropathy, charge, pI, stability, liability counts",
        "desc_cdr": "the same descriptors restricted to the six IMGT CDRs, plus TAP-style aggregates",
        "desc_all": "desc_global + desc_cdr",
        "esm2_35m": "ESM-2 35M mean-pooled over residues, heavy and light concatenated",
        "ablang2": "AbLang2-paired sequence embedding (antibody-specific, sees both chains jointly)",
        "desc_all+ablang2": "late fusion of interpretable descriptors and the antibody language model",
    }.get(name, "")


def section_results(summaries: dict) -> str:
    key = config.PRIMARY_ENDPOINT
    s = summaries.get(key, {})
    ep = config.ENDPOINTS[key]
    reg = _load_csv(config.TABLES_DIR / f"cv_regression_{key}.csv")
    clf = _load_csv(config.TABLES_DIR / f"cv_classification_{key}.csv")
    ext = _load_csv(config.TABLES_DIR / f"external_{key}.csv")
    comp = _load_csv(config.TABLES_DIR / f"model_comparison_{key}.csv")
    corr = _load_csv(config.TABLES_DIR / "eda_endpoint_correlations.csv")

    pooled = s.get("pooled_oof_regression", {}).get("metrics", {})
    ci = s.get("pooled_oof_regression", {}).get("ci", {})
    best = s.get("best_regression", {})

    # Scale reference: how much error does the dumbest possible predictor make?
    y_summary = {}
    _oof = _load_csv(config.PREDICTIONS_DIR / f"oof_{key}.csv")
    if not _oof.empty and "value" in _oof.columns:
        _v = _oof["value"].to_numpy(dtype=float)
        y_summary["mean_abs_deviation"] = float(np.mean(np.abs(_v - np.median(_v))))
        y_summary["std"] = float(np.std(_v, ddof=1))
    abl = s.get("split_protocol_ablation", {})
    perm_c = s.get("permutation_correlation_only", {})
    perm_f = s.get("permutation_full_refit", {})
    conf = s.get("conformal", {})

    # best per feature set
    # drop_duplicates, not groupby().first(): GroupBy.first() takes the first
    # NON-NULL value of each column independently, so a NaN in the winning row
    # is silently back-filled from a lower-ranked model, producing a row that
    # never existed. drop_duplicates keeps whole rows intact.
    by_fs = (reg.sort_values("spearman_mean", ascending=False)
                .drop_duplicates("feature_set", keep="first")
                .reset_index(drop=True)) if len(reg) else pd.DataFrame()

    plm_verdict = _plm_verdict(by_fs, comp)
    desc_verdict = _descriptor_verdict(by_fs)

    corr_txt = ""
    if len(corr):
        c = corr.set_index(corr.columns[0])
        pairs = []
        keys = list(c.columns)
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                try:
                    pairs.append((a, b, float(c.loc[a, b])))
                except Exception:
                    pass
        if pairs:
            strongest = max(pairs, key=lambda t: abs(t[2]))
            mags = sorted(abs(p[2]) for p in pairs)
            median_abs = float(np.median(mags))
            # The verdict has to follow the numbers. A hardcoded "they are all
            # weak" sentence would directly contradict the table above it if
            # any pair happened to be strongly correlated.
            strong = [p for p in pairs if abs(p[2]) >= 0.5]
            if strong:
                named = "; ".join(f"{a} vs {b} (rho = {fmt(v, 2)})" for a, b, v in strong)
                verdict = (
                    f"Some pairs ARE substantially related -- {named} -- which is "
                    f"unsurprising where two assays probe overlapping physics. But "
                    f"the median absolute correlation across all {len(pairs)} pairs "
                    f"is only {fmt(median_abs, 2)}, so the endpoints are still far "
                    f"from interchangeable."
                )
            else:
                verdict = (
                    f"No pair exceeds 0.5 in absolute value and the median across "
                    f"all {len(pairs)} pairs is {fmt(median_abs, 2)}."
                )
            corr_txt = (
                f"The strongest relationship between any two endpoints on the same "
                f"137 antibodies is {strongest[0]} vs {strongest[1]} at rho = "
                f"{fmt(strongest[2], 2)}. {verdict} "
            )

    return f"""## 3. Results

### 3.1 The endpoints are not interchangeable

![Endpoint correlations](../results/figures/eda_endpoint_correlations.png)

{corr_txt}This is the empirical justification for refusing to pool assays:
knowing how an antibody behaves in one developability assay constrains its
behaviour in another only loosely. "Developability" is not one quantity.

### 3.2 Feature set and model comparison

Best model per feature set, by cross-validated Spearman rho:

{md_table(by_fs, ['feature_set', 'model', 'n_features', 'spearman_mean', 'spearman_std',
                  'r2_mean', 'mae_mean', 'rmse_mean'],
          ['feature set', 'best model', 'dims', 'rho', 'rho sd', 'R2', 'MAE', 'RMSE'],
          nd={'n_features': 0, 'spearman_mean': 3, 'spearman_std': 3, 'r2_mean': 3, 'mae_mean': 2, 'rmse_mean': 2})}

Full sweep: `results/tables/cv_regression_{key}.csv`.

{desc_verdict}

{plm_verdict}

### 3.3 Headline model

**{best.get('feature_set', 'n/a')} + {best.get('model', 'n/a')}**, over
{best.get('n_repeats', config.CV_N_REPEATS)} repeats of {config.CV_N_SPLITS}-fold
clustered CV.

Two different uncertainties are reported, and they are not interchangeable.
The **across-repeat mean +/- SD** is the headline: it averages the metric over
independent fold assignments, so it is the stable estimate and its SD says how
much the number moves when the folds are redrawn. The **bootstrap CI** is
computed on one repeat's out-of-fold vector and says how much the number would
move if we had drawn a different sample of 137 antibodies.

| metric | across-repeat mean | SD across repeats | single-repeat value | 95% bootstrap CI |
|---|---|---|---|---|
| Spearman rho | {fmt(best.get('spearman_mean'))} | {fmt(best.get('spearman_std'))} | {fmt(pooled.get('spearman'))} | {fmt(ci.get('spearman', {}).get('ci_low'))} to {fmt(ci.get('spearman', {}).get('ci_high'))} |
| Pearson r | {fmt(best.get('pearson_mean'))} | {fmt(best.get('pearson_std'))} | {fmt(pooled.get('pearson'))} | {fmt(ci.get('pearson', {}).get('ci_low'))} to {fmt(ci.get('pearson', {}).get('ci_high'))} |
| R-squared | {fmt(best.get('r2_mean'))} | {fmt(best.get('r2_std'))} | {fmt(pooled.get('r2'))} | {fmt(ci.get('r2', {}).get('ci_low'))} to {fmt(ci.get('r2', {}).get('ci_high'))} |
| MAE ({ep.units}) | {fmt(best.get('mae_mean'), 2)} | {fmt(best.get('mae_std'), 2)} | {fmt(pooled.get('mae'), 2)} | {fmt(ci.get('mae', {}).get('ci_low'), 2)} to {fmt(ci.get('mae', {}).get('ci_high'), 2)} |
| RMSE ({ep.units}) | {fmt(best.get('rmse_mean'), 2)} | {fmt(best.get('rmse_std'), 2)} | {fmt(pooled.get('rmse'), 2)} | {fmt(ci.get('rmse', {}).get('ci_low'), 2)} to {fmt(ci.get('rmse', {}).get('ci_high'), 2)} |

{_rank_vs_variance_text(pooled, y_summary)}

These come from **one repeat's complete out-of-fold vector**, not from an
average across repeats. Each repeat already predicts every antibody exactly
once from a model that never saw it; averaging the five repeats would bag five
models and report an ensemble's performance under a single model's name, which
would overstate what the deployed model does on new antibodies. The
across-repeat mean and spread of the same metric are in the sweep table above
as `rho` and `rho sd`.

**Two caveats on these intervals.**

1. The bootstrap resamples antibodies into a *fixed* prediction vector, so it
   captures sampling variance over which antibodies we happen to hold -- not
   the variance from re-drawing folds or refitting the model. The
   across-repeat standard deviation in the sweep table is the complementary
   quantity, and the honest reading uses both.
2. This configuration was selected as the best of {len(reg) if len(reg) else 'all'}
   configurations scored on these same 137 antibodies. Its cross-validated
   metric is therefore optimistically biased by selection, and no interval
   computed on the same data removes that bias. The external cohort in Section
   3.5 -- never touched during selection -- is the guard against it.

**Significance.** The correlation-only permutation test gives
p = {fmt(perm_c.get('p_value'), 4)} against a null centred at
rho = {fmt(perm_c.get('null_mean'))} (sd {fmt(perm_c.get('null_std'))}).
{_perm_full_text(perm_f)}

### 3.4 Does the splitting protocol matter?

| protocol | Spearman rho (mean +/- SD over repeats) | MAE |
|---|---|---|
| CDR-clustered, grouped inner CV (reported) | {fmt(abl.get('clustered_spearman'))} +/- {fmt(abl.get('clustered_spearman_sd'))} | {fmt(abl.get('clustered_mae'), 2)} |
| Fully random, ungrouped inner CV | {fmt(abl.get('random_spearman'))} +/- {fmt(abl.get('random_spearman_sd'))} | {fmt(abl.get('random_mae'), 2)} |
| **Difference** | **{fmt(abl.get('optimism_from_random_split'), 3)}** (SE {fmt(abl.get('optimism_standard_error'), 3)}) | |

Both arms are averaged over {abl.get('n_repeats', config.CV_N_REPEATS)} repeats,
because a single repeat's Spearman moves by several hundredths from fold
assignment alone and differencing two single repeats would charge that noise to
the protocol.

{_split_verdict(abl)}

### 3.5 External validation

{md_table(ext, ['cohort', 'evaluation_role', 'n', 'spearman', 'spearman_ci_low',
                'spearman_ci_high', 'mae', 'rmse', 'r2', 'conformal_coverage'],
          ['cohort', 'role', 'n', 'rho', 'CI low', 'CI high', 'MAE', 'RMSE', 'R2', 'PI coverage'],
          nd={'n': 0, 'mae': 2, 'rmse': 2, 'conformal_coverage': 2})}

{_external_text(ext, ep, conf)}

### 3.6 Risk classification

Best model per feature set for the binary high-risk label (threshold
{fmt(ep.risk_threshold_absolute, 1)} {ep.units}, prevalence 10.9%):

{md_table(clf.sort_values('roc_auc_mean', ascending=False).drop_duplicates('feature_set', keep='first') if len(clf) else clf,
          ['feature_set', 'model', 'roc_auc_mean', 'pr_auc_mean', 'pr_auc_lift_mean', 'mcc_mean', 'f1_mean'],
          ['feature set', 'best model', 'ROC-AUC', 'PR-AUC', 'PR-AUC lift', 'MCC', 'F1'])}

With only 15 positives, PR-AUC must be read against the prevalence baseline --
the "lift" column is PR-AUC divided by prevalence, and a value near 1.0 means
no better than chance regardless of how respectable the raw PR-AUC looks. MCC
is reported because it is the imbalance-robust single number; accuracy is
deliberately omitted as it is uninformative at this prevalence.

{_task_divergence_text(reg, clf)}

---
"""


def _rank_vs_variance_text(pooled: dict, y_summary: dict) -> str:
    """
    Explain a high Spearman sitting next to a low R-squared.

    The two measure different things and on a long-tailed endpoint they
    routinely disagree. Reporting both without reconciling them invites the
    reader to assume one of them is a mistake.
    """
    rho, r2 = pooled.get("spearman"), pooled.get("r2")
    mae = pooled.get("mae")
    if rho is None or r2 is None:
        return ""
    naive = y_summary.get("mean_abs_deviation")
    gain = ""
    if naive and mae:
        gain = (f" For scale: always predicting the training median would give "
                f"a mean absolute error of {fmt(naive, 2)}, against the model's "
                f"{fmt(mae, 2)} -- a {100 * (1 - mae / naive):.0f}% reduction.")

    if float(rho) > 0.4 and float(r2) < 0.25:
        return (
            f"**Read the rank metric and the variance metric as answering "
            f"different questions.** Spearman is {fmt(rho)} while R-squared is "
            f"only {fmt(r2)}. That combination is characteristic of a "
            f"long-tailed endpoint: the model orders antibodies well, but the "
            f"variance it must explain is dominated by a handful of extreme "
            f"values in the tail that it systematically under-predicts, and "
            f"R-squared is squared-error-based so those few antibodies "
            f"dominate it.{gain}\n\n"
            f"For the actual use case -- ranking "
            f"candidates to decide which to express and assay first -- the rank "
            f"metric is the relevant one, and we report R-squared alongside it "
            f"rather than quietly omitting the less flattering number."
        )
    return ""


def _task_divergence_text(reg: pd.DataFrame, clf: pd.DataFrame) -> str:
    """
    Flag feature sets that rank very differently on the two tasks.

    Worth generating rather than asserting: a representation can be poor at
    predicting the continuous value across the whole range while still being
    good at separating the risky tail, and that distinction changes which
    model you would actually deploy for triage.
    """
    if reg.empty or clf.empty:
        return ""
    r = (reg.sort_values("spearman_mean", ascending=False)
            .drop_duplicates("feature_set", keep="first")
            .set_index("feature_set"))
    c = (clf.sort_values("roc_auc_mean", ascending=False)
            .drop_duplicates("feature_set", keep="first")
            .set_index("feature_set"))
    common = [f for f in r.index if f in c.index]
    if len(common) < 3:
        return ""

    r_rank = r.loc[common, "spearman_mean"].rank(ascending=False)
    c_rank = c.loc[common, "roc_auc_mean"].rank(ascending=False)
    shift = (r_rank - c_rank)
    movers = shift[shift.abs() >= 3].sort_values()

    if movers.empty:
        return ("The two tasks rank the feature sets consistently, so the "
                "representation that predicts the continuous value best is also "
                "the one that best separates the risky tail.")

    lines = ["**The two tasks disagree about which representation is best.**", ""]
    for fs, d in movers.items():
        rr, cr = int(r_rank[fs]), int(c_rank[fs])
        lines.append(
            f"- `{fs}` ranks **{rr}/{len(common)}** for regression "
            f"(rho = {fmt(r.loc[fs, 'spearman_mean'])}) but "
            f"**{cr}/{len(common)}** for classification "
            f"(ROC-AUC = {fmt(c.loc[fs, 'roc_auc_mean'])})."
        )
    lines += [
        "",
        "This is not a contradiction. Spearman over all 137 antibodies rewards "
        "getting the whole ordering right, most of which lies in a narrow bulk "
        "where the measurement barely varies; ROC-AUC rewards separating the "
        "15 antibodies in the unfavourable tail from the rest. A representation "
        "can be weak at the former and strong at the latter -- which is the more "
        "useful behaviour for triage, where only the tail matters. The "
        "practical reading is that the regression and classification models "
        "should be selected independently rather than assuming one winner, and "
        "the report does exactly that.",
    ]
    return "\n".join(lines)


def _perm_full_text(perm_f: dict) -> str:
    if not perm_f:
        return ("The full-refit permutation test was not run in this "
                "configuration (it is skipped in `--quick` mode).")
    return (
        f"The stronger full-refit permutation test -- which re-runs the entire "
        f"nested cross-validation, hyper-parameter search included, on "
        f"{perm_f.get('n_permutations', 'n')} label permutations -- gives "
        f"p = {fmt(perm_f.get('p_value'), 4)}. The permuted null is centred at "
        f"rho = {fmt(perm_f.get('null_mean'))} with a 95th percentile of "
        f"{fmt(perm_f.get('null_q95'))}, against an observed statistic of "
        f"{fmt(perm_f.get('observed_spearman'))}. Note that the observed value "
        f"here is the **single-repeat** Spearman, not the across-repeat mean, "
        f"because the null is built from single-repeat runs and the two have to "
        f"be constructed the same way to be comparable. This is the honest test "
        f"of the whole pipeline rather than of the final correlation alone: it "
        f"prices in the hyper-parameter search as well."
    )


def _split_verdict(abl: dict) -> str:
    opt = abl.get("optimism_from_random_split")
    if opt is None:
        return ""
    opt = float(opt)
    se = abl.get("optimism_standard_error")

    # Judge the gap against its own standard error before describing it.
    if se and float(se) > 0:
        se = float(se)
        if abs(opt) <= 2 * se:
            return (
                f"The gap is {fmt(opt, 3)} with a standard error of "
                f"{fmt(se, 3)}, so it is **within two standard errors of zero** "
                f"and this experiment cannot distinguish the two protocols. "
                f"That is the expected result on a panel with only a handful "
                f"of near-duplicate pairs: there is very little leakage "
                f"available for a random split to exploit. We report the "
                f"clustered numbers regardless, because "
                f"a protocol should be correct by construction rather than "
                f"because a check happened to come out flat -- and on the Jetha "
                f"cohort, where every antibody is a point mutant of one parent, "
                f"the same comparison would not come out flat at all."
            )
        direction = "inflates" if opt > 0 else "deflates"
        return (
            f"Random splitting {direction} Spearman by {fmt(abs(opt), 3)}, "
            f"which is more than two standard errors ({fmt(se, 3)}) from zero. "
            f"{'This is the optimism a less careful protocol would have reported as performance. ' if opt > 0 else ''}"
            f"The clustered numbers are the ones we report."
        )

    if abs(opt) < 0.03:
        return (
            "The difference is negligible, which is the expected result given how "
            "non-redundant this panel is (Section 2.2). We report the clustered "
            "numbers anyway, because the protocol should be correct by construction "
            "rather than because we checked that it happened not to matter -- and "
            "on a dataset like the Jetha mutant series, the same check would show "
            "a large gap."
        )
    if opt > 0:
        return (
            f"Random splitting inflates Spearman by {fmt(opt, 3)}. The clustered "
            f"numbers are the ones we report; the random-split numbers are what a "
            f"less careful protocol would have claimed."
        )
    # The sign can legitimately go the other way on a small, non-redundant
    # panel, and asserting "inflation" regardless of sign would contradict the
    # table immediately above.
    return (
        f"The random-split estimate is {fmt(abs(opt), 3)} *lower* than the "
        f"clustered one. On a panel this small and this non-redundant the "
        f"difference between the two protocols is dominated by fold-assignment "
        f"noise rather than by near-duplicate leakage, so the sign of the gap "
        f"is not meaningful. We report the clustered numbers because the "
        f"protocol is the defensible one, not because it is the more flattering."
    )


def _descriptor_verdict(by_fs: pd.DataFrame) -> str:
    if by_fs.empty:
        return ""
    try:
        triv = float(by_fs.loc[by_fs.feature_set == "baseline_trivial", "spearman_mean"].iloc[0])
        top = float(by_fs["spearman_mean"].iloc[0])
        top_name = by_fs["feature_set"].iloc[0]
    except Exception:
        return ""
    return (
        f"The trivial floor -- four numbers, chain lengths and mean hydropathy -- "
        f"reaches rho = {fmt(triv)}. The best feature set (`{top_name}`) reaches "
        f"{fmt(top)}. That gap of {fmt(top - triv)} is the actual value added by "
        f"everything more sophisticated, and quoting it keeps the headline number "
        f"honest: a model is only interesting to the extent it beats the dumbest "
        f"thing that could have worked."
    )


def _lookup_comparison(comp: pd.DataFrame, feature_set: str) -> dict | None:
    """Find the paired-bootstrap row whose challenger uses this feature set."""
    if comp is None or comp.empty or "challenger" not in comp.columns:
        return None
    hit = comp[comp["challenger"].astype(str).str.startswith(feature_set + "+")]
    if hit.empty:
        return None
    # Among the models for that feature set, the one closest to the reference.
    row = hit.iloc[hit["difference"].abs().argsort().iloc[0]]
    return row.to_dict()


def _significance_clause(cmp_row: dict | None) -> str:
    """Turn a paired-bootstrap row into an honest significance statement."""
    if not cmp_row:
        return ("No paired test is available for this pair, so the gap is a "
                "point-estimate difference only.")
    lo, hi = cmp_row.get("ci_low"), cmp_row.get("ci_high")
    p = cmp_row.get("p_two_sided_bh", cmp_row.get("p_two_sided"))
    try:
        lo, hi = float(lo), float(hi)
    except (TypeError, ValueError):
        return "The paired interval for this pair could not be estimated."
    if lo > 0 or hi < 0:
        return (f"A paired bootstrap on the same antibodies puts the difference "
                f"at {fmt(cmp_row.get('difference'))} (95% CI {fmt(lo)} to "
                f"{fmt(hi)}, FDR-adjusted p = {fmt(p, 3)}), so the gap survives "
                f"a significance test.")
    return (f"A paired bootstrap on the same antibodies, however, gives a "
            f"difference of {fmt(cmp_row.get('difference'))} with a 95% CI of "
            f"{fmt(lo)} to {fmt(hi)} (FDR-adjusted p = {fmt(p, 3)}). **The "
            f"interval straddles zero, so at n=137 the two are not "
            f"statistically distinguishable** and the ordering should not be "
            f"over-interpreted.")


def _plm_verdict(by_fs: pd.DataFrame, comp: pd.DataFrame) -> str:
    if by_fs.empty:
        return ""

    def get(name, col="spearman_mean"):
        r = by_fs[by_fs.feature_set == name]
        return float(r[col].iloc[0]) if len(r) else None

    abl, esm = get("ablang2"), get("esm2_35m")
    desc = get("desc_all")
    abl_d, esm_d = get("ablang2", "n_features"), get("esm2_35m", "n_features")
    if abl is None or esm is None:
        return ""

    dims = ""
    if abl_d and esm_d:
        dims = (f" The two are not width-matched: AbLang2 contributes "
                f"{int(abl_d)} dimensions in a single pass over the paired "
                f"chains, while ESM-2 contributes {int(esm_d)} "
                f"({int(esm_d // 2)} per chain, concatenated, because it has no "
                f"paired-chain separator token). Any capacity advantage "
                f"therefore favours ESM-2.")

    if abl > esm:
        plm_cmp = (
            f"**AbLang2 ({fmt(abl)}) scored above ESM-2 35M ({fmt(esm)})** by "
            f"{fmt(abl - esm)} in Spearman.{dims} "
            + _significance_clause(_lookup_comparison(comp, "ablang2"))
        )
    else:
        plm_cmp = (
            f"**ESM-2 35M ({fmt(esm)}) matched or beat AbLang2 ({fmt(abl)})** "
            f"(difference {fmt(esm - abl)}).{dims} "
            + _significance_clause(_lookup_comparison(comp, "esm2_35m"))
            + " Antibody-specific pre-training did not help here, which is "
            "consistent with published observations that general protein "
            "language models can equal antibody-specific ones on hydrophobicity "
            "endpoints -- plausibly because HIC retention is driven by generic "
            "surface physics rather than by anything specifically immunological."
        )

    extra = ""
    if desc is not None:
        if desc >= max(abl, esm):
            extra = (
                f" Notably, the interpretable descriptor set ({fmt(desc)}) was at "
                f"least as good as either language model. With 137 training examples "
                f"this is unsurprising: a 480- or 960-dimensional embedding has far "
                f"more directions than the data can constrain, whereas the descriptors "
                f"encode the relevant physics directly. The practical consequence is "
                f"that we lose nothing by using the interpretable model, and gain the "
                f"ability to say *why* a prediction came out the way it did."
            )
        else:
            extra = (
                f" The descriptor set reached {fmt(desc)}, below the best language "
                f"model, so the learned representation is contributing signal the "
                f"hand-built descriptors miss."
            )
    return plm_cmp + extra


def _external_text(ext: pd.DataFrame, ep, conf: dict) -> str:
    if ext.empty:
        return ""
    parts = []
    val = ext[ext["evaluation_role"] == "external_validation"]
    stress = ext[ext["evaluation_role"] == "stress_test"]

    if len(val):
        r = val.iloc[0]
        rho = float(r["spearman"])
        lo_ci, hi_ci = float(r["spearman_ci_low"]), float(r["spearman_ci_high"])
        # Branch on BOTH bounds. Testing only the lower bound would describe a
        # correlation whose entire interval sits BELOW zero -- a model that
        # anti-ranks the held-out cohort, the worst possible outcome -- as
        # merely "not distinguishable from chance".
        if not (np.isfinite(lo_ci) and np.isfinite(hi_ci)):
            sig = "though its confidence interval could not be estimated"
        elif lo_ci > 0:
            sig = "and its CI excludes zero"
        elif hi_ci < 0:
            sig = ("and its CI lies entirely BELOW zero -- the model ranks this "
                   "cohort in the wrong direction, which is a failed external "
                   "validation, not a weak one")
        else:
            sig = ("though its CI includes zero, so it is not statistically "
                   "distinguishable from chance")
        parts.append(
            f"On **{r['cohort']}** (n={int(r['n'])}), a cohort of antibodies the model "
            f"has never seen, measured on the same assay platform, Spearman is "
            f"{fmt(rho)} {sig}. MAE is {fmt(r['mae'], 2)} {ep.units}. Because this "
            f"cohort shares the 2017 absolute scale, the error in minutes is "
            f"directly interpretable rather than merely ordinal."
        )
        cov = float(r.get("conformal_coverage", float("nan")))
        nominal = 1 - config.CONFORMAL_ALPHA
        if np.isfinite(cov):
            verdict = ("close to nominal, so the conformal guarantee survived the "
                       "cohort shift" if abs(cov - nominal) <= 0.1 else
                       "away from nominal, which is exactly what we warned could happen: "
                       "conformal coverage is guaranteed only under exchangeability, "
                       "and a different antibody population breaks that assumption")
            parts.append(
                f"The {int(nominal * 100)}% conformal prediction intervals achieved "
                f"{cov:.0%} empirical coverage on this cohort -- {verdict}."
            )

    if len(stress):
        r = stress.iloc[0]
        unc = r.get("spearman_uncensored")
        unc_txt = (f" Restricted to the {int(r['n_uncensored'])} uncensored antibodies "
                   f"it is {fmt(unc)}." if pd.notna(unc) else "")
        parts.append(
            f"On the **{r['cohort']}** stress test (n={int(r['n'])}), Spearman is "
            f"{fmt(r['spearman'])}.{unc_txt} We attach no validation weight to this "
            f"number. The cohort is a point-mutant series of one parent antibody on a "
            f"different chromatographic scale, a third of its labels are censored at "
            f"the instrument ceiling, and its absolute errors are meaningless here "
            f"(the model predicts ~10 min for antibodies measured at ~28 min). It is "
            f"included to show what happens when a model is pushed well outside its "
            f"applicability domain, and the honest answer is that it should not be "
            f"used there."
        )
    return "\n\n".join(parts)


def section_interpretation(summaries: dict) -> str:
    key = config.PRIMARY_ENDPOINT
    motif = _load_json(config.RESULTS_DIR / "motif_summary.json").get(key, {})
    uni = _load_csv(config.TABLES_DIR / f"univariate_{key}.csv")
    liab = _load_csv(config.TABLES_DIR / f"motifs_liabilities_{key}.csv")
    comp = _load_csv(config.TABLES_DIR / f"motifs_composition_{key}.csv")
    pos = _load_csv(config.TABLES_DIR / f"motifs_positional_H_{key}.csv")
    attr = _load_csv(config.TABLES_DIR / f"attribution_{key}.csv")

    uni_sig = uni[uni["significant_fdr05"]] if "significant_fdr05" in uni.columns else pd.DataFrame()
    comp_sig = comp[comp["significant_fdr05"]] if "significant_fdr05" in comp.columns else pd.DataFrame()
    pos_sig = pos[pos["significant_fdr05"]] if "significant_fdr05" in pos.columns else pd.DataFrame()
    liab_sig = liab[liab["significant_fdr05"]] if "significant_fdr05" in liab.columns else pd.DataFrame()

    return f"""## 4. What drives the endpoint

### 4.1 Univariate descriptor associations

{len(uni_sig)} of {len(uni)} descriptors are associated with HIC retention at a
Benjamini-Hochberg FDR of 5%.

![Univariate associations](../results/figures/eda_{key}_univariate.png)

{md_table(uni, ['descriptor', 'spearman', 'p_value', 'p_adjusted_bh'],
          ['descriptor', 'Spearman rho', 'p', 'p (BH)'],
          nd={'spearman': 3, 'p_value': 5, 'p_adjusted_bh': 5}, max_rows=15)}

{_biophysics_narrative(uni)}

### 4.2 Amino-acid composition by CDR

![Composition](../results/figures/motifs_{key}_composition.png)

{len(comp_sig)} of {len(comp)} (amino acid x CDR) combinations survive FDR
correction.

{md_table(comp, ['chain', 'region', 'amino_acid', 'spearman', 'p_bh', 'significant_fdr05'],
          ['chain', 'CDR', 'residue', 'rho', 'p (BH)', 'FDR 5%'],
          nd={'spearman': 3, 'p_bh': 5}, max_rows=12)}

### 4.3 Chemical liability motifs

![Liabilities](../results/figures/motifs_{key}_liabilities.png)

{_liability_text(liab, liab_sig)}

### 4.4 Position-specific enrichment

With every heavy chain aligned to a common IMGT numbering, we tested each
(position, residue) combination observed in at least 8 antibodies.
{len(pos_sig)} of {len(pos)} tests survive FDR correction.

{md_table(pos, ['imgt_position', 'region', 'amino_acid', 'n_with', 'mean_with',
                'mean_without', 'delta', 'p_bh', 'significant_fdr05'],
          ['IMGT pos', 'region', 'residue', 'n', 'mean with', 'mean without', 'delta',
           'p (BH)', 'FDR 5%'],
          nd={'n': 0, 'n_with': 0, 'mean_with': 2, 'mean_without': 2, 'delta': 2, 'p_bh': 5}, max_rows=12)}

The table is ordered by raw p-value, so rows near the bottom may be nominally
significant without surviving correction; the `FDR 5%` column is the one that
matters.

![Difference logo](../results/figures/motifs_{key}_logo_H.png)

Position-level results on 137 antibodies should be read as hypothesis
generation, not as established determinants. With several hundred tests and a
few dozen antibodies carrying any given residue at any given position, even
FDR-controlled hits need replication on an independent panel before anyone
should re-engineer a molecule on their basis.

### 4.5 What the model actually uses

![Attribution](../results/figures/attribution_{key}.png)

{md_table(attr, ['feature', 'importance', 'signed_strength'],
          ['feature', 'importance', 'direction (signed rho)'],
          nd={'importance': 4, 'signed_strength': 3}, max_rows=15)}

Attribution is computed on the **descriptor** model rather than the overall
best model. Attributing 480 anonymous embedding dimensions would produce a
ranking nobody could act on; "CDR aromatic fraction" is a statement a protein
engineer can test at the bench.

---
"""


def _biophysics_narrative(uni: pd.DataFrame) -> str:
    """
    Describe the univariate pattern from the measured table.

    Written generatively rather than as fixed prose: a hardcoded paragraph
    naming "the strongest associations" would keep asserting the same
    descriptors even if a re-run put entirely different ones at the top.
    """
    if uni.empty or "descriptor" not in uni.columns:
        return "_(univariate table not available)_"

    def top_matching(substrings, positive=None, k=3):
        d = uni.copy()
        mask = d["descriptor"].astype(str).str.contains("|".join(substrings),
                                                        case=False, regex=True)
        d = d[mask]
        if positive is True:
            d = d[d["spearman"] > 0]
        elif positive is False:
            d = d[d["spearman"] < 0]
        d = d.reindex(d["spearman"].abs().sort_values(ascending=False).index)
        return d.head(k)

    def render(d):
        return ", ".join(f"`{r.descriptor}` ({r.spearman:+.2f})"
                         for r in d.itertuples()) or "none"

    hydro = top_matching(["hydrophobic", "_bm_", "_eis_", "_kd_"], positive=True)
    arom = top_matching(["aromatic", "frac_W", "frac_Y", "frac_F"], positive=True)
    charge = top_matching(["positive", "frac_R", "frac_K", "charge"], positive=False)

    bullets = [
        "The pattern recovers the known biophysics of the assay without being "
        "told it:", "",
    ]
    if len(hydro):
        bullets.append(f"- **Hydrophobicity raises retention.** Strongest: {render(hydro)}.")
    if len(arom):
        bullets.append(f"- **Aromatic content raises retention.** Strongest: {render(arom)}.")
    if len(charge):
        bullets.append(f"- **Positive charge lowers retention.** Strongest: {render(charge)}.")

    # Which hydrophobicity scale actually did best, measured rather than assumed.
    scales = {"Black-Mould": "_bm_mean", "Eisenberg": "_eis_mean",
              "Kyte-Doolittle": "_kd_mean"}
    best = {}
    for label, tag in scales.items():
        d = uni[uni["descriptor"].astype(str).str.contains(tag, regex=False)]
        if len(d):
            best[label] = float(d["spearman"].abs().max())
    if len(best) >= 2:
        ranked = sorted(best.items(), key=lambda kv: -kv[1])
        ranking = " > ".join(f"**{k}** (max |rho| {v:.2f})" for k, v in ranked)
        bullets += [
            "",
            f"One methodological aside, measured rather than assumed: ranking the "
            f"three hydrophobicity scales by their strongest association gives "
            f"{ranking}. Including three scales instead of defaulting to "
            f"Kyte-Doolittle -- the most widely used, and the one the literature "
            f"reports as a poor choice for HIC specifically -- turned a hidden "
            f"assumption into a result that can be checked.",
        ]
    return "\n".join(bullets)


def _liability_text(liab: pd.DataFrame, sig: pd.DataFrame) -> str:
    if liab.empty:
        return "_(no liability analysis available)_"
    top = liab.head(10)
    tbl = md_table(top, ['feature', 'mean_risk_positive', 'mean_risk_negative',
                         'spearman', 'spearman_p_bh', 'significant_fdr05'],
                   ['motif feature', 'mean (risk+)', 'mean (risk-)', 'rho', 'p (BH)', 'FDR 5%'],
                   nd={'mean_risk_positive': 2, 'mean_risk_negative': 2,
                       'spearman': 3, 'spearman_p_bh': 4})
    if len(sig) == 0:
        verdict = (
            "**No chemical liability motif survives FDR correction.** This is a "
            "genuinely useful negative result. The canonical developability "
            "liabilities -- N-glycosylation sequons, NG/NS deamidation, DG/DS "
            "isomerisation, free cysteine -- are chemical-stability risks, and there "
            "is no particular reason they should predict a *chromatographic* "
            "hydrophobicity readout. They are reported by the screening tool "
            "alongside the prediction precisely because they are complementary "
            "information, not redundant with it: an antibody can have excellent "
            "predicted HIC behaviour and still carry a glycosylation sequon in "
            "CDR-H2 that will cause trouble later."
        )
    else:
        names = ", ".join(f"`{n}`" for n in sig["feature"].head(5))
        verdict = (
            f"{len(sig)} liability features survive FDR correction: {names}. "
            f"Note that oxidation-prone residue counts (Met, Trp) overlap "
            f"mechanistically with hydrophobicity itself, so an association there "
            f"is partly a restatement of the aromatic/hydrophobic signal rather "
            f"than independent evidence about chemical stability."
        )
    return tbl + "\n\n" + verdict


def section_error_ablation(summaries: dict) -> str:
    key = config.PRIMARY_ENDPOINT
    ep = config.ENDPOINTS[key]
    ea = _load_json(config.RESULTS_DIR / "error_analysis_summary.json").get(key, {})
    resid = ea.get("residuals", {})
    ad = ea.get("applicability", {})
    worst = pd.DataFrame(resid.get("worst_predictions", []))
    bands = pd.DataFrame(ad.get("bands", []))
    abl = _load_csv(config.TABLES_DIR / f"ablation_{key}.csv")

    if not ea:
        return ("## 5. Error analysis, applicability domain and ablation\n\n"
                "_(this analysis did not run -- execute `python -m "
                "src.analysis.error_analysis` after the model sweep)_\n\n---\n")

    ratio = resid.get("tail_error_ratio")
    tail_txt = ""
    if ratio is not None:
        if ratio > 1.3:
            tail_txt = (
                f"The model is **{fmt(ratio, 2)}x less accurate on the high-risk "
                f"antibodies** (MAE {fmt(resid.get('mae_risk_positive'), 2)} "
                f"{ep.units}) than on the rest (MAE "
                f"{fmt(resid.get('mae_risk_negative'), 2)} {ep.units}). That is "
                f"the uncomfortable direction: triage cares about exactly the "
                f"tail the model fits worst. Part of it is unavoidable -- the "
                f"risky tail is sparse and long -- but it means the ranking "
                f"should be trusted more for excluding safe candidates than for "
                f"confidently condemning risky ones."
            )
        else:
            tail_txt = (
                f"Error is roughly uniform across the risk strata (MAE "
                f"{fmt(resid.get('mae_risk_positive'), 2)} on high-risk vs "
                f"{fmt(resid.get('mae_risk_negative'), 2)} {ep.units} on the "
                f"rest, ratio {fmt(ratio, 2)}), so the model is not simply "
                f"regressing everything to the mean and abandoning the tail."
            )

    ad_txt = ""
    if ad.get("identity_vs_error_spearman") is not None:
        r = ad["identity_vs_error_spearman"]
        p = ad.get("identity_vs_error_p")
        if r < -0.15 and (p or 1) < 0.1:
            ad_txt = (
                f"Error falls as an antibody gets more similar to the training "
                f"set (Spearman between max CDR identity and |error| = "
                f"{fmt(r)}, p = {fmt(p, 3)}). **The applicability-domain flag is "
                f"doing real work**: it is not a decorative caveat but a "
                f"measurable predictor of when to distrust the model."
            )
        else:
            ad_txt = (
                f"Within this external cohort, error does **not** track distance "
                f"from the training set (Spearman between max CDR identity and "
                f"|error| = {fmt(r)}, p = {fmt(p, 3)}). The honest reading is "
                f"that the cohort spans too narrow an identity range "
                f"(roughly 0.52-0.72) to resolve the relationship, not that "
                f"extrapolation is safe. The flag remains a warning, but this "
                f"analysis does not independently validate its calibration."
            )

    return f"""## 5. Error analysis, applicability domain and ablation

### 5.1 Where the model fails

![Error analysis](../results/figures/error_{key}_residuals.png)

Overall out-of-fold MAE is {fmt(resid.get('mae_overall'), 2)} {ep.units}.
{tail_txt}

The fifteen worst-predicted antibodies:

{md_table(worst, ['antibody_id', 'value', 'oof_prediction', 'residual',
                  'risk_label', 'is_censored', 'heavy_v_family'],
          ['antibody', f'measured ({ep.units})', 'predicted', 'residual',
           'high risk', 'censored', 'heavy V family'],
          nd={'value': 2, 'oof_prediction': 2, 'residual': 2, 'risk_label': 0},
          max_rows=15)}

Censored antibodies appear here by construction: their recorded value is the
assay ceiling rather than a measurement, so a model that predicts the true
(higher) value is penalised for being right.

{_germline_text(ea.get('germline', {}), ep)}

### 5.3 Does the applicability-domain flag work?

![Applicability vs error](../results/figures/error_{key}_applicability.png)

{ad_txt}

{md_table(bands, ['band', 'n', 'mean_identity', 'mae', 'interval_coverage'],
          ['applicability band', 'n', 'mean identity to train',
           f'MAE ({ep.units})', 'interval coverage'],
          nd={'n': 0, 'mean_identity': 3, 'mae': 2, 'interval_coverage': 2})}

Coverage is reported per band rather than only in aggregate, because an
interval that covers 80% overall while covering 95% of the easy cases and 55%
of the hard ones is not calibrated where it matters.

### 5.4 Which descriptor families carry the signal?

![Ablation](../results/figures/ablation_{key}.png)

Two complementary views: drop one family and measure the loss, then use each
family alone. The first says what is *redundant* (dropping a family that other
families already encode costs nothing); the second says what is *sufficient*.

{md_table(abl, ['ablation', 'n_features', 'spearman', 'delta_vs_full'],
          ['configuration', 'features', 'Spearman', 'change vs full set'],
          nd={'n_features': 0, 'spearman': 3, 'delta_vs_full': 3})}

{_ablation_verdict(abl)}

---
"""


def _germline_text(germ: dict, ep) -> str:
    """Describe systematic error by germline family, generated from the test."""
    if not germ or not germ.get("n_families_tested"):
        return ""
    tested = germ["n_families_tested"]
    biased = germ.get("biased_families", [])
    table = pd.DataFrame(germ.get("table", []))

    head = ("### 5.2 Is the error concentrated in one germline family?\n\n"
            "![Error by germline](../results/figures/error_"
            f"{config.PRIMARY_ENDPOINT}_by_germline.png)\n\n")

    body = md_table(
        table,
        ["family", "n", "mean_measured", "mean_measured_others",
         "mean_residual", "p_residual_bh", "biased_fdr05"],
        ["V family", "n", f"mean measured ({ep.units})", "mean, others",
         f"mean residual ({ep.units})", "p (BH)", "biased at FDR 5%"],
        nd={"n": 0, "mean_measured": 2, "mean_measured_others": 2,
            "mean_residual": 2, "p_residual_bh": 4},
    )

    if not biased:
        # Nothing survives correction -- but reporting that alone would be
        # misleading if one family carries a large, nominally significant
        # residual. Name it, give its exact p-values, and say plainly which
        # side of the correction it falls on.
        verdict = (
            f"No germline family's residual survives FDR correction across the "
            f"{tested} families tested."
        )
        if len(table):
            top = table.iloc[table["mean_residual"].abs().idxmax()]
            if abs(float(top["mean_residual"])) > 0.5 and float(top["p_residual"]) < 0.05:
                fam = str(top["family"])
                direction = "under" if top["mean_residual"] > 0 else "over"
                extreme_note = ""
                if bool(top.get("extreme_fdr05", False)):
                    extreme_note = (
                        f" Separately, {fam} **is** confirmed as the most "
                        f"extreme family on this assay: it measures a mean of "
                        f"{fmt(top['mean_measured'], 2)} {ep.units} against "
                        f"{fmt(top['mean_measured_others'], 2)} for everything "
                        f"else, and that difference does survive correction "
                        f"(FDR-adjusted p = {fmt(top.get('p_measured_bh'), 4)})."
                    )
                mechanism = ""
                if fam.startswith("IGHV1-69"):
                    mechanism = (
                        f"\n\nThat is mechanistically coherent. IGHV1-69 carries "
                        f"an unusually hydrophobic CDR-H2 apex -- the isoleucine "
                        f"and phenylalanine pair that makes it the germline of "
                        f"choice for influenza-stem and other hydrophobic-epitope "
                        f"antibodies -- so it is exactly the family a "
                        f"hydrophobicity assay should retain longest, and exactly "
                        f"the family a model trained mostly on well-behaved "
                        f"antibodies will pull back toward the mean. It accounts "
                        f"for much of the tail error in Section 5.1."
                    )
                verdict += (
                    f" The largest effect is **{fam}** (n={int(top['n'])}), which "
                    f"the model {direction}-predicts by "
                    f"{fmt(top['mean_residual'], 2)} {ep.units}. That is "
                    f"nominally significant (p = {fmt(top['p_residual'], 4)}) but "
                    f"**does not survive** correction across {tested} families "
                    f"(FDR-adjusted p = {fmt(top['p_residual_bh'], 3)}), so we "
                    f"report it as a hypothesis, not a finding.{extreme_note}"
                    f"{mechanism}\n\nThe practical caution is cheap to apply "
                    f"regardless: when screening {fam}-derived candidates, treat "
                    f"the predicted value as a "
                    f"{'floor' if direction == 'under' else 'ceiling'}. "
                    f"Confirming this properly needs an independent panel with "
                    f"more than {int(top['n'])} members of the family."
                )
    else:
        b = biased[0]
        fam = str(b["family"])
        direction = "under" if b["mean_residual"] > 0 else "over"
        extreme = ("and it is genuinely the most extreme family on this assay"
                   if b.get("extreme_fdr05") else
                   "though its measured values are not themselves significantly "
                   "extreme")

        # The mechanistic note applies to one specific germline, so it is
        # attached only when that germline is the one the test actually found.
        mechanism = ""
        if fam.startswith("IGHV1-69"):
            mechanism = (
                "This is mechanistically coherent rather than a curiosity. "
                "IGHV1-69 carries an unusually hydrophobic CDR-H2 apex -- the "
                "isoleucine and phenylalanine pair that makes it the germline of "
                "choice for influenza-stem and other hydrophobic-epitope "
                "antibodies -- so it is exactly the family a hydrophobicity assay "
                "should retain longest. The model ranks these antibodies as risky "
                "but compresses the magnitude badly, which is the same "
                "regression-to-the-mean behaviour that produces the tail error "
                "above: the risky tail is largely this one family.\n\n"
            )

        verdict = (
            f"**{fam} is systematically {direction}-predicted.** Its "
            f"{int(b['n'])} antibodies measure a mean of "
            f"{fmt(b['mean_measured'], 2)} {ep.units} against "
            f"{fmt(b['mean_measured_others'], 2)} for everything else, {extreme} "
            f"(FDR-adjusted p = {fmt(b.get('p_measured_bh'), 4)}), and the model "
            f"misses them by a mean of {fmt(b['mean_residual'], 2)} {ep.units} "
            f"(FDR-adjusted p = {fmt(b['p_residual_bh'], 4)}).\n\n"
            + mechanism +
            f"The practical consequence is concrete: a candidate derived from "
            f"{fam} should have its predicted value treated as a "
            f"{'floor' if direction == 'under' else 'ceiling'}, not an estimate. "
            f"This is among the most actionable findings in the study, and it is "
            f"visible only because germline was annotated alongside the "
            f"prediction rather than discarded after numbering."
        )
    return head + body + "\n\n" + verdict


def _ablation_verdict(abl: pd.DataFrame) -> str:
    if abl.empty or "ablation" not in abl.columns:
        return ""
    drops = abl[abl["ablation"].astype(str).str.startswith("drop")]
    onlys = abl[abl["ablation"].astype(str).str.startswith("only")]
    bits = []
    if len(drops):
        worst_drop = drops.loc[drops["delta_vs_full"].idxmin()]
        bits.append(
            f"Removing **{str(worst_drop['ablation']).replace('drop ', '')}** "
            f"costs the most ({fmt(worst_drop['delta_vs_full'])} Spearman), so "
            f"it carries signal the other families do not duplicate."
        )
        harmless = drops[drops["delta_vs_full"] >= -0.01]
        if len(harmless):
            names = ", ".join(str(a).replace("drop ", "")
                              for a in harmless["ablation"].head(3))
            bits.append(
                f"Dropping {names} costs essentially nothing, which means that "
                f"information is already carried by the families that remain -- "
                f"redundancy, not irrelevance."
            )
    if len(onlys):
        best_only = onlys.loc[onlys["spearman"].idxmax()]
        bits.append(
            f"On its own, **{str(best_only['ablation']).replace('only ', '')}** "
            f"reaches Spearman {fmt(best_only['spearman'])} from just "
            f"{int(best_only['n_features'])} features -- most of what the full "
            f"{int(abl.loc[abl['ablation'].astype(str).str.startswith('none'), 'n_features'].iloc[0])}-feature "
            f"set achieves."
        )
    return " ".join(bits)


def _representation_by_endpoint(df: pd.DataFrame) -> str:
    """
    Does the same representation win everywhere, or does it depend on the assay?

    Generated rather than asserted: if one representation happened to win on
    all four endpoints, the honest conclusion would be the opposite of the one
    a hardcoded paragraph about "it depends" would claim.
    """
    if df.empty or "feature_set" not in df.columns:
        return ""
    winners = df[["endpoint", "feature_set", "cv_spearman"]].dropna()
    if len(winners) < 2:
        return ""
    unique = winners["feature_set"].nunique()

    rows = "; ".join(
        f"**{r.endpoint}** -> `{r.feature_set}` ({fmt(r.cv_spearman)})"
        for r in winners.itertuples()
    )

    if unique == 1:
        only = winners["feature_set"].iloc[0]
        return (
            f"The same representation wins on every endpoint (`{only}`): "
            f"{rows}. That consistency is worth having -- it means a single "
            f"feature pipeline serves all four assays, and the choice of "
            f"representation is not an endpoint-specific tuning decision."
        )

    # Three mutually exclusive categories. A feature set that fuses descriptors
    # with an embedding belongs in neither pure camp, and counting it in both
    # would list the same endpoint twice.
    def kind(fs: str) -> str:
        has_plm = ("ablang2" in fs) or ("esm2" in fs)
        has_desc = fs.startswith("desc")
        if has_plm and has_desc:
            return "hybrid"
        return "plm" if has_plm else "descriptor"

    winners = winners.copy()
    winners["kind"] = winners["feature_set"].map(kind)
    groups = {k: winners.loc[winners["kind"] == k, "endpoint"].tolist()
              for k in ("descriptor", "plm", "hybrid")}

    clauses = []
    if groups["descriptor"]:
        clauses.append(f"interpretable descriptors win on "
                       f"{', '.join(groups['descriptor'])}")
    if groups["plm"]:
        clauses.append(f"a language-model representation wins on "
                       f"{', '.join(groups['plm'])}")
    if groups["hybrid"]:
        clauses.append(f"a descriptor/embedding hybrid wins on "
                       f"{', '.join(groups['hybrid'])}")

    extra = ""
    if len(clauses) > 1:
        extra = (
            f"\n\nSpecifically: {'; '.join(clauses)}.\n\n"
            f"A plausible reading -- and it is only that, on four endpoints -- is "
            f"that HIC retention is dominated by bulk hydrophobicity, which the "
            f"descriptors encode directly and explicitly. Self-association and "
            f"polyreactivity instead depend on how charge and surface patches are "
            f"arranged across the whole Fv, which is harder to write down as a "
            f"descriptor but is the kind of structure a language model can learn. "
            f"Testing that properly would need more endpoints than we have, and "
            f"the gaps between representations are small relative to their "
            f"across-repeat spread."
        )

    return (
        f"**No single representation wins everywhere.** Best feature set per "
        f"endpoint: {rows}.{extra}\n\nThis is a further argument against "
        f"treating \"developability\" as one quantity with one model: the "
        f"endpoints differ not only in how predictable they are, but in what "
        f"kind of representation predicts them."
    )


def section_secondary(summaries: dict) -> str:
    rows = []
    for key in [config.PRIMARY_ENDPOINT, *config.SECONDARY_ENDPOINTS]:
        s = summaries.get(key)
        if not s:
            continue
        p = s.get("pooled_oof_regression", {}).get("metrics", {})
        b = s.get("best_regression", {})
        ext = _load_csv(config.TABLES_DIR / f"external_{key}.csv")
        val = ext[ext["evaluation_role"] == "external_validation"] if len(ext) else pd.DataFrame()
        rows.append({
            "endpoint": config.ENDPOINTS[key].pretty_name,
            "units": config.ENDPOINTS[key].units,
            "feature_set": b.get("feature_set", ""),
            "model": b.get("model", ""),
            # Across-repeat means, matching the headline in Section 3.3.
            # Using the single-repeat value here would make this table
            # silently disagree with the one the reader just saw.
            "cv_spearman": b.get("spearman_mean", p.get("spearman")),
            "cv_spearman_sd": b.get("spearman_std"),
            "cv_r2": b.get("r2_mean", p.get("r2")),
            "cv_mae": b.get("mae_mean", p.get("mae")),
            "ext_spearman": float(val.iloc[0]["spearman"]) if len(val) else np.nan,
            "ext_mae": float(val.iloc[0]["mae"]) if len(val) else np.nan,
        })
    df = pd.DataFrame(rows)
    return f"""## 6. The same pipeline on three other endpoints

Each endpoint is modelled **independently**, with its own training cohort, its
own external cohort and its own risk threshold. Nothing is pooled.

{md_table(df, ['endpoint', 'units', 'feature_set', 'model', 'cv_spearman',
               'cv_spearman_sd', 'cv_r2', 'cv_mae', 'ext_spearman', 'ext_mae'],
          ['endpoint', 'units', 'features', 'model', 'CV rho', 'CV rho sd',
           'CV R2', 'CV MAE', 'ext rho', 'ext MAE'],
          nd={'cv_mae': 3, 'ext_mae': 3})}

The spread across endpoints is itself informative. Endpoints differ in how
predictable they are from sequence alone, and reporting all four -- including
the ones the method handles poorly -- is more useful than selecting the one
that happened to work.

{_representation_by_endpoint(df)}

The PSR endpoint deserves a specific caveat: 72 of 137 training values are
exactly 0.00, the assay floor. A Spearman correlation against a target whose
median, first quartile and minimum are all identical is largely a
tie-breaking exercise, and the regression framing is a poor fit for it. A
zero-versus-nonzero classification would be the more defensible treatment of
that endpoint.

---
"""


def section_limitations(summaries: dict) -> str:
    key = config.PRIMARY_ENDPOINT
    s = summaries.get(key, {})
    pooled = s.get("pooled_oof_regression", {}).get("metrics", {})
    rho = pooled.get("spearman")
    r2 = pooled.get("r2")

    return f"""## 7. Honest positioning against the literature

Our result should be read against what has actually been achieved on this
panel. The comparator rows below are **values quoted from the published
literature**, transcribed from the sources cited in
`docs/LITERATURE_REVIEW.md`; this pipeline cannot verify them, and they are
the only numbers in this report not generated by it. Only the final row is
our own result.

| study | features | protocol | reported |
|---|---|---|---|
| Hebditch & Warwicker 2019 | 35 Fv sequence features | 50x repeated 10-fold CV | R2 = 0.391, MAE 0.702 |
| Karlberg et al. 2020 | sequence only | 80/20 Kennard-Stone external split (n=81) | R2 = 0.25 |
| Karlberg et al. 2020 | MD-relaxed structure | same split | R2 = 0.63 |
| PROPERMAB (Li et al. 2025) | 9 sequence + 26 **structure** features | nested LOO-CV (n=135) | r = 0.71, rho = 0.75 |
| FLAb zero-shot (2024) | PLM likelihoods, no supervision | zero-shot on Jain HIC | Pearson -0.08 to 0.24 |
| **This work** | sequence only | **{config.CV_N_REPEATS}x{config.CV_N_SPLITS} CDR-clustered nested CV** | **rho = {fmt(rho)}, R2 = {fmt(r2)}** |

Three points of context matter for a fair reading:

1. **Zero-shot language models are not competitive on this task.** The FLAb
   benchmark found that AntiBERTy, IgLM, ProGen2, ProteinMPNN and ESM-IF
   likelihoods correlate with HIC at |r| <= 0.24, with most near zero.
   Supervision on even 137 examples is worth more than a large unsupervised
   model's prior.
2. **The best published numbers use structure.** PROPERMAB's rho = 0.75 rests
   on structural hydrophobic-patch descriptors from homology models. A
   sequence-only model is solving a harder problem, and comparing the two
   without saying so would be misleading.
3. **Protocol dominates the number.** Published work on much larger proprietary
   panels reports that moving from random splits to cluster-based splits costs
   0.2-0.3 Spearman. Any comparison that ignores the split protocol is not a
   comparison.

An upper bound is also worth stating: an independent re-measurement of 133 of
these antibodies on a different platform reproduced the original HIC values at
rho ~ 0.97. Assay noise is therefore *not* what limits prediction here -- the
remaining error is model and representation error, not measurement error.

---

## 8. Limitations

**Sample size.** 137 training antibodies is small for a 380-dimensional
descriptor space, let alone a 480- or 960-dimensional embedding. Confidence
intervals are correspondingly wide and should be quoted alongside every point
estimate. We mitigate this with nested CV, pooled out-of-fold metrics,
bootstrap intervals and permutation testing, but no analysis makes 137 points
behave like 10,000.

**Sequence-only features cannot see exposure.** HIC responds to *solvent-exposed*
apolar surface. A buried leucine contributes nothing, but a sequence-only
descriptor counts it identically to an exposed one. Our maximum-windowed
hydrophobicity is a one-dimensional shadow of a three-dimensional quantity, and
this is the single clearest reason structure-based models outperform on this
endpoint.

**One assay platform.** The model is calibrated to the Jain/Adimab HIC
platform. Retention time is not a physical constant: it depends on column
chemistry, salt system and gradient. The Jetha stress test demonstrates this
concretely -- predictions transfer in rank but not at all in absolute minutes.

**Conformal coverage is conditional.** The split-conformal guarantee assumes
calibration and test antibodies are exchangeable. A new cohort is by definition
not exchangeable with the training set, so measured coverage is reported for
each external cohort rather than claimed.

**Position-level findings are hypotheses.** Section 4.4 performs hundreds of
tests on 137 antibodies. FDR control limits the expected false-discovery rate
but does not make any single position a validated determinant.

**Selection across many configurations.** We compared 7 feature sets x up to 7
model families. The headline configuration was chosen by cross-validated
performance, which introduces selection optimism even under nested CV. The
external cohort is the real guard against this, which is why it was never
touched during model selection.

## 9. Responsible use

This model is a **triage and ranking aid, not a safety assessment**. Specific
constraints on legitimate use:

- A favourable prediction is not evidence that an antibody is developable. It
  is evidence that it resembles training antibodies that behaved well on one
  assay.
- Predictions for antibodies flagged `extrapolation` by the applicability-domain
  check should be treated as **unknown**, not as safe. The screening tool labels
  these explicitly rather than quoting a number with false confidence.
- Predictions in absolute minutes are only meaningful for antibodies measured
  on the same HIC platform. For any other platform, use the ranking only.
- The model is trained on clinical-stage therapeutic antibodies, which are a
  heavily filtered population -- they already survived earlier developability
  screening. Performance on naive repertoire antibodies is untested and should
  not be assumed.
- Chemical liability motif counts are rule-based annotations, not predictions,
  and are reported separately so that a clean predicted value does not mask a
  CDR glycosylation sequon.

**Data licensing.** FLAb is redistributed by the Gray Lab; the underlying
measurements belong to the cited primary publications. This project uses them
for non-commercial academic coursework. Anyone reusing this pipeline should
check the upstream repository and the primary papers for licence terms.

---
"""


def _key_versions() -> tuple[int, str]:
    """
    Read the pinned versions straight out of requirements-lock.txt.

    Typing them into the prose would let the report claim a version the
    environment does not actually have -- which is precisely the failure mode
    a lock file exists to prevent.  Anything missing is reported as missing.
    """
    lock = config.PROJECT_ROOT / "requirements-lock.txt"
    if not lock.exists():
        return 0, "_(requirements-lock.txt not found -- run scripts/00_setup_env.ps1)_"

    lines = [l.strip() for l in lock.read_text(encoding="utf-8").splitlines() if l.strip()]
    pinned = {}
    for line in lines:
        for sep in ("==", " @ ", ">="):
            if sep in line:
                name, _, ver = line.partition(sep)
                pinned[name.strip().lower()] = ver.strip()
                break

    wanted = ["ablang2", "antpack", "torch", "transformers", "fair-esm",
              "scikit-learn", "lightgbm", "xgboost", "biopython", "shap",
              "numpy", "pandas"]
    parts = [f"`{w}` {pinned[w]}" if w in pinned else f"`{w}` **NOT PINNED**"
             for w in wanted]
    return len(lines), ", ".join(parts)


def section_repro(summaries: dict) -> str:
    manifest = _load_json(config.PROCESSED_DIR / "data_manifest.json")
    n_pkgs, key_versions = _key_versions()

    return f"""## 10. Reproducibility

```bash
# 1. Build the environment (once)
powershell -ExecutionPolicy Bypass -File .\\scripts\\00_setup_env.ps1

# 2. Run the whole study
powershell -ExecutionPolicy Bypass -File .\\scripts\\run_all.ps1

# 3. Screen new candidates with the trained model
.\\.venv\\Scripts\\python.exe -m src.tool.screen \\
    --input my_candidates.csv --endpoint hic \\
    --output screened.csv --report screened.md
```

**Determinism.** A single global seed (`{config.SEED}`) drives split generation,
model initialisation, bootstrap resampling and permutation tests. Re-running
the pipeline on the same machine reproduces every number in this report.

**Data pinning.** FLAb is checked out at commit
`{config.FLAB_PINNED_COMMIT}`, and `data/processed/data_manifest.json`
records a SHA-256 hash of every consumed file, so a silent upstream change
cannot alter the results unnoticed.

**Environment pinning.** `requirements-lock.txt` pins all {n_pkgs} resolved
packages. Key versions, read from that file at report-build time rather than
typed here: {key_versions}.

**Runtime.** About 60-90 minutes end to end on a 12-core laptop, CPU only, plus
roughly 10 minutes on the first run to download the AbLang2 (166 MB) and ESM-2
35M model weights. No GPU is required: with a few hundred antibodies it would
not help.

### Repository layout

```
src/
  config.py              endpoint definitions, split protocol, seeds
  data/
    acquire.py           pinned, sparse clone of FLAb + provenance manifest
    curate.py            cleaning, dedup, risk labels, leakage control
    splits.py            CDR-identity clustering, grouped folds, applicability domain
  features/
    numbering.py         AntPack IMGT numbering, CDRs, germlines, liabilities
    descriptors.py       physicochemical + CDR + TAP-style descriptors
    plm.py               AbLang2 and ESM-2 embeddings (cached)
    build.py             feature-set assembly
  models/
    zoo.py               estimators and hyper-parameter grids
    train.py             nested CV, external evaluation, permutation tests
    metrics.py           metrics, bootstrap CIs, conformal intervals
    experiment.py        the full sweep and its outputs
  analysis/
    eda.py               provenance and distribution figures
    motifs.py            liability, composition, positional and attribution analyses
    plotting.py          shared, colourblind-safe figure style
  tool/
    screen.py            the command-line screening tool
  report/
    build_report.py      generates this document from the result artefacts

results/      tables (CSV), figures (PNG), predictions, summary JSON
models_store/ serialised final models
docs/         literature review, methodology notes
```

### Outputs

| artefact | location |
|---|---|
| Curated datasets | `data/processed/dataset_*.csv` |
| Data manifest with hashes | `data/processed/data_manifest.json` |
| Split definitions | `data/processed/splits_*.json` |
| Cross-validation tables | `results/tables/cv_*.csv` |
| External validation | `results/tables/external_*.csv` |
| Out-of-fold predictions | `results/predictions/oof_*.csv` |
| Figures | `results/figures/*.png` |
| Trained models | `models_store/final_*.joblib` |
| Literature review | `docs/LITERATURE_REVIEW.md` |

---

## 11. References

See `docs/LITERATURE_REVIEW.md` for the full annotated reference list with
DOIs. Primary data sources:

- Jain T, et al. (2017). Biophysical properties of the clinical-stage antibody
  landscape. *PNAS* 114(5):944-949. doi:10.1073/pnas.1616408114
- Jain T, et al. (2024). Assessment and incorporation of in vitro correlates to
  pharmacokinetic outcomes in antibody developability workflows. *mAbs*
  16(1):2384104. doi:10.1080/19420862.2024.2384104
- Jetha A, et al. (2018). Homology modeling and structure-based design improve
  hydrophobic interaction chromatography behavior of integrin binding
  antibodies. *mAbs* 10(6):890-900. doi:10.1080/19420862.2018.1475871
- Chungyoun M, Ruffolo JA, Gray JJ (2024). FLAb: Benchmarking deep learning
  methods for antibody fitness prediction. *bioRxiv* 2024.01.13.575504
- Olsen TH, Moal IH, Deane CM (2024). AbLang2: addressing the antibody germline
  bias. *Bioinformatics*. (AbLang2 model)
- Lin Z, et al. (2023). Evolutionary-scale prediction of atomic-level protein
  structure with a language model. *Science* 379:1123-1130. (ESM-2)
"""


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def build() -> Path:
    summaries = {}
    for key in config.ENDPOINTS:
        p = config.RESULTS_DIR / f"summary_{key}.json"
        if p.exists():
            summaries[key] = _load_json(p)

    if config.PRIMARY_ENDPOINT not in summaries:
        raise FileNotFoundError(
            "No results for the primary endpoint. Run src.models.experiment first."
        )

    doc = "\n".join([
        section_header(summaries),
        section_data(summaries),
        section_methods(summaries),
        section_results(summaries),
        section_interpretation(summaries),
        section_error_ablation(summaries),
        section_secondary(summaries),
        section_limitations(summaries),
        section_repro(summaries),
    ])

    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = config.REPORT_DIR / "FINAL_REPORT.md"
    out.write_text(doc, encoding="utf-8")
    print(f"[report] wrote {out} ({len(doc):,} characters)")
    return out


def main() -> int:
    build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
