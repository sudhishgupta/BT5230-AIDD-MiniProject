"""
Curate FLAb source CSVs into a versioned, analysis-ready dataset.

What this module guarantees about its output:

  * One row per unique antibody per endpoint per cohort.
  * Sequences are uppercase, whitespace-free and contain only the twenty
    standard amino acids.  Anything else is dropped and logged.
  * The measurement is read from the *named raw assay column*, never from
    FLAb's derived ``fitness`` column.  ``fitness`` is a convenience column
    whose sign convention varies between files; reading the raw column
    removes a whole class of silent sign errors.
  * Exact duplicate (heavy, light) pairs within a cohort are collapsed to a
    single row carrying the median measurement, with the replicate count and
    spread recorded.
  * Antibodies that appear in BOTH a training cohort and an external cohort
    are flagged, and removed from the external cohort, so that "external"
    really means "never seen in training".

Run:
    python -m src.data.curate
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.config import Cohort, Endpoint

VALID_AA = set(config.STANDARD_AA)


# --------------------------------------------------------------------------
# Sequence handling
# --------------------------------------------------------------------------

def canonicalise_sequence(seq: object) -> str | None:
    """Return an uppercase, gap-free protein sequence, or None if unusable."""
    if seq is None or (isinstance(seq, float) and np.isnan(seq)):
        return None
    s = str(seq).strip().upper()
    # Strip anything that is pure formatting: whitespace, gaps, stop codons.
    for ch in (" ", "\t", "\n", "\r", "-", ".", "*"):
        s = s.replace(ch, "")
    if not s:
        return None
    if not set(s) <= VALID_AA:
        return None
    return s


def antibody_key(heavy: str, light: str | None) -> str:
    """Stable identity for an antibody: its concatenated variable domains."""
    return f"{heavy}|{light or ''}"


def make_antibody_id(cohort_key: str, index: int) -> str:
    return f"{cohort_key}_{index:04d}"


# --------------------------------------------------------------------------
# Loading one cohort
# --------------------------------------------------------------------------

def load_cohort(cohort: Cohort, endpoint: Endpoint) -> tuple[pd.DataFrame, dict]:
    """Load and clean one source CSV. Returns (dataframe, curation_log)."""
    path = config.FLAB_DIR / cohort.relpath
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])

    log: dict = {
        "cohort": cohort.key,
        "endpoint": endpoint.key,
        "source_file": cohort.relpath,
        "rows_in_source": int(len(raw)),
        "source_columns": list(raw.columns),
    }

    if cohort.value_column not in raw.columns:
        raise KeyError(
            f"Column {cohort.value_column!r} not found in {cohort.relpath}. "
            f"Available: {list(raw.columns)}"
        )

    df = pd.DataFrame({
        "heavy_chain_raw": raw["heavy"],
        "light_chain_raw": raw["light"] if "light" in raw.columns else None,
        "value_raw": raw[cohort.value_column],
    })

    # -- sequences ---------------------------------------------------------
    df["heavy_chain"] = df["heavy_chain_raw"].map(canonicalise_sequence)
    if cohort.paired:
        df["light_chain"] = df["light_chain_raw"].map(canonicalise_sequence)
    else:
        df["light_chain"] = None

    bad_heavy = df["heavy_chain"].isna()
    bad_light = df["light_chain"].isna() if cohort.paired else pd.Series(False, index=df.index)
    log["dropped_bad_heavy"] = int(bad_heavy.sum())
    log["dropped_bad_light"] = int(bad_light.sum())
    log["examples_bad_sequences"] = (
        df.loc[bad_heavy | bad_light, ["heavy_chain_raw", "light_chain_raw"]]
        .head(5).to_dict("records")
    )
    df = df.loc[~(bad_heavy | bad_light)].copy()

    # -- measurement -------------------------------------------------------
    df["value"] = pd.to_numeric(df["value_raw"], errors="coerce")
    bad_value = df["value"].isna()
    log["dropped_non_numeric_value"] = int(bad_value.sum())
    log["examples_bad_values"] = (
        df.loc[bad_value, "value_raw"].head(10).tolist()
    )
    df = df.loc[~bad_value].copy()

    non_finite = ~np.isfinite(df["value"].to_numpy(dtype=float))
    log["dropped_non_finite_value"] = int(non_finite.sum())
    df = df.loc[~non_finite].copy()

    # -- collapse exact duplicate antibodies -------------------------------
    df["ab_key"] = [
        antibody_key(h, l) for h, l in zip(df["heavy_chain"], df["light_chain"])
    ]
    n_before = len(df)
    grouped = (
        df.groupby("ab_key", sort=False)
          .agg(
              heavy_chain=("heavy_chain", "first"),
              light_chain=("light_chain", "first"),
              value=("value", "median"),
              n_replicates=("value", "size"),
              value_min=("value", "min"),
              value_max=("value", "max"),
              value_std=("value", "std"),
          )
          .reset_index()
    )
    log["rows_before_dedup"] = int(n_before)
    log["rows_after_dedup"] = int(len(grouped))
    log["duplicate_pairs_collapsed"] = int(n_before - len(grouped))
    dup = grouped.loc[grouped["n_replicates"] > 1]
    log["max_replicate_spread"] = (
        float((dup["value_max"] - dup["value_min"]).max()) if len(dup) else 0.0
    )

    grouped = grouped.sort_values("ab_key", kind="stable").reset_index(drop=True)
    grouped.insert(0, "antibody_id",
                   [make_antibody_id(cohort.key, i) for i in range(len(grouped))])

    grouped["endpoint"] = endpoint.key
    grouped["property"] = endpoint.pretty_name
    grouped["assay_column"] = cohort.value_column
    grouped["units"] = cohort.units
    grouped["cohort"] = cohort.key
    grouped["cohort_name"] = cohort.cohort_name
    grouped["publication"] = cohort.publication
    grouped["year"] = cohort.year
    grouped["role"] = "train" if cohort.key == endpoint.train.key else "external"
    grouped["evaluation_role"] = (
        "train" if cohort.key == endpoint.train.key else cohort.evaluation_role
    )
    grouped["heavy_len"] = grouped["heavy_chain"].str.len()
    grouped["light_len"] = grouped["light_chain"].fillna("").str.len()

    # -- censoring -------------------------------------------------------
    # A value pinned at the assay ceiling is an inequality, not a measurement.
    # We keep those rows (dropping them would bias the distribution) but mark
    # them so every downstream metric can be reported both with and without.
    if cohort.censoring_value is not None:
        if cohort.censoring_direction == "right":
            grouped["is_censored"] = grouped["value"] >= cohort.censoring_value
        else:
            grouped["is_censored"] = grouped["value"] <= cohort.censoring_value
    else:
        grouped["is_censored"] = False
    log["n_censored"] = int(grouped["is_censored"].sum())
    log["censoring_value"] = cohort.censoring_value
    log["frac_censored"] = float(grouped["is_censored"].mean())

    # -- within-cohort near-duplicate diagnostics --------------------------
    log["duplicate_heavy_chains"] = int(
        len(grouped) - grouped["heavy_chain"].nunique()
    )
    log["duplicate_light_chains"] = int(
        len(grouped) - grouped["light_chain"].fillna("").nunique()
    )
    v = grouped["value"].to_numpy(dtype=float)
    log["value_summary"] = {
        "n": int(len(v)),
        "min": float(np.min(v)), "q1": float(np.percentile(v, 25)),
        "median": float(np.median(v)), "q3": float(np.percentile(v, 75)),
        "max": float(np.max(v)), "mean": float(np.mean(v)),
        "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
    }
    log["length_summary"] = {
        "heavy_min": int(grouped["heavy_len"].min()),
        "heavy_median": float(grouped["heavy_len"].median()),
        "heavy_max": int(grouped["heavy_len"].max()),
        "light_min": int(grouped["light_len"].min()),
        "light_median": float(grouped["light_len"].median()),
        "light_max": int(grouped["light_len"].max()),
    }

    cols = [
        "antibody_id", "heavy_chain", "light_chain", "endpoint", "property",
        "value", "units", "assay_column", "cohort", "cohort_name",
        "publication", "year", "role", "evaluation_role", "is_censored",
        "heavy_len", "light_len", "n_replicates", "value_std", "ab_key",
    ]
    return grouped[cols], log


# --------------------------------------------------------------------------
# Risk labels
# --------------------------------------------------------------------------

def add_risk_labels(df: pd.DataFrame, endpoint: Endpoint,
                    train_values: np.ndarray) -> pd.DataFrame:
    """
    Binarise the continuous measurement into a developability risk label.

    The threshold is always derived from the TRAINING cohort distribution and
    then applied unchanged to external cohorts.  Deriving a separate threshold
    per cohort would make the external evaluation meaningless, because a
    cohort-specific quantile guarantees the same positive rate everywhere.

    Convention follows Jain et al. 2017: an antibody is flagged when its value
    falls in the unfavourable tail of the reference distribution.
    """
    df = df.copy()
    higher_is_worse = endpoint.favourable_direction == "lower"

    # Preferred: a threshold published for this assay.  Jain et al. 2017 define
    # the HIC warning flag at > 11.7 min from the worst decile of APPROVED
    # antibodies, which is a clinically anchored cut rather than a property of
    # whatever sample we happen to hold.
    if endpoint.risk_threshold_absolute is not None:
        thr = float(endpoint.risk_threshold_absolute)
        df["risk_label_published"] = (
            (df["value"] >= thr) if higher_is_worse else (df["value"] <= thr)
        ).astype(int)
        df["risk_label_published_threshold"] = thr

    for q in endpoint.risk_quantile_sensitivity:
        if higher_is_worse:
            thr = float(np.quantile(train_values, q))
            flag = (df["value"] >= thr).astype(int)
        else:
            thr = float(np.quantile(train_values, 1.0 - q))
            flag = (df["value"] <= thr).astype(int)
        col = f"risk_label_q{int(round(q * 100))}"
        df[col] = flag
        df[f"{col}_threshold"] = thr

    if endpoint.risk_threshold_absolute is not None:
        df["risk_label"] = df["risk_label_published"]
        df["risk_threshold"] = df["risk_label_published_threshold"]
        df["risk_label_source"] = "published"
    else:
        primary_col = f"risk_label_q{int(round(endpoint.risk_quantile * 100))}"
        df["risk_label"] = df[primary_col]
        df["risk_threshold"] = df[f"{primary_col}_threshold"]
        df["risk_label_source"] = f"train_quantile_{endpoint.risk_quantile}"
    return df


# --------------------------------------------------------------------------
# Cross-cohort leakage control
# --------------------------------------------------------------------------

def resolve_cross_cohort_overlap(train: pd.DataFrame,
                                 externals: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], dict]:
    """
    Remove from each external cohort any antibody that also occurs in training.

    We check three levels of overlap and report all of them, because a shared
    light chain across cohorts is a weaker but real form of leakage that the
    reader should know about even though we do not remove it.
    """
    train_pairs = set(train["ab_key"])
    train_heavy = set(train["heavy_chain"])
    train_light = set(train["light_chain"].dropna())

    report: dict = {"train_n": int(len(train)), "externals": {}}
    cleaned: dict[str, pd.DataFrame] = {}

    for key, ext in externals.items():
        exact = ext["ab_key"].isin(train_pairs)
        heavy_only = ext["heavy_chain"].isin(train_heavy) & ~exact
        light_only = ext["light_chain"].isin(train_light) & ~exact

        report["externals"][key] = {
            "n_before": int(len(ext)),
            "exact_pair_overlap": int(exact.sum()),
            "heavy_chain_overlap_not_exact": int(heavy_only.sum()),
            "light_chain_overlap_not_exact": int(light_only.sum()),
            "n_after_removing_exact": int(len(ext) - exact.sum()),
            "removed_ids": ext.loc[exact, "antibody_id"].tolist(),
        }
        out = ext.loc[~exact].copy()
        out["shares_heavy_with_train"] = out["heavy_chain"].isin(train_heavy)
        out["shares_light_with_train"] = out["light_chain"].isin(train_light)
        cleaned[key] = out

    return cleaned, report


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def curate_endpoint(endpoint: Endpoint) -> tuple[pd.DataFrame, dict]:
    train_df, train_log = load_cohort(endpoint.train, endpoint)
    logs = {"train": train_log, "externals": {}}

    ext_dfs: dict[str, pd.DataFrame] = {}
    for cohort in endpoint.externals:
        ext_df, ext_log = load_cohort(cohort, endpoint)
        ext_dfs[cohort.key] = ext_df
        logs["externals"][cohort.key] = ext_log

    ext_clean, overlap_report = resolve_cross_cohort_overlap(train_df, ext_dfs)
    logs["cross_cohort_overlap"] = overlap_report

    train_values = train_df["value"].to_numpy(dtype=float)
    train_df = add_risk_labels(train_df, endpoint, train_values)
    for key in ext_clean:
        ext_clean[key] = add_risk_labels(ext_clean[key], endpoint, train_values)

    combined = pd.concat([train_df, *ext_clean.values()], ignore_index=True)
    combined["shares_heavy_with_train"] = combined.get(
        "shares_heavy_with_train", pd.Series(False, index=combined.index)
    ).fillna(False)
    combined["shares_light_with_train"] = combined.get(
        "shares_light_with_train", pd.Series(False, index=combined.index)
    ).fillna(False)

    logs["risk_label_summary"] = {
        cohort: {
            "n": int(len(g)),
            "n_positive": int(g["risk_label"].sum()),
            "prevalence": float(g["risk_label"].mean()),
            "threshold": float(g["risk_threshold"].iloc[0]),
        }
        for cohort, g in combined.groupby("cohort")
    }

    return combined, logs


def build_multi_endpoint_table() -> pd.DataFrame:
    """
    Join the four Jain-2017 endpoints on antibody identity.

    Used only for exploratory analysis (how correlated are the assays?), never
    for training: mixing endpoints into one target is exactly the error the
    project brief warns against.
    """
    keys = [config.PRIMARY_ENDPOINT, *config.SECONDARY_ENDPOINTS]

    first_ep = config.ENDPOINTS[keys[0]]
    first_df, _ = load_cohort(first_ep.train, first_ep)
    out = first_df[["ab_key", "heavy_chain", "light_chain", "value"]].rename(
        columns={"value": keys[0]}
    )

    for key in keys[1:]:
        ep = config.ENDPOINTS[key]
        df, _ = load_cohort(ep.train, ep)
        out = out.merge(
            df[["ab_key", "value"]].rename(columns={"value": key}),
            on="ab_key", how="outer",
        )

    return out


def main() -> int:
    config.ensure_dirs()
    all_logs = {}

    for key, ep in config.ENDPOINTS.items():
        print(f"\n[curate] === endpoint: {key} ({ep.pretty_name}) ===")
        combined, logs = curate_endpoint(ep)
        out = config.PROCESSED_DIR / f"dataset_{key}.csv"
        combined.to_csv(out, index=False)
        all_logs[key] = logs

        for cohort, g in combined.groupby("cohort"):
            role = g["role"].iloc[0]
            print(f"  {cohort:<18} role={role:<8} n={len(g):<4} "
                  f"median={g['value'].median():.3f} "
                  f"risk+={int(g['risk_label'].sum())}")
        print(f"  -> {out}")

    multi = build_multi_endpoint_table()
    multi_out = config.PROCESSED_DIR / "jain2017_multi_endpoint.csv"
    multi.to_csv(multi_out, index=False)
    print(f"\n[curate] Multi-endpoint table ({len(multi)} antibodies) -> {multi_out}")

    log_out = config.PROCESSED_DIR / "curation_report.json"
    log_out.write_text(json.dumps(all_logs, indent=2, default=str), encoding="utf-8")
    print(f"[curate] Curation report -> {log_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
