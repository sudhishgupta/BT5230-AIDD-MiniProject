"""
abdev-screen -- a sequence-based antibody developability screening tool.

Takes a CSV of paired antibody variable domains and returns, for each one:
  * a predicted value for the endpoint, in the assay's own units
  * a conformal prediction interval with a stated nominal coverage
  * a binary risk call against the published assay threshold
  * an applicability-domain flag: how similar is this antibody to anything the
    model was trained on, and should the prediction be trusted at all
  * a chemical liability report (glycosylation sequons, deamidation,
    isomerisation, oxidation sites), annotated by whether the hit sits in a CDR

Usage
-----
    python -m src.tool.screen --input candidates.csv --endpoint hic \
        --output screened.csv [--report screened.md]

The input CSV needs columns ``heavy_chain`` and ``light_chain``; an
``antibody_id`` column is used if present and generated otherwise.

Honest-use note
---------------
The model is trained on 137 clinical-stage antibodies measured on one HIC
platform.  It ranks candidates; it does not replace the assay.  Predictions
for antibodies far outside the training domain are reported with an
``extrapolation`` flag and should be read as "unknown", not as "safe".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import config
from src.features import descriptors as D
from src.features import numbering as nb
from src.features import plm
from src.data.curate import canonicalise_sequence

# Applicability-domain bands, expressed as max CDR identity to any training
# antibody.  The cut points come from the observed training-set geometry: the
# median pairwise CDR identity inside the training panel is ~0.35, and the
# external cohort we validated on sits at ~0.63.
AD_BANDS = [
    (0.60, "in_domain", "comparable to the external cohort the model was validated on"),
    (0.45, "borderline", "less similar to training data than the validated external set"),
    (0.00, "extrapolation", "outside the region where the model was ever checked"),
]


def ad_band(identity: float) -> tuple[str, str]:
    for cut, label, note in AD_BANDS:
        if identity >= cut:
            return label, note
    return "extrapolation", AD_BANDS[-1][2]


# --------------------------------------------------------------------------
# Feature reconstruction for new sequences
# --------------------------------------------------------------------------

def featurise(df: pd.DataFrame, feature_set: str, expected_columns: list[str],
              device: str | None = None) -> np.ndarray:
    """Rebuild exactly the feature set the stored model was trained on."""
    seqs = [(h, (l if isinstance(l, str) else ""))
            for h, l in zip(df["heavy_chain"], df["light_chain"])]

    blocks: list[np.ndarray] = []
    needs_desc = feature_set.startswith("desc") or "desc_all" in feature_set
    desc_cols: list[str] = []

    if needs_desc:
        ann = nb.annotate_dataset(df, cache_name="screen_tmp", use_cache=False)
        g = D.build_global_descriptors(df)
        c = D.build_cdr_descriptors(df, ann)
        dsc = g.merge(c, on="antibody_id")
        desc_cols = [col for col in expected_columns if col in dsc.columns]
        missing = [col for col in expected_columns
                   if col not in dsc.columns and not col.startswith(("ablang2_", "esm2_"))]
        if missing:
            raise ValueError(
                f"{len(missing)} descriptor columns the model expects are missing, "
                f"e.g. {missing[:5]}. The descriptor code has changed since the "
                "model was trained; retrain before screening."
            )
        blocks.append(dsc[desc_cols].to_numpy(dtype=float))

    if "ablang2" in feature_set:
        blocks.append(plm.embed_ablang2(seqs, device=device, use_cache=False).astype(float))
    if "esm2" in feature_set:
        blocks.append(plm.embed_esm2(seqs, device=device, use_cache=False).astype(float))

    if not blocks:
        raise ValueError(f"Unrecognised feature set: {feature_set}")

    X = np.concatenate(blocks, axis=1) if len(blocks) > 1 else blocks[0]
    if X.shape[1] != len(expected_columns):
        raise ValueError(
            f"Rebuilt {X.shape[1]} features but the model expects "
            f"{len(expected_columns)}. Refusing to predict from a mismatched matrix."
        )
    return X


# --------------------------------------------------------------------------
# Liability reporting
# --------------------------------------------------------------------------

def liability_report(heavy: str, light: str) -> dict:
    out = {"n_total": 0, "n_in_cdr": 0, "details": []}
    for chain_label, seq in (("H", heavy), ("L", light)):
        if not seq:
            continue
        ann = nb.annotate_chain(seq, with_germline=False)
        for hit in ann.liabilities:
            if "description" not in hit:
                continue
            out["n_total"] += 1
            if hit.get("in_cdr"):
                out["n_in_cdr"] += 1
            out["details"].append({
                "chain": chain_label, "motif": hit["motif"],
                "position": hit["start"], "in_cdr": bool(hit.get("in_cdr")),
                "description": hit["description"],
            })
    return out


# --------------------------------------------------------------------------
# Main screening routine
# --------------------------------------------------------------------------

def screen(input_csv: Path, endpoint: str, output_csv: Path,
           report_md: Path | None = None, device: str | None = None) -> pd.DataFrame:
    from src.data import splits as S

    bundle_path = config.MODELS_DIR / f"final_{endpoint}.joblib"
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"No trained model at {bundle_path}. Run the pipeline first "
            f"(scripts/run_all.ps1)."
        )
    bundle = joblib.load(bundle_path)
    ep = config.ENDPOINTS[endpoint]

    df = pd.read_csv(input_csv)
    for col in ("heavy_chain", "light_chain"):
        if col not in df.columns:
            raise ValueError(f"Input CSV must contain a '{col}' column. "
                             f"Found: {list(df.columns)}")
    if "antibody_id" not in df.columns:
        df.insert(0, "antibody_id", [f"query_{i:04d}" for i in range(len(df))])

    df["heavy_chain"] = df["heavy_chain"].map(canonicalise_sequence)
    df["light_chain"] = df["light_chain"].map(canonicalise_sequence)
    bad = df["heavy_chain"].isna()
    if bad.any():
        print(f"[screen] dropping {int(bad.sum())} rows with unusable heavy chains")
        df = df.loc[~bad].reset_index(drop=True)
    df["light_chain"] = df["light_chain"].fillna("")

    print(f"[screen] scoring {len(df)} antibodies for '{ep.pretty_name}'")
    X = featurise(df, bundle["feature_set"], bundle["feature_columns"], device)
    preds = bundle["model"].predict(X)
    q = float(bundle["conformal_half_width"])
    thr = float(bundle["risk_threshold"])
    higher_is_worse = bundle["favourable_direction"] == "lower"

    # Applicability domain against the training cohort.
    train_ds = pd.read_csv(config.PROCESSED_DIR / f"dataset_{endpoint}.csv")
    train_ds = train_ds[train_ds["role"] == "train"].reset_index(drop=True)
    train_ann = pd.read_csv(config.PROCESSED_DIR / f"annotations_{endpoint}.csv",
                            keep_default_na=False, na_values=["__NA__"])
    train_ann = train_ann.set_index("antibody_id").reindex(
        train_ds["antibody_id"]).reset_index()
    query_ann = nb.annotate_dataset(df, cache_name="screen_ad", use_cache=False)
    identity = S.nearest_train_identity(query_ann, train_ann)

    out = df[["antibody_id"]].copy()
    out["predicted_value"] = preds
    out["units"] = ep.units
    out["pi_low"] = preds - q
    out["pi_high"] = preds + q
    out["pi_nominal_coverage"] = 1.0 - config.CONFORMAL_ALPHA
    out["risk_threshold"] = thr
    out["predicted_risk"] = ((preds >= thr) if higher_is_worse
                             else (preds <= thr)).astype(int)
    # An antibody whose whole interval sits on the bad side is a confident call;
    # one whose interval straddles the threshold is genuinely undecided, and
    # saying so is more useful than a hard 0/1.
    if higher_is_worse:
        out["risk_call"] = np.where(preds - q >= thr, "high_risk",
                          np.where(preds + q < thr, "low_risk", "uncertain"))
    else:
        out["risk_call"] = np.where(preds + q <= thr, "high_risk",
                          np.where(preds - q > thr, "low_risk", "uncertain"))
    out["max_cdr_identity_to_training"] = identity
    bands = [ad_band(float(v)) for v in identity]
    out["applicability"] = [b[0] for b in bands]
    out["applicability_note"] = [b[1] for b in bands]

    liab = [liability_report(h, l) for h, l in zip(df["heavy_chain"], df["light_chain"])]
    out["n_liability_motifs"] = [x["n_total"] for x in liab]
    out["n_liability_motifs_in_cdr"] = [x["n_in_cdr"] for x in liab]
    out["liability_detail"] = [json.dumps(x["details"]) for x in liab]

    out["rank"] = out["predicted_value"].rank(
        ascending=higher_is_worse, method="min").astype(int)
    out = out.sort_values("predicted_value", ascending=higher_is_worse).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"[screen] -> {output_csv}")

    if report_md:
        _write_report(out, ep, bundle, report_md)
    return out


def _write_report(out: pd.DataFrame, ep, bundle: dict, path: Path) -> None:
    n = len(out)
    counts = out["risk_call"].value_counts().to_dict()
    ad_counts = out["applicability"].value_counts().to_dict()

    lines = [
        f"# Developability screening report: {ep.pretty_name}",
        "",
        f"- Antibodies screened: **{n}**",
        f"- Model: `{bundle['feature_set']}` + `{bundle['model_name']}`",
        f"- Endpoint: {ep.pretty_name} ({ep.units}); "
        f"{'lower' if ep.favourable_direction == 'lower' else 'higher'} is better",
        f"- Risk threshold: {bundle['risk_threshold']:g} {ep.units} "
        f"(Jain et al. 2017 warning flag)",
        f"- Prediction intervals: +/-{bundle['conformal_half_width']:.2f} {ep.units} "
        f"at {int((1 - bundle['conformal_alpha']) * 100)}% nominal coverage",
        "",
        "## Risk calls",
        "",
        "| call | n | meaning |",
        "|---|---|---|",
        f"| high_risk | {counts.get('high_risk', 0)} | whole interval on the unfavourable side |",
        f"| uncertain | {counts.get('uncertain', 0)} | interval straddles the threshold |",
        f"| low_risk | {counts.get('low_risk', 0)} | whole interval on the favourable side |",
        "",
        "## Applicability domain",
        "",
        "| band | n |",
        "|---|---|",
    ]
    for band in ("in_domain", "borderline", "extrapolation"):
        lines.append(f"| {band} | {ad_counts.get(band, 0)} |")

    lines += [
        "",
        "## Ranked candidates",
        "",
        f"| rank | antibody | predicted {ep.units} | interval | risk | applicability | CDR liabilities |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in out.head(50).itertuples():
        lines.append(
            f"| {r.rank} | {r.antibody_id} | {r.predicted_value:.2f} | "
            f"[{r.pi_low:.2f}, {r.pi_high:.2f}] | {r.risk_call} | "
            f"{r.applicability} | {r.n_liability_motifs_in_cdr} |"
        )
    if n > 50:
        lines.append(f"\n*(showing the first 50 of {n}; the full table is in the CSV)*")

    lines += [
        "",
        "## How to read this",
        "",
        "The model ranks candidates by predicted assay value. It does not replace "
        "the assay. An `extrapolation` applicability band means the antibody is "
        "less similar to the training set than anything the model was validated "
        "on, so its prediction should be treated as unknown rather than as "
        "evidence of safety. `uncertain` risk calls are exactly that: the "
        "prediction interval covers the threshold, so the model cannot separate "
        "the two outcomes for that antibody.",
        "",
        "Liability motif counts are independent of the model. They are "
        "rule-based sequence annotations (glycosylation sequons, deamidation and "
        "isomerisation dipeptides, oxidation-prone residues) and are reported so "
        "that a candidate with a clean predicted value but an exposed CDR "
        "liability is not overlooked.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[screen] report -> {path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="abdev-screen",
        description="Sequence-based antibody developability screening.")
    ap.add_argument("--input", required=True, type=Path,
                    help="CSV with heavy_chain and light_chain columns")
    ap.add_argument("--endpoint", default=config.PRIMARY_ENDPOINT,
                    choices=sorted(config.ENDPOINTS))
    ap.add_argument("--output", type=Path,
                    default=config.RESULTS_DIR / "screening_results.csv")
    ap.add_argument("--report", type=Path, default=None,
                    help="optional Markdown report path")
    ap.add_argument("--device", default=None, help="cpu or cuda")
    args = ap.parse_args(argv)

    config.ensure_dirs()
    screen(args.input, args.endpoint, args.output, args.report, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
