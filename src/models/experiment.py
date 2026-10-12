"""
The main experiment: feature-set x model sweep, external validation,
uncertainty quantification and significance testing.

Outputs (all under results/)
  tables/cv_regression_<endpoint>.csv        every feature set x model
  tables/cv_classification_<endpoint>.csv    same, for the risk label
  tables/external_<endpoint>.csv             held-out cohort performance
  tables/model_comparison_<endpoint>.csv     paired bootstrap comparisons
  predictions/oof_<endpoint>.csv             out-of-fold predictions
  predictions/external_<endpoint>.csv        external predictions + intervals
  <endpoint>_summary.json                    headline numbers and provenance
  models_store/final_<endpoint>.joblib       the deployable model
"""

from __future__ import annotations

import json
import sys
import time

import joblib
import numpy as np
import pandas as pd

from src import config
from src.models import metrics as M
from src.models import train as T
from src.models import zoo as Z


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_endpoint(key: str):
    ds = pd.read_csv(config.PROCESSED_DIR / f"dataset_{key}.csv")
    npz = np.load(config.PROCESSED_DIR / f"features_{key}.npz", allow_pickle=True)
    splits = json.loads((config.PROCESSED_DIR / f"splits_{key}.json").read_text(encoding="utf-8"))

    mats = {k[3:]: npz[k] for k in npz.files if k.startswith("X__")}
    cols = {k[6:]: list(npz[k]) for k in npz.files if k.startswith("cols__")}

    ids = list(npz["antibody_id"])
    assert ids == ds["antibody_id"].tolist(), "feature/dataset row order mismatch"

    is_train = (ds["role"] == "train").to_numpy()
    train_ids = splits["train_ids"]
    assert ds.loc[is_train, "antibody_id"].tolist() == train_ids, \
        "split train_ids do not match the dataset's training rows"

    groups = np.asarray(splits["cluster_labels"])
    folds = [[(np.asarray(tr), np.asarray(te)) for tr, te in rep]
             for rep in splits["clustered_folds"]]
    rand_folds = [[(np.asarray(tr), np.asarray(te)) for tr, te in rep]
                  for rep in splits["random_folds"]]

    return ds, mats, cols, is_train, groups, folds, rand_folds, splits


def external_slices(ds: pd.DataFrame, mats: dict, feature_set: str) -> dict[str, np.ndarray]:
    out = {}
    for cohort in ds.loc[ds["role"] == "external", "cohort"].unique():
        mask = (ds["cohort"] == cohort).to_numpy()
        out[cohort] = mats[feature_set][mask]
    return out


# --------------------------------------------------------------------------
# The sweep
# --------------------------------------------------------------------------

def run_sweep(key: str, task: str, n_repeats: int | None, feature_sets: list[str],
              quick: bool = False) -> tuple[pd.DataFrame, dict]:
    ds, mats, cols, is_train, groups, folds, rand_folds, splits = load_endpoint(key)
    ep = config.ENDPOINTS[key]

    y_all = ds["value"].to_numpy(dtype=float)
    y = y_all[is_train]
    if task == "classification":
        y = ds.loc[is_train, "risk_label"].to_numpy(dtype=int)

    use_folds = folds[: (n_repeats or len(folds))]
    zoo = Z.regression_zoo() if task == "regression" else Z.classification_zoo()
    if quick:
        zoo = {k: v for k, v in zoo.items() if k in ("ridge", "random_forest", "logistic")}

    rows, store = [], {}
    for fs in feature_sets:
        X = mats[fs][is_train]
        for model_name, (est, grid) in zoo.items():
            t0 = time.time()
            res = T.run_nested_cv(X, y, groups, use_folds, est, grid,
                                  fs, model_name, task=task)
            summ = T.summarise_cv(res)
            rows.append(summ)
            store[(fs, model_name)] = res
            primary = "spearman_mean" if task == "regression" else "roc_auc_mean"
            print(f"  [{task[:3]}] {fs:<20} {model_name:<14} "
                  f"{primary.replace('_mean',''):<9}="
                  f"{summ.get(primary, float('nan')):.3f}  "
                  f"({time.time() - t0:.0f}s)")

    df = pd.DataFrame(rows)
    sort_key = "spearman_mean" if task == "regression" else "roc_auc_mean"
    df = df.sort_values(sort_key, ascending=False).reset_index(drop=True)
    return df, store


# --------------------------------------------------------------------------
# Endpoint driver
# --------------------------------------------------------------------------

def run_endpoint(key: str, quick: bool = False, n_repeats: int | None = None,
                 do_permutation: bool = True) -> dict:
    print(f"\n{'=' * 72}\nENDPOINT: {key}\n{'=' * 72}")
    ds, mats, cols, is_train, groups, folds, rand_folds, splits = load_endpoint(key)
    ep = config.ENDPOINTS[key]

    feature_sets = [f for f in config.FEATURE_SETS if f in mats]
    y_train = ds.loc[is_train, "value"].to_numpy(dtype=float)
    cls_train = ds.loc[is_train, "risk_label"].to_numpy(dtype=int)

    summary: dict = {
        "endpoint": key,
        "pretty_name": ep.pretty_name,
        "units": ep.units,
        "favourable_direction": ep.favourable_direction,
        "n_train": int(is_train.sum()),
        "train_cohort": ep.train.key,
        "feature_sets": {f: int(mats[f].shape[1]) for f in feature_sets},
        "split_report": splits["report"],
    }

    # ---- regression sweep -------------------------------------------------
    print("\n-- regression sweep --")
    reg_df, reg_store = run_sweep(key, "regression", n_repeats, feature_sets, quick)
    reg_df.to_csv(config.TABLES_DIR / f"cv_regression_{key}.csv", index=False)

    # ---- classification sweep --------------------------------------------
    print("\n-- classification sweep --")
    clf_df, clf_store = run_sweep(key, "classification", n_repeats, feature_sets, quick)
    clf_df.to_csv(config.TABLES_DIR / f"cv_classification_{key}.csv", index=False)

    # ---- headline model ---------------------------------------------------
    best = reg_df.iloc[0]
    best_fs, best_model = best["feature_set"], best["model"]
    print(f"\n-- headline: {best_fs} + {best_model} "
          f"(Spearman {best['spearman_mean']:.3f} +/- {best.get('spearman_std', 0):.3f}) --")
    summary["best_regression"] = {
        "feature_set": best_fs, "model": best_model,
        **{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
           for k, v in best.items() if k not in ("feature_set", "model", "task")},
    }
    best_clf = clf_df.iloc[0]
    summary["best_classification"] = {
        "feature_set": best_clf["feature_set"], "model": best_clf["model"],
        **{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
           for k, v in best_clf.items() if k not in ("feature_set", "model", "task")},
    }

    # ---- out-of-fold metrics with CIs ------------------------------------
    # Use ONE repeat's complete out-of-fold vector, not the across-repeat
    # average.  Each repeat already predicts every antibody exactly once from
    # a model that never saw it; averaging repeats would bag five models and
    # report an ensemble's performance under a single model's name.
    best_res = reg_store[(best_fs, best_model)]
    oof = best_res.representative_oof(0)
    oof_ensemble = best_res.ensemble_mean_oof()
    pooled = M.regression_metrics(y_train, oof)
    cis = {}
    for m in ("spearman", "r2", "mae", "rmse", "pearson"):
        lo, hi = M.bootstrap_ci(y_train, oof, m, "regression", seed=config.SEED)
        cis[m] = {"value": pooled[m], "ci_low": lo, "ci_high": hi}
    summary["pooled_oof_regression"] = {
        "metrics": pooled,
        "ci": cis,
        "note": ("Computed on repeat 0's complete out-of-fold vector. The "
                 "across-repeat mean and standard deviation of the same metric "
                 "are in best_regression.*_mean / *_std."),
        "ensemble_mean_spearman": float(
            M.regression_metrics(y_train, oof_ensemble)["spearman"]),
    }

    perm = M.permutation_pvalue(pooled["spearman"], y_train, oof, "spearman",
                                seed=config.SEED)
    summary["permutation_correlation_only"] = perm

    # ---- the expensive, full-pipeline permutation test --------------------
    if do_permutation and not quick:
        print("-- full-refit permutation test (this takes a while) --")
        est, grid = Z.regression_zoo()[best_model]
        # The null must be built the same way as the observed statistic:
        # both use repeat 0's folds and repeat 0's single out-of-fold vector.
        summary["permutation_full_refit"] = T.refit_permutation_test(
            mats[best_fs][is_train], y_train, groups, folds[0], est, grid,
            pooled["spearman"], n_perm=100,
        )
        print(f"   p = {summary['permutation_full_refit']['p_value']:.4f}")

    # ---- clustered vs random split ablation -------------------------------
    print("-- split-protocol ablation (clustered vs random) --")
    est, grid = Z.regression_zoo()[best_model]
    # The random arm must be random all the way down. Passing the real cluster
    # labels as `groups` would keep the INNER hyper-parameter search grouped by
    # CDR cluster, so the "random" arm would still be protected against
    # near-duplicate leakage during tuning and the comparison would understate
    # how optimistic a fully random protocol is. Giving every antibody its own
    # group makes the inner split ungrouped, which is what a naive pipeline
    # would actually do.
    ungrouped = np.arange(len(y_train))
    rand_res = T.run_nested_cv(mats[best_fs][is_train], y_train, ungrouped,
                               rand_folds[: (n_repeats or len(rand_folds))],
                               est, grid, best_fs, best_model, "regression")

    # Compare the ACROSS-REPEAT MEANS, not two single repeats. A single
    # repeat's Spearman moves by several hundredths purely from fold
    # assignment, so differencing two of them would attribute that noise to
    # the splitting protocol. The standard deviations are reported alongside
    # so the reader can judge whether the gap exceeds the noise.
    def _repeat_stats(res):
        vals = np.array([m["spearman"] for m in res.metrics_per_repeat], dtype=float)
        maes = np.array([m["mae"] for m in res.metrics_per_repeat], dtype=float)
        return (float(vals.mean()),
                float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                float(maes.mean()), int(len(vals)))

    c_rho, c_sd, c_mae, c_n = _repeat_stats(best_res)
    r_rho, r_sd, r_mae, r_n = _repeat_stats(rand_res)
    gap = r_rho - c_rho
    # Standard error of the difference between two independent repeat means.
    se = float(np.sqrt((c_sd ** 2) / max(c_n, 1) + (r_sd ** 2) / max(r_n, 1)))
    summary["split_protocol_ablation"] = {
        "clustered_spearman": c_rho, "clustered_spearman_sd": c_sd,
        "random_spearman": r_rho, "random_spearman_sd": r_sd,
        "n_repeats": c_n,
        "optimism_from_random_split": gap,
        "optimism_standard_error": se,
        "optimism_exceeds_2se": bool(abs(gap) > 2 * se) if se > 0 else False,
        "clustered_mae": c_mae, "random_mae": r_mae,
    }
    print(f"   clustered rho={c_rho:.3f}+/-{c_sd:.3f}  "
          f"random rho={r_rho:.3f}+/-{r_sd:.3f}  "
          f"optimism={gap:+.3f} (SE {se:.3f})")

    # ---- paired model comparisons ----------------------------------------
    comps = []
    ref = oof
    for (fs, mdl), res in reg_store.items():
        if (fs, mdl) == (best_fs, best_model):
            continue
        d = M.paired_bootstrap_difference(y_train, ref, res.representative_oof(0),
                                          "spearman", seed=config.SEED)
        comps.append({"reference": f"{best_fs}+{best_model}",
                      "challenger": f"{fs}+{mdl}", **d})
    comp_df = pd.DataFrame(comps).sort_values("difference")
    if len(comp_df):
        # Two corrections the raw p-values need before anyone reads them.
        #
        # (1) Multiplicity: the reference is compared against every other
        #     configuration on the same 137 antibodies, so we control the FDR
        #     across that family of tests.
        # (2) Selection bias, which no correction removes: the reference was
        #     chosen as the ARGMAX of this very sweep on these very antibodies,
        #     so the differences are biased in its favour by construction. The
        #     column below records that explicitly so the table cannot be read
        #     as an unbiased head-to-head.
        _, padj = M.benjamini_hochberg(comp_df["p_two_sided"].to_numpy())
        comp_df["p_two_sided_bh"] = padj
        comp_df["reference_selected_on_same_data"] = True
    comp_df.to_csv(config.TABLES_DIR / f"model_comparison_{key}.csv", index=False)

    # ---- external validation ---------------------------------------------
    print("-- external validation --")
    X_tr = mats[best_fs][is_train]
    X_ext = external_slices(ds, mats, best_fs)
    est, grid = Z.regression_zoo()[best_model]
    model, best_params, ext_preds = T.fit_full_and_predict(
        X_tr, y_train, groups, est, grid, X_ext, "regression")
    summary["final_model_params"] = {str(k): str(v) for k, v in best_params.items()}

    # Conformal calibration on the pooled out-of-fold residuals.
    lo_tr, hi_tr, q = M.conformal_intervals(y_train - oof, oof)
    summary["conformal"] = {
        "alpha": config.CONFORMAL_ALPHA,
        "half_width": q,
        # Deliberately NOT reported as evidence of calibration: q is the
        # (1-alpha) quantile of these very residuals, so coverage measured on
        # them is forced to approximately 1-alpha by construction, whatever the
        # model does.  Only the external cohorts give a real coverage test.
        "calibration_set_coverage_is_tautological": True,
        "calibration_set_coverage": M.interval_coverage(y_train, lo_tr, hi_tr),
        "note": ("Calibrated on repeat 0's single-model out-of-fold residuals, "
                 "so the calibration predictor matches the variance of the "
                 "single refitted model the intervals are applied to."),
    }

    ext_rows, ext_pred_rows = [], []
    ad = json.loads((config.PROCESSED_DIR / f"applicability_{key}.json").read_text(encoding="utf-8"))

    # The risk threshold used to define the labels. It is identical for every
    # cohort by construction (curate.py derives it once from the training
    # distribution, or takes the published value, and applies it unchanged),
    # which is exactly what makes a cross-cohort classification comparison
    # meaningful.
    thr = float(ds.loc[is_train, "risk_threshold"].iloc[0])

    for cohort_key, preds in ext_preds.items():
        mask = (ds["cohort"] == cohort_key).to_numpy()
        y_ext = ds.loc[mask, "value"].to_numpy(dtype=float)
        cohort_cfg = next((c for c in ep.externals if c.key == cohort_key), None)
        role = cohort_cfg.evaluation_role if cohort_cfg else "external_validation"
        comparable = ep.external_absolute_comparable.get(cohort_key, False)

        m = M.regression_metrics(y_ext, preds)
        rho_lo, rho_hi = M.bootstrap_ci(y_ext, preds, "spearman", "regression",
                                        seed=config.SEED)
        lo, hi = preds - q, preds + q
        cov = M.interval_coverage(y_ext, lo, hi)

        row = {
            "cohort": cohort_key, "evaluation_role": role,
            "absolute_scale_comparable": comparable,
            "n": int(mask.sum()), **m,
            "spearman_ci_low": rho_lo, "spearman_ci_high": rho_hi,
            "conformal_coverage": cov["coverage"],
            "conformal_width": cov["mean_width"],
        }
        # Censored cohorts: repeat the rank metric on the uncensored subset,
        # because a third of the labels being tied at the assay ceiling makes
        # the full-set Spearman mostly a statement about that tie block.
        if "is_censored" in ds.columns:
            unc = (~ds.loc[mask, "is_censored"].to_numpy(dtype=bool))
            if unc.sum() >= 10 and unc.sum() < mask.sum():
                mu = M.regression_metrics(y_ext[unc], preds[unc])
                row["spearman_uncensored"] = mu["spearman"]
                row["n_uncensored"] = int(unc.sum())

        # Treat the regression output as a risk score and threshold it at the
        # SAME cut used to define the label.  The predictions are on the
        # assay's own scale, so this needs no calibration and -- unlike a
        # quantile of the predicted scores -- uses no information from the
        # external labels at all.
        #
        # The score must be ORIENTED first.  classification_metrics assumes a
        # higher score means a higher chance of being the positive (risky)
        # class.  That holds for HIC, AC-SINS and PSR, where high is bad, but
        # it is exactly backwards for Fab Tm, where LOW melting temperature is
        # the liability.  Feeding raw predictions in for Tm would report
        # ROC-AUC values mirrored about 0.5, turning a good model into an
        # apparently terrible one.
        risky_is_high = ep.favourable_direction == "lower"
        score = preds if risky_is_high else -preds
        score_thr = thr if risky_is_high else -thr
        cls_ext = ds.loc[mask, "risk_label"].to_numpy(dtype=int)
        cm = M.classification_metrics(cls_ext, score, threshold=score_thr)
        row.update({f"cls_{k}": v for k, v in cm.items()})
        ext_rows.append(row)

        sub = ds.loc[mask, ["antibody_id", "cohort", "value", "risk_label"]].copy()
        sub["prediction"] = preds
        sub["pi_low"], sub["pi_high"] = lo, hi
        sub["in_interval"] = (y_ext >= lo) & (y_ext <= hi)
        sub["max_cdr_identity_to_train"] = [
            ad.get(cohort_key, {}).get("per_antibody", {}).get(a, np.nan)
            for a in sub["antibody_id"]
        ]
        ext_pred_rows.append(sub)

        print(f"   {cohort_key:<16} [{role}] n={row['n']:<3} "
              f"rho={m['spearman']:+.3f} [{rho_lo:+.3f},{rho_hi:+.3f}] "
              f"MAE={m['mae']:.2f} coverage={cov['coverage']:.2f}")

    pd.DataFrame(ext_rows).to_csv(config.TABLES_DIR / f"external_{key}.csv", index=False)
    if ext_pred_rows:
        pd.concat(ext_pred_rows).to_csv(
            config.PREDICTIONS_DIR / f"external_{key}.csv", index=False)

    # ---- persist OOF predictions and the final model ----------------------
    oof_df = ds.loc[is_train, ["antibody_id", "cohort", "value", "risk_label"]].copy()
    oof_df["oof_prediction"] = oof
    oof_df["residual"] = y_train - oof
    oof_df["pi_low"], oof_df["pi_high"] = lo_tr, hi_tr
    for (fs, mdl), res in reg_store.items():
        oof_df[f"pred__{fs}__{mdl}"] = res.representative_oof(0)
    oof_df.to_csv(config.PREDICTIONS_DIR / f"oof_{key}.csv", index=False)

    joblib.dump({
        "model": model, "feature_set": best_fs, "model_name": best_model,
        "endpoint": key, "feature_columns": cols[best_fs],
        "conformal_half_width": q, "conformal_alpha": config.CONFORMAL_ALPHA,
        "risk_threshold": float(ds.loc[is_train, "risk_threshold"].iloc[0]),
        "favourable_direction": ep.favourable_direction,
        "train_median": float(np.median(y_train)),
        "seed": config.SEED,
    }, config.MODELS_DIR / f"final_{key}.joblib")

    (config.RESULTS_DIR / f"summary_{key}.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def main() -> int:
    config.ensure_dirs()
    quick = "--quick" in sys.argv
    only = None
    for a in sys.argv[1:]:
        if a.startswith("--endpoint="):
            only = a.split("=", 1)[1]
    n_repeats = None
    for a in sys.argv[1:]:
        if a.startswith("--repeats="):
            n_repeats = int(a.split("=", 1)[1])

    keys = [only] if only else [config.PRIMARY_ENDPOINT, *config.SECONDARY_ENDPOINTS]
    all_summ = {}
    for k in keys:
        all_summ[k] = run_endpoint(k, quick=quick, n_repeats=n_repeats,
                                   do_permutation=(k == config.PRIMARY_ENDPOINT))
    (config.RESULTS_DIR / "summary_all.json").write_text(
        json.dumps(all_summ, indent=2, default=str), encoding="utf-8")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
