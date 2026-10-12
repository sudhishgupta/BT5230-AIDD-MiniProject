"""
Error analysis, applicability-domain diagnostics and feature ablation.

A single headline correlation hides the questions that decide whether a model
is usable:

  * Is the error uniform, or concentrated in the tail that actually matters?
    A model that ranks the bulk well but cannot identify the worst antibodies
    is useless for triage, because triage is entirely about the tail.
  * Does the error grow as antibodies get less like the training set? If it
    does not, the applicability-domain flag is noise; if it does, the flag is
    doing real work.
  * Are the prediction intervals honest *within* each applicability band, or
    only on average?
  * Which descriptor families carry the signal, and how much is lost by
    dropping each one?

Run:
    python -m src.analysis.error_analysis
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from src import config
from src.analysis import plotting as P
from src.models import metrics as M
from src.models import train as T
from src.models import zoo as Z


# --------------------------------------------------------------------------
# Residual diagnostics on the training cohort
# --------------------------------------------------------------------------

def residual_diagnostics(key: str) -> dict:
    import matplotlib.pyplot as plt

    oof_path = config.PREDICTIONS_DIR / f"oof_{key}.csv"
    if not oof_path.exists():
        return {}
    oof = pd.read_csv(oof_path)
    ep = config.ENDPOINTS[key]
    ds = pd.read_csv(config.PROCESSED_DIR / f"dataset_{key}.csv")
    ann = pd.read_csv(config.PROCESSED_DIR / f"annotations_{key}.csv",
                      keep_default_na=False, na_values=["__NA__"])
    df = oof.merge(ds[["antibody_id", "heavy_len", "light_len", "is_censored"]],
                   on="antibody_id", how="left")
    df = df.merge(ann[["antibody_id", "heavy_v_gene"]], on="antibody_id", how="left")
    df["abs_error"] = df["residual"].abs()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # (a) predicted vs measured
    ax = axes[0]
    cen = df["is_censored"].fillna(False).to_numpy(dtype=bool)
    ax.scatter(df.loc[~cen, "oof_prediction"], df.loc[~cen, "value"], s=26,
               color=P.BLUE, alpha=0.75, edgecolor=P.SURFACE, linewidth=0.6,
               label="measured")
    if cen.any():
        ax.scatter(df.loc[cen, "oof_prediction"], df.loc[cen, "value"], s=52,
                   color=P.ORANGE, marker="^", edgecolor=P.SURFACE, linewidth=0.8,
                   label=f"censored at assay ceiling (n={int(cen.sum())})")
    lim = [min(df["value"].min(), df["oof_prediction"].min()) - 0.5,
           max(df["value"].max(), df["oof_prediction"].max()) + 0.5]
    ax.plot(lim, lim, color=P.INK_MUTED, linestyle="--", linewidth=1.2, zorder=0)
    thr = float(df["risk_threshold"].iloc[0]) if "risk_threshold" in df else None
    if thr:
        ax.axhline(thr, color=P.INK, linewidth=0.9, linestyle=":")
        ax.axvline(thr, color=P.INK, linewidth=0.9, linestyle=":")
    ax.set_xlabel(f"out-of-fold prediction ({ep.units})")
    ax.set_ylabel(f"measured ({ep.units})")
    ax.set_title("Predicted vs measured")
    ax.legend(loc="upper left", fontsize=7.5)
    P.despine(ax)

    # (b) residual vs measured -- does the model fail in the risky tail?
    ax = axes[1]
    ax.scatter(df["value"], df["residual"], s=26, color=P.BLUE, alpha=0.75,
               edgecolor=P.SURFACE, linewidth=0.6)
    ax.axhline(0, color=P.INK_MUTED, linewidth=1.2)
    if thr:
        ax.axvline(thr, color=P.ORANGE, linestyle="--", linewidth=1.4)
        ax.annotate("risk threshold", xy=(thr, ax.get_ylim()[1] * 0.85),
                    xytext=(5, 0), textcoords="offset points",
                    fontsize=8, color=P.ORANGE)
    ax.set_xlabel(f"measured ({ep.units})")
    ax.set_ylabel(f"residual (measured - predicted, {ep.units})")
    ax.set_title("Residual vs measured value")
    P.despine(ax)

    # (c) error by decile of the measurement
    ax = axes[2]
    df["decile"] = pd.qcut(df["value"], 5, labels=False, duplicates="drop")
    grp = df.groupby("decile")["abs_error"].agg(["mean", "std", "size"])
    ax.bar(grp.index, grp["mean"], yerr=grp["std"] / np.sqrt(grp["size"]),
           color=P.BLUE, width=0.68, capsize=3,
           error_kw={"ecolor": P.INK_MUTED, "elinewidth": 1})
    for i, (m, n) in enumerate(zip(grp["mean"], grp["size"])):
        ax.text(grp.index[i], m, f"n={int(n)}", ha="center", va="bottom",
                fontsize=7.5, color=P.INK_SECONDARY)
    ax.set_xticks(list(grp.index))
    ax.set_xticklabels([f"Q{i+1}" for i in range(len(grp))])
    ax.set_xlabel(f"quintile of measured {ep.pretty_name} (Q5 = riskiest)")
    ax.set_ylabel(f"mean |error| ({ep.units})")
    ax.set_title("Where the model fails")
    P.despine(ax)

    fig.suptitle(f"Error analysis: {ep.pretty_name} ({key})", fontsize=12,
                 fontweight="bold", color=P.INK, y=1.04)
    fig_path = P.save(fig, f"error_{key}_residuals")

    # Worst-predicted antibodies, for a concrete look at the failure mode.
    # AntPack returns every allele consistent with the sequence, which can be
    # a 150-character string. The gene family is what is actually informative
    # here, so the table carries that and the full call stays in the CSV.
    df["heavy_v_family"] = (df["heavy_v_gene"].fillna("").astype(str)
                            .str.split("*").str[0])
    worst = df.nlargest(15, "abs_error")[
        ["antibody_id", "value", "oof_prediction", "residual", "abs_error",
         "risk_label", "is_censored", "heavy_len", "heavy_v_family", "heavy_v_gene"]
    ]
    worst.to_csv(config.TABLES_DIR / f"error_{key}_worst_predictions.csv", index=False)

    tail = df[df["risk_label"] == 1]
    bulk = df[df["risk_label"] == 0]
    out = {
        "figure": fig_path,
        "mae_overall": float(df["abs_error"].mean()),
        "mae_risk_positive": float(tail["abs_error"].mean()) if len(tail) else None,
        "mae_risk_negative": float(bulk["abs_error"].mean()) if len(bulk) else None,
        "n_risk_positive": int(len(tail)),
        "error_vs_value_spearman": float(
            stats.spearmanr(df["value"], df["abs_error"]).statistic),
        "worst_predictions": worst.to_dict("records"),
    }
    if len(tail) and len(bulk):
        out["tail_error_ratio"] = out["mae_risk_positive"] / out["mae_risk_negative"]
    return out


# --------------------------------------------------------------------------
# Applicability domain vs error, on the external cohorts
# --------------------------------------------------------------------------

def error_by_germline(key: str, min_n: int = 5) -> dict:
    """
    Is the error concentrated in particular germline families?

    A model can rank well overall while being systematically wrong about one
    structurally distinctive group. That is a different failure from random
    noise and has a different remedy, so it is worth testing for explicitly
    rather than leaving it to be noticed in a table of worst predictions.

    Each family with at least ``min_n`` members is compared against all other
    antibodies on (a) its residuals, to find systematic over- or
    under-prediction, and (b) its measured values, to establish whether the
    family is genuinely extreme or the model merely thinks so. Both are
    Benjamini-Hochberg corrected across families.
    """
    import matplotlib.pyplot as plt

    oof_path = config.PREDICTIONS_DIR / f"oof_{key}.csv"
    if not oof_path.exists():
        return {}
    ep = config.ENDPOINTS[key]
    oof = pd.read_csv(oof_path)
    ann = pd.read_csv(config.PROCESSED_DIR / f"annotations_{key}.csv",
                      keep_default_na=False, na_values=["__NA__"])
    df = oof.merge(ann[["antibody_id", "heavy_v_gene"]], on="antibody_id", how="left")
    df["family"] = df["heavy_v_gene"].fillna("").astype(str).str.split("*").str[0]

    rows = []
    for fam, grp in df.groupby("family"):
        if len(grp) < min_n or len(grp) == len(df):
            continue
        rest = df[df["family"] != fam]
        rows.append({
            "family": fam, "n": int(len(grp)),
            "mean_residual": float(grp["residual"].mean()),
            "mean_residual_others": float(rest["residual"].mean()),
            "mean_measured": float(grp["value"].mean()),
            "mean_measured_others": float(rest["value"].mean()),
            "mae": float(grp["residual"].abs().mean()),
            "p_residual": float(stats.mannwhitneyu(
                grp["residual"], rest["residual"], alternative="two-sided").pvalue),
            "p_measured": float(stats.mannwhitneyu(
                grp["value"], rest["value"], alternative="two-sided").pvalue),
        })
    if not rows:
        return {}

    res = pd.DataFrame(rows)
    from src.models.metrics import benjamini_hochberg
    _, res["p_residual_bh"] = benjamini_hochberg(res["p_residual"].to_numpy())
    _, res["p_measured_bh"] = benjamini_hochberg(res["p_measured"].to_numpy())
    res["biased_fdr05"] = res["p_residual_bh"] <= 0.05
    res["extreme_fdr05"] = res["p_measured_bh"] <= 0.05
    res = res.sort_values("mean_residual", ascending=False).reset_index(drop=True)
    res.to_csv(config.TABLES_DIR / f"error_{key}_by_germline.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.6, 0.42 * len(res) + 1.8))
    colours = [P.RED if b else P.BLUE for b in res["biased_fdr05"]]
    y = range(len(res))
    ax.barh(list(y)[::-1], res["mean_residual"], color=colours, height=0.66)
    ax.set_yticks(list(y)[::-1])
    ax.set_yticklabels([f"{r.family}  (n={r.n})" for r in res.itertuples()],
                       fontsize=8.5)
    ax.axvline(0, color=P.INK_SECONDARY, linewidth=1.1)
    for i, r in enumerate(res.itertuples()):
        ax.text(r.mean_residual + (0.06 if r.mean_residual >= 0 else -0.06),
                len(res) - 1 - i,
                f"{r.mean_residual:+.2f}{'*' if r.biased_fdr05 else ''}",
                va="center", ha="left" if r.mean_residual >= 0 else "right",
                fontsize=8, color=P.INK_SECONDARY)
    ax.margins(x=0.25)
    ax.set_xlabel(f"mean residual, measured - predicted ({ep.units})")
    ax.set_title(f"Systematic error by heavy-chain V family ({key})\n"
                 f"positive = the model UNDER-predicts this family; "
                 f"* survives FDR 5%", fontsize=9.5)
    ax.grid(axis="y", visible=False)
    P.despine(ax)
    path = P.save(fig, f"error_{key}_by_germline")

    biased = res[res["biased_fdr05"]]
    return {
        "figure": path,
        "n_families_tested": int(len(res)),
        "n_biased_fdr05": int(len(biased)),
        "biased_families": biased.to_dict("records"),
        "table": res.to_dict("records"),
    }


def applicability_vs_error(key: str) -> dict:
    import matplotlib.pyplot as plt

    path = config.PREDICTIONS_DIR / f"external_{key}.csv"
    if not path.exists():
        return {}
    ext = pd.read_csv(path)
    ep = config.ENDPOINTS[key]
    ext["abs_error"] = (ext["value"] - ext["prediction"]).abs()

    cohorts = [c for c in ext["cohort"].unique()]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4))

    ax = axes[0]
    for i, c in enumerate(cohorts):
        sub = ext[ext["cohort"] == c]
        ax.scatter(sub["max_cdr_identity_to_train"], sub["abs_error"], s=30,
                   color=P.COHORT_COLOURS.get(c, P.CATEGORICAL[i % 3]),
                   alpha=0.78, edgecolor=P.SURFACE, linewidth=0.6,
                   label=f"{c} (n={len(sub)})")
    ax.set_xlabel("max CDR identity to any training antibody")
    ax.set_ylabel(f"|error| ({ep.units})")
    ax.set_title("Does error grow with distance from training?")
    ax.legend(fontsize=7.5)
    P.despine(ax)

    rows = []
    ax = axes[1]
    val = ext[ext["cohort"].isin(
        [c.key for c in ep.externals if c.evaluation_role == "external_validation"])]
    if len(val):
        bands = pd.qcut(val["max_cdr_identity_to_train"], 3,
                        labels=["least similar", "middle", "most similar"],
                        duplicates="drop")
        g = val.groupby(bands, observed=True)
        means = g["abs_error"].mean()
        covs = g["in_interval"].mean()
        x = np.arange(len(means))
        ax.bar(x - 0.19, means, width=0.36, color=P.BLUE, label=f"mean |error| ({ep.units})")
        ax2 = ax.twinx()
        ax2.bar(x + 0.19, covs, width=0.36, color=P.ORANGE,
                label="interval coverage")
        ax2.axhline(1 - config.CONFORMAL_ALPHA, color=P.INK, linestyle="--",
                    linewidth=1.2)
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("prediction-interval coverage")
        ax2.grid(False)
        ax.set_xticks(x)
        ax.set_xticklabels(list(means.index), fontsize=8)
        ax.set_ylabel(f"mean |error| ({ep.units})")
        ax.set_title("Error and coverage by applicability band")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="upper center")
        P.despine(ax)
        for name, grp in g:
            rows.append({
                "band": str(name), "n": int(len(grp)),
                "mean_identity": float(grp["max_cdr_identity_to_train"].mean()),
                "mae": float(grp["abs_error"].mean()),
                "interval_coverage": float(grp["in_interval"].mean()),
            })

    fig.suptitle(f"Applicability domain vs error ({key})", fontsize=12,
                 fontweight="bold", color=P.INK, y=1.04)
    fig_path = P.save(fig, f"error_{key}_applicability")

    band_df = pd.DataFrame(rows)
    if len(band_df):
        band_df.to_csv(config.TABLES_DIR / f"error_{key}_ad_bands.csv", index=False)

    out: dict = {"figure": fig_path, "bands": rows}
    if len(val) > 5:
        r = stats.spearmanr(val["max_cdr_identity_to_train"], val["abs_error"])
        out["identity_vs_error_spearman"] = float(r.statistic)
        out["identity_vs_error_p"] = float(r.pvalue)
    return out


# --------------------------------------------------------------------------
# Leave-one-block-out descriptor ablation
# --------------------------------------------------------------------------

DESCRIPTOR_BLOCKS = {
    "composition": ("_frac_",),
    "hydropathy": ("_kd_", "_eis_", "_bm_"),
    "charge_and_pI": ("_charge_", "_pI", "charge_symmetry", "net_charge"),
    "cdr_restricted": ("H_cdr", "L_cdr", "CDRall_"),
    "liability_motifs": ("_liab_",),
    "size_and_stability": ("_len", "_mw", "_instability", "_aromaticity",
                           "_n_cys", "_cys_odd"),
}


def descriptor_ablation(key: str, model_name: str = "ridge") -> pd.DataFrame:
    """
    Drop one descriptor family at a time and measure the loss.

    This answers "which physics is the model actually using?" in a way the
    feature-set sweep cannot: the sweep compares whole representations, while
    this isolates families inside the winning one.  Ridge is used because it
    is stable at p >> n and fast enough to repeat seven times.
    """
    ds = pd.read_csv(config.PROCESSED_DIR / f"dataset_{key}.csv")
    npz = np.load(config.PROCESSED_DIR / f"features_{key}.npz", allow_pickle=True)
    splits = json.loads((config.PROCESSED_DIR / f"splits_{key}.json").read_text(encoding="utf-8"))
    is_train = (ds["role"] == "train").to_numpy()

    X_all = npz["X__desc_all"][is_train]
    cols = [str(c) for c in npz["cols__desc_all"]]
    y = ds.loc[is_train, "value"].to_numpy(dtype=float)
    groups = np.asarray(splits["cluster_labels"])
    folds = [[(np.asarray(tr), np.asarray(te)) for tr, te in rep]
             for rep in splits["clustered_folds"]][: config.CV_N_REPEATS]

    est, grid = Z.regression_zoo()[model_name]

    base = T.run_nested_cv(X_all, y, groups, folds, est, grid,
                           "desc_all", model_name, "regression")
    base_rho = M.regression_metrics(y, base.representative_oof(0))["spearman"]

    rows = [{"ablation": "none (full descriptor set)", "n_features": X_all.shape[1],
             "n_removed": 0, "spearman": base_rho, "delta_vs_full": 0.0}]

    for block, patterns in DESCRIPTOR_BLOCKS.items():
        keep = [i for i, c in enumerate(cols)
                if not any(p in c for p in patterns)]
        if len(keep) < 3 or len(keep) == len(cols):
            continue
        res = T.run_nested_cv(X_all[:, keep], y, groups, folds, est, grid,
                              f"desc_all-minus-{block}", model_name, "regression")
        rho = M.regression_metrics(y, res.representative_oof(0))["spearman"]
        rows.append({
            "ablation": f"drop {block}", "n_features": len(keep),
            "n_removed": len(cols) - len(keep),
            "spearman": rho, "delta_vs_full": rho - base_rho,
        })
        print(f"    drop {block:<20} kept {len(keep):>3} feats  "
              f"rho={rho:.3f}  delta={rho - base_rho:+.3f}")

    # And the mirror image: each block ON ITS OWN.
    for block, patterns in DESCRIPTOR_BLOCKS.items():
        keep = [i for i, c in enumerate(cols) if any(p in c for p in patterns)]
        if len(keep) < 3:
            continue
        res = T.run_nested_cv(X_all[:, keep], y, groups, folds, est, grid,
                              f"only-{block}", model_name, "regression")
        rho = M.regression_metrics(y, res.representative_oof(0))["spearman"]
        rows.append({
            "ablation": f"only {block}", "n_features": len(keep),
            "n_removed": len(cols) - len(keep),
            "spearman": rho, "delta_vs_full": rho - base_rho,
        })
        print(f"    only {block:<20} kept {len(keep):>3} feats  "
              f"rho={rho:.3f}  delta={rho - base_rho:+.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(config.TABLES_DIR / f"ablation_{key}.csv", index=False)
    return df


def fig_ablation(key: str, df: pd.DataFrame) -> str | None:
    import matplotlib.pyplot as plt
    if df.empty:
        return None
    drops = df[df["ablation"].str.startswith("drop")].copy()
    onlys = df[df["ablation"].str.startswith("only")].copy()
    if drops.empty:
        return None
    drops["block"] = drops["ablation"].str.replace("drop ", "", regex=False)
    onlys["block"] = onlys["ablation"].str.replace("only ", "", regex=False)
    merged = drops[["block", "delta_vs_full"]].merge(
        onlys[["block", "spearman"]], on="block", how="outer")
    merged = merged.sort_values("delta_vs_full")

    base = float(df.loc[df["ablation"].str.startswith("none"), "spearman"].iloc[0])
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.42 * len(merged) + 2.4))

    ax = axes[0]
    ax.barh(range(len(merged)), merged["delta_vs_full"],
            color=[P.RED if v < 0 else P.AQUA for v in merged["delta_vs_full"]],
            height=0.66)
    ax.set_yticks(range(len(merged)))
    ax.set_yticklabels(merged["block"], fontsize=8.5)
    ax.axvline(0, color=P.INK_SECONDARY, linewidth=1.1)
    for i, v in enumerate(merged["delta_vs_full"]):
        ax.text(v + (0.004 if v >= 0 else -0.004), i, f"{v:+.3f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8,
                color=P.INK_SECONDARY)
    ax.margins(x=0.22)
    ax.set_xlabel("change in Spearman when this family is REMOVED")
    ax.set_title(f"Leave-one-family-out\n(full descriptor set: rho = {base:.3f})",
                 fontsize=10)
    ax.grid(axis="y", visible=False)
    P.despine(ax)

    ax = axes[1]
    sub = merged.dropna(subset=["spearman"]).sort_values("spearman")
    ax.barh(range(len(sub)), sub["spearman"], color=P.BLUE, height=0.66)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(sub["block"], fontsize=8.5)
    ax.axvline(base, color=P.ORANGE, linestyle="--", linewidth=1.5)
    ax.annotate(f"full set {base:.3f}", xy=(base, len(sub) - 0.5),
                xytext=(4, 0), textcoords="offset points", fontsize=8,
                color=P.ORANGE)
    for i, v in enumerate(sub["spearman"]):
        ax.text(v + 0.006, i, f"{v:.3f}", va="center", fontsize=8,
                color=P.INK_SECONDARY)
    ax.margins(x=0.22)
    ax.set_xlabel("Spearman using ONLY this family")
    ax.set_title("Each family alone", fontsize=10)
    ax.grid(axis="y", visible=False)
    P.despine(ax)

    fig.suptitle(f"Descriptor family ablation ({key})", fontsize=12,
                 fontweight="bold", color=P.INK, y=1.03)
    return P.save(fig, f"ablation_{key}")


def main() -> int:  # pragma: no cover
    config.ensure_dirs()
    P.apply_style()
    summary: dict = {}
    for key in config.ENDPOINTS:
        if not (config.PREDICTIONS_DIR / f"oof_{key}.csv").exists():
            print(f"[error_analysis] skipping {key}: run src.models.experiment first")
            continue
        print(f"\n[error_analysis] === {key} ===")
        resid = residual_diagnostics(key)
        germ = error_by_germline(key)
        ad = applicability_vs_error(key)
        if resid:
            print(f"  MAE overall={resid['mae_overall']:.3f}  "
                  f"risk+={resid.get('mae_risk_positive', float('nan')):.3f}  "
                  f"risk-={resid.get('mae_risk_negative', float('nan')):.3f}")
        if ad.get("identity_vs_error_spearman") is not None:
            print(f"  identity vs |error| rho="
                  f"{ad['identity_vs_error_spearman']:+.3f} "
                  f"(p={ad.get('identity_vs_error_p', float('nan')):.3f})")

        print("  -- descriptor family ablation --")
        abl = descriptor_ablation(key)
        fig = fig_ablation(key, abl)
        if germ.get("n_biased_fdr05"):
            for b in germ["biased_families"]:
                print(f"  germline bias: {b['family']:<12} n={b['n']:<3} "
                      f"mean residual {b['mean_residual']:+.2f} "
                      f"(p_BH={b['p_residual_bh']:.4f})")
        summary[key] = {"residuals": resid, "germline": germ, "applicability": ad,
                        "ablation": abl.to_dict("records"), "ablation_figure": fig}

    (config.RESULTS_DIR / "error_analysis_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("\n[error_analysis] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
