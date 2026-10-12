"""
Exploratory data analysis and provenance figures.

These figures exist to let a reader check the study's foundations before
believing any model number: are the cohorts on the same scale, how censored
are they, how redundant is the training panel, and how far outside it do the
external antibodies sit.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from src import config
from src.analysis import plotting as P
from src.features import numbering as nb


def _load(key: str):
    ds = pd.read_csv(config.PROCESSED_DIR / f"dataset_{key}.csv")
    ann = pd.read_csv(config.PROCESSED_DIR / f"annotations_{key}.csv",
                      keep_default_na=False, na_values=["__NA__"])
    return ds, ann


# --------------------------------------------------------------------------
# 1. Target distributions across cohorts
# --------------------------------------------------------------------------

def fig_target_distributions(key: str) -> dict:
    import matplotlib.pyplot as plt

    ds, _ = _load(key)
    ep = config.ENDPOINTS[key]
    cohorts = list(ds["cohort"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4),
                             gridspec_kw={"width_ratios": [1.45, 1]})

    ax = axes[0]
    for i, c in enumerate(cohorts):
        v = ds.loc[ds["cohort"] == c, "value"].to_numpy(dtype=float)
        colour = P.COHORT_COLOURS.get(c, P.CATEGORICAL[i % 3])
        ax.hist(v, bins=22, alpha=0.62, color=colour, label=f"{c}  (n={len(v)})",
                edgecolor=P.SURFACE, linewidth=0.6)
    thr = float(ds["risk_threshold"].iloc[0])
    ax.axvline(thr, color=P.INK, linestyle="--", linewidth=1.4)
    ax.annotate(f"risk threshold\n{thr:g} {ep.units}", xy=(thr, ax.get_ylim()[1] * 0.88),
                xytext=(6, 0), textcoords="offset points", fontsize=8,
                color=P.INK, va="top")
    ax.set_xlabel(f"{ep.pretty_name} ({ep.units})")
    ax.set_ylabel("antibodies")
    ax.set_title("Measurement distribution by cohort")
    ax.legend(loc="upper right")
    P.despine(ax)

    ax = axes[1]
    data, labels, colours = [], [], []
    for i, c in enumerate(cohorts):
        data.append(ds.loc[ds["cohort"] == c, "value"].to_numpy(dtype=float))
        labels.append(c.replace("_" + key, ""))
        colours.append(P.COHORT_COLOURS.get(c, P.CATEGORICAL[i % 3]))
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=True,
                    medianprops={"color": P.INK, "linewidth": 1.6},
                    flierprops={"marker": "o", "markersize": 3,
                                "markerfacecolor": P.INK_MUTED,
                                "markeredgecolor": "none", "alpha": 0.6})
    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.62)
        patch.set_edgecolor(P.SURFACE)
    for el in ("whiskers", "caps"):
        for art in bp[el]:
            art.set_color(P.INK_MUTED)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel(f"{ep.units}")
    ax.set_title("Absolute-scale comparability")
    P.despine(ax)

    fig.suptitle(f"{ep.pretty_name}: cohort comparison", fontsize=12,
                 fontweight="bold", color=P.INK, y=1.02)
    path = P.save(fig, f"eda_{key}_target_distributions")

    rows = []
    for c in cohorts:
        v = ds.loc[ds["cohort"] == c, "value"].to_numpy(dtype=float)
        cen = ds.loc[ds["cohort"] == c, "is_censored"].to_numpy(dtype=bool)
        rows.append({
            "cohort": c, "n": len(v), "min": v.min(), "q1": np.percentile(v, 25),
            "median": np.median(v), "q3": np.percentile(v, 75), "max": v.max(),
            "mean": v.mean(), "std": v.std(ddof=1),
            "n_censored": int(cen.sum()),
            "frac_censored": float(cen.mean()),
            "n_risk_positive": int(ds.loc[ds["cohort"] == c, "risk_label"].sum()),
        })
    tbl = pd.DataFrame(rows)
    tbl.to_csv(config.TABLES_DIR / f"eda_{key}_cohort_summary.csv", index=False)
    return {"figure": path, "table": tbl.to_dict("records")}


# --------------------------------------------------------------------------
# 2. Endpoint correlation on the shared 2017 panel
# --------------------------------------------------------------------------

def fig_endpoint_correlations() -> dict:
    import matplotlib.pyplot as plt

    path_in = config.PROCESSED_DIR / "jain2017_multi_endpoint.csv"
    if not path_in.exists():
        return {}
    df = pd.read_csv(path_in)
    keys = [k for k in [config.PRIMARY_ENDPOINT, *config.SECONDARY_ENDPOINTS]
            if k in df.columns]
    sub = df[keys].dropna()

    rho = np.zeros((len(keys), len(keys)))
    pval = np.ones((len(keys), len(keys)))
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            r = stats.spearmanr(sub[a], sub[b])
            rho[i, j], pval[i, j] = r.statistic, r.pvalue

    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(rho, cmap=P.DIVERGING, vmin=-1, vmax=1)
    names = [config.ENDPOINTS[k].pretty_name for k in keys]
    ax.set_xticks(range(len(keys)))
    ax.set_yticks(range(len(keys)))
    ax.set_xticklabels(names, rotation=28, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    for i in range(len(keys)):
        for j in range(len(keys)):
            star = "*" if pval[i, j] < 0.05 and i != j else ""
            ax.text(j, i, f"{rho[i, j]:.2f}{star}", ha="center", va="center",
                    fontsize=8.5,
                    color="white" if abs(rho[i, j]) > 0.55 else P.INK)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.82)
    cb.set_label("Spearman rho", fontsize=8.5, color=P.INK_SECONDARY)

    # Title generated from the matrix. A fixed "endpoints are weakly related"
    # headline would contradict the figure the moment any pair turned out to be
    # strongly correlated -- and on this panel one of them is.
    off = rho[~np.eye(len(keys), dtype=bool)]
    strongest = float(np.max(np.abs(off)))
    median_abs = float(np.median(np.abs(off)))
    headline = ("Developability endpoints are mostly, but not entirely, independent"
                if strongest >= 0.5 else
                "Developability endpoints are only weakly related")
    ax.set_title(f"{headline}\nJain 2017 panel, n={len(sub)}; "
                 f"median |rho| {median_abs:.2f}, max |rho| {strongest:.2f}  "
                 f"(* p<0.05)", fontsize=9.5)
    p = P.save(fig, "eda_endpoint_correlations")

    out = pd.DataFrame(rho, index=keys, columns=keys)
    out.to_csv(config.TABLES_DIR / "eda_endpoint_correlations.csv")
    return {"figure": p, "spearman": out.to_dict()}


# --------------------------------------------------------------------------
# 3. Sequence / CDR properties
# --------------------------------------------------------------------------

def fig_sequence_properties(key: str) -> dict:
    import matplotlib.pyplot as plt

    ds, ann = _load(key)
    m = ds.merge(ann, on="antibody_id", how="left")
    m["h3_len"] = m["H_cdr3"].fillna("").astype(str).str.len()
    m["l3_len"] = m["L_cdr3"].fillna("").astype(str).str.len()
    m["cdr_total"] = sum(
        m[f"{c}_{r}"].fillna("").astype(str).str.len()
        for c in ("H", "L") for r in nb.CDR_REGIONS
    )
    cohorts = list(m["cohort"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, col, title in zip(
        axes, ["heavy_len", "h3_len", "cdr_total"],
        ["Heavy chain length", "CDR-H3 length", "Total CDR length (IMGT)"],
    ):
        for i, c in enumerate(cohorts):
            v = m.loc[m["cohort"] == c, col].to_numpy(dtype=float)
            colour = P.COHORT_COLOURS.get(c, P.CATEGORICAL[i % 3])
            ax.hist(v, bins=18, alpha=0.6, color=colour,
                    label=f"{c} (n={len(v)})", edgecolor=P.SURFACE, linewidth=0.6)
        ax.set_xlabel("residues")
        ax.set_ylabel("antibodies")
        ax.set_title(title)
        P.despine(ax)
    axes[0].legend(loc="upper right", fontsize=7.5)
    fig.suptitle(f"Sequence geometry across cohorts ({key})", fontsize=12,
                 fontweight="bold", color=P.INK, y=1.04)
    p1 = P.save(fig, f"eda_{key}_sequence_properties")

    # V-gene usage, training cohort only
    tr = m[m["role"] == "train"].copy()
    tr["vfam"] = tr["heavy_v_gene"].fillna("").astype(str).str.split("*").str[0]
    top = tr["vfam"].value_counts().head(12)
    fig, ax = plt.subplots(figsize=(7.2, 4))
    ax.barh(range(len(top))[::-1], top.to_numpy(), color=P.BLUE, height=0.68)
    ax.set_yticks(range(len(top))[::-1])
    ax.set_yticklabels(top.index, fontsize=8)
    for i, v in enumerate(top.to_numpy()):
        ax.text(v + 0.25, len(top) - 1 - i, str(int(v)), va="center",
                fontsize=8, color=P.INK_SECONDARY)
    ax.set_xlabel("antibodies")
    ax.set_title(f"Heavy-chain V-gene usage, training cohort (n={len(tr)})")
    ax.grid(axis="y", visible=False)
    P.despine(ax)
    p2 = P.save(fig, f"eda_{key}_vgene_usage")
    return {"figures": [p1, p2]}


# --------------------------------------------------------------------------
# 4. Redundancy and applicability domain
# --------------------------------------------------------------------------

def fig_identity_and_applicability(key: str) -> dict:
    import matplotlib.pyplot as plt
    from src.data import splits as S

    ds, ann = _load(key)
    ann = ann.set_index("antibody_id").reindex(ds["antibody_id"]).reset_index()
    is_train = (ds["role"] == "train").to_numpy()
    ident = S.cdr_identity_matrix(ann.loc[is_train].reset_index(drop=True),
                                  cache_tag=f"{key}_train")
    off = ident[~np.eye(len(ident), dtype=bool)]

    ad_path = config.PROCESSED_DIR / f"applicability_{key}.json"
    ad = json.loads(ad_path.read_text(encoding="utf-8")) if ad_path.exists() else {}

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4))

    ax = axes[0]
    ax.hist(off, bins=40, color=P.BLUE, alpha=0.8, edgecolor=P.SURFACE,
            linewidth=0.5)
    # Log scale, because the whole point of this panel is the TAIL. On a linear
    # axis the handful of near-duplicate pairs -- the only ones that can
    # actually leak -- are a sub-pixel bar next to a mode of >2000, so the
    # figure would appear to show that no such pairs exist.
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.7)
    n_above = int((off > config.CDR_IDENTITY_THRESHOLD).sum() // 2)
    ax.axvline(config.CDR_IDENTITY_THRESHOLD, color=P.ORANGE, linestyle="--",
               linewidth=1.6)
    ax.annotate(f"clustering cut {config.CDR_IDENTITY_THRESHOLD:g}\n"
                f"{n_above} pairs merged",
                xy=(config.CDR_IDENTITY_THRESHOLD, ax.get_ylim()[1] * 0.35),
                xytext=(-6, 0), textcoords="offset points", fontsize=8,
                color=P.ORANGE, va="top", ha="right")
    ax.set_xlabel("pairwise CDR identity")
    ax.set_ylabel("antibody pairs (log scale)")
    ax.set_title(f"Training panel is non-redundant, but not entirely\n"
                 f"median identity {np.median(off):.2f}; "
                 f"max {off.max():.2f}; {n_above} pairs above the cut")
    P.despine(ax)

    ax = axes[1]
    for i, (cohort, info) in enumerate(ad.items()):
        sims = np.array(list(info["per_antibody"].values()), dtype=float)
        colour = P.COHORT_COLOURS.get(cohort, P.CATEGORICAL[(i + 1) % 3])
        ax.hist(sims, bins=18, alpha=0.62, color=colour,
                label=f"{cohort} (n={len(sims)})", edgecolor=P.SURFACE,
                linewidth=0.6)
    ax.set_xlabel("max CDR identity to any training antibody")
    ax.set_ylabel("antibodies")
    ax.set_title("Applicability domain of the external cohorts")
    ax.legend(loc="upper right", fontsize=8)
    P.despine(ax)

    fig.suptitle(f"Sequence redundancy and extrapolation ({key})", fontsize=12,
                 fontweight="bold", color=P.INK, y=1.03)
    p = P.save(fig, f"eda_{key}_identity_applicability")

    # Name the near-duplicate pairs. The aggregate effect of clustering on this
    # panel is small, but "small" is not "none", and a reader should be able to
    # see the actual antibodies the protocol protects against rather than take
    # the claim on trust.
    ann_tr = ann.loc[is_train].reset_index(drop=True)
    ids = ds.loc[is_train, "antibody_id"].to_numpy()
    vals = ds.loc[is_train, "value"].to_numpy(dtype=float)
    iu = np.triu_indices(len(ident), 1)
    flat = ident[iu]
    order = np.argsort(-flat)[:15]
    rows = []
    for o in order:
        i, j = int(iu[0][o]), int(iu[1][o])
        rows.append({
            "identity": float(flat[o]),
            "antibody_a": ids[i], "antibody_b": ids[j],
            "value_a": float(vals[i]), "value_b": float(vals[j]),
            "value_difference": float(abs(vals[i] - vals[j])),
            "cdrh3_a": str(ann_tr.get("H_cdr3", pd.Series([""] * len(ann_tr)))[i]),
            "cdrh3_b": str(ann_tr.get("H_cdr3", pd.Series([""] * len(ann_tr)))[j]),
            "v_gene_a": str(ann_tr.get("heavy_v_gene", pd.Series([""] * len(ann_tr)))[i]).split("*")[0],
            "v_gene_b": str(ann_tr.get("heavy_v_gene", pd.Series([""] * len(ann_tr)))[j]).split("*")[0],
            "merged_by_clustering": bool(flat[o] >= config.CDR_IDENTITY_THRESHOLD),
        })
    near = pd.DataFrame(rows)
    near.to_csv(config.TABLES_DIR / f"eda_{key}_near_duplicates.csv", index=False)

    return {"figure": p,
            "median_pairwise_identity": float(np.median(off)),
            "max_pairwise_identity": float(off.max()),
            "pairs_above_threshold": int((off > config.CDR_IDENTITY_THRESHOLD).sum() // 2),
            "near_duplicates": near.head(6).to_dict("records")}


# --------------------------------------------------------------------------
# 5. Univariate descriptor associations
# --------------------------------------------------------------------------

def fig_univariate_associations(key: str, top_n: int = 20) -> dict:
    import matplotlib.pyplot as plt
    from src.models.metrics import benjamini_hochberg

    ds, _ = _load(key)
    dsc = pd.read_csv(config.PROCESSED_DIR / f"descriptors_{key}.csv")
    dsc = dsc.set_index("antibody_id").reindex(ds["antibody_id"]).reset_index()
    is_train = (ds["role"] == "train").to_numpy()
    y = ds.loc[is_train, "value"].to_numpy(dtype=float)

    rows = []
    for col in dsc.columns:
        if col == "antibody_id":
            continue
        x = dsc.loc[is_train, col].to_numpy(dtype=float)
        if not np.isfinite(x).all() or np.std(x) == 0:
            continue
        r = stats.spearmanr(x, y)
        rows.append({"descriptor": col, "spearman": float(r.statistic),
                     "p_value": float(r.pvalue)})
    res = pd.DataFrame(rows)
    rej, padj = benjamini_hochberg(res["p_value"].to_numpy())
    res["p_adjusted_bh"] = padj
    res["significant_fdr05"] = rej
    res["abs_spearman"] = res["spearman"].abs()
    res = res.sort_values("abs_spearman", ascending=False).reset_index(drop=True)
    res.to_csv(config.TABLES_DIR / f"univariate_{key}.csv", index=False)

    top = res.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.4, 0.34 * top_n + 1.6))
    colours = [P.ORANGE if v > 0 else P.BLUE for v in top["spearman"]]
    ax.barh(range(len(top)), top["spearman"], color=colours, height=0.7)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["descriptor"], fontsize=7.5)
    ax.axvline(0, color=P.INK_SECONDARY, linewidth=1.0)
    for i, (v, sig) in enumerate(zip(top["spearman"], top["significant_fdr05"])):
        ax.text(v + (0.012 if v > 0 else -0.012), i,
                f"{v:+.2f}{'*' if sig else ''}", va="center",
                ha="left" if v > 0 else "right", fontsize=7.5,
                color=P.INK_SECONDARY)
    ax.margins(x=0.18)
    ax.set_xlabel("Spearman rho with measured value")
    ax.set_title(f"Strongest univariate descriptor associations ({key})\n"
                 f"orange = higher descriptor, higher measurement; "
                 f"* = significant at FDR 5%", fontsize=9.5)
    ax.grid(axis="y", visible=False)
    P.despine(ax)
    p = P.save(fig, f"eda_{key}_univariate")
    return {"figure": p, "n_significant_fdr05": int(res["significant_fdr05"].sum()),
            "top": res.head(15).to_dict("records")}


def main() -> int:  # pragma: no cover
    config.ensure_dirs()
    P.apply_style()
    out: dict = {}
    for key in config.ENDPOINTS:
        print(f"\n[eda] === {key} ===")
        out[key] = {
            "target_distributions": fig_target_distributions(key),
            "sequence_properties": fig_sequence_properties(key),
            "identity_applicability": fig_identity_and_applicability(key),
            "univariate": fig_univariate_associations(key),
        }
    out["endpoint_correlations"] = fig_endpoint_correlations()
    (config.RESULTS_DIR / "eda_summary.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("\n[eda] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
