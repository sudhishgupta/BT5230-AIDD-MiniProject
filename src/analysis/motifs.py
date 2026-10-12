"""
Sequence determinants of poor developability.

Four complementary views, because any single one is easy to fool on 137
antibodies:

1. **Chemical liability motifs.**  Do the canonical developability liabilities
   (N-glycosylation sequons, NG/NS deamidation, DG/DS isomerisation, exposed
   Met/Trp, free cysteine) actually associate with the measured endpoint?
2. **Composition.**  Which amino acids, in which CDR, track the measurement?
3. **IMGT position-wise enrichment.**  With every chain aligned to a common
   IMGT numbering, which specific positions differ between the risky and safe
   tails?  This is the only view that can name a position rather than a
   property.
4. **Model attribution.**  What does the fitted model actually lean on?

Everything is corrected for multiple testing with Benjamini-Hochberg, and the
report distinguishes "survives FDR" from "nominally significant".  With a few
hundred tests on 137 antibodies, uncorrected p-values are not evidence.
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats

from src import config
from src.analysis import plotting as P
from src.features import numbering as nb
from src.models.metrics import benjamini_hochberg

# IMGT region boundaries (position codes are strings, possibly with an
# insertion letter such as "111A").
IMGT_CDR_RANGES = {"cdr1": (27, 38), "cdr2": (56, 65), "cdr3": (105, 117)}


def _imgt_num(code: str) -> int:
    digits = "".join(ch for ch in str(code) if ch.isdigit())
    return int(digits) if digits else -1


def _region_of(code: str) -> str:
    n = _imgt_num(code)
    for name, (lo, hi) in IMGT_CDR_RANGES.items():
        if lo <= n <= hi:
            return name
    return "framework"


def _load(key: str):
    ds = pd.read_csv(config.PROCESSED_DIR / f"dataset_{key}.csv")
    ann = pd.read_csv(config.PROCESSED_DIR / f"annotations_{key}.csv",
                      keep_default_na=False, na_values=["__NA__"])
    ann = ann.set_index("antibody_id").reindex(ds["antibody_id"]).reset_index()
    is_train = (ds["role"] == "train").to_numpy()
    return ds.loc[is_train].reset_index(drop=True), ann.loc[is_train].reset_index(drop=True)


# --------------------------------------------------------------------------
# 1. Chemical liability motifs
# --------------------------------------------------------------------------

def liability_associations(key: str) -> pd.DataFrame:
    ds, ann = _load(key)
    y = ds["value"].to_numpy(dtype=float)
    risk = ds["risk_label"].to_numpy(dtype=int)

    rows = []

    # (a) counts from our own regex scan, per chain and summed over the Fv
    regex_cols = [c for c in ann.columns if "_regex_" in c]
    frames: dict[str, np.ndarray] = {}
    for c in regex_cols:
        frames[c] = ann[c].to_numpy(dtype=float)
    for motif in {c.split("_regex_")[1] for c in regex_cols}:
        h = ann.get(f"H_regex_{motif}")
        l = ann.get(f"L_regex_{motif}")
        if h is None:
            continue
        frames[f"Fv_regex_{motif}"] = (
            h.to_numpy(dtype=float) + (l.to_numpy(dtype=float) if l is not None else 0.0)
        )

    # (b) AntPack's curated, severity-annotated hits, split by CDR location.
    #
    # Collected in TWO passes on purpose.  Building the per-type lists lazily
    # in a single pass desynchronises them: a liability class first seen at
    # antibody 40 starts its list at length 0 while the classes seen from
    # antibody 0 are already at length 40, so every list ends up a different
    # length and the length check below silently discards most classes.  The
    # first pass therefore establishes the full vocabulary, and the second
    # fills a dense row for every antibody.
    for chain in ("H", "L"):
        col = f"{chain}_liabilities"
        if col not in ann.columns:
            continue

        parsed: list[tuple[Counter, Counter]] = []
        vocabulary: set[str] = set()
        for raw in ann[col].fillna("[]"):
            try:
                hits = json.loads(raw) if isinstance(raw, str) else []
            except Exception:
                hits = []
            ca, cc = Counter(), Counter()
            for h in hits:
                if "description" not in h:
                    continue
                kind = h["description"].split("(")[0].strip()
                ca[kind] += 1
                if h.get("in_cdr"):
                    cc[kind] += 1
            parsed.append((ca, cc))
            vocabulary.update(ca)
            vocabulary.update(cc)

        for kind in sorted(vocabulary):
            safe = kind.replace(" ", "_").replace("/", "_")
            frames[f"{chain}_antpack_{safe}"] = np.array(
                [ca.get(kind, 0) for ca, _ in parsed], dtype=float)
            frames[f"{chain}_antpackCDR_{safe}"] = np.array(
                [cc.get(kind, 0) for _, cc in parsed], dtype=float)

    for name, x in frames.items():
        if np.std(x) == 0:
            continue
        sp = stats.spearmanr(x, y)
        pos, neg = x[risk == 1], x[risk == 0]
        try:
            mw = stats.mannwhitneyu(pos, neg, alternative="two-sided")
            mw_p = float(mw.pvalue)
        except ValueError:
            mw_p = float("nan")
        rows.append({
            "feature": name,
            "mean_all": float(x.mean()),
            "mean_risk_positive": float(pos.mean()) if len(pos) else np.nan,
            "mean_risk_negative": float(neg.mean()) if len(neg) else np.nan,
            "spearman": float(sp.statistic),
            "spearman_p": float(sp.pvalue),
            "mannwhitney_p": mw_p,
        })

    res = pd.DataFrame(rows)
    if len(res):
        _, padj = benjamini_hochberg(res["spearman_p"].to_numpy())
        res["spearman_p_bh"] = padj
        res["significant_fdr05"] = res["spearman_p_bh"] <= 0.05
        res = res.reindex(res["spearman"].abs().sort_values(ascending=False).index)
    res.to_csv(config.TABLES_DIR / f"motifs_liabilities_{key}.csv", index=False)
    return res.reset_index(drop=True)


# --------------------------------------------------------------------------
# 2. Composition within each CDR
# --------------------------------------------------------------------------

def composition_associations(key: str) -> pd.DataFrame:
    ds, ann = _load(key)
    y = ds["value"].to_numpy(dtype=float)

    rows = []
    for chain in ("H", "L"):
        for reg in nb.CDR_REGIONS:
            seqs = ann[f"{chain}_{reg}"].fillna("").astype(str).tolist()
            lens = np.array([max(len(s), 1) for s in seqs], dtype=float)
            for aa in config.STANDARD_AA:
                counts = np.array([s.count(aa) for s in seqs], dtype=float)
                if counts.sum() < 5:
                    continue
                frac = counts / lens
                if np.std(frac) == 0:
                    continue
                sp = stats.spearmanr(frac, y)
                rows.append({
                    "chain": chain, "region": reg, "amino_acid": aa,
                    "total_occurrences": float(counts.sum()),
                    "spearman": float(sp.statistic), "p_value": float(sp.pvalue),
                })
    res = pd.DataFrame(rows)
    if len(res):
        _, padj = benjamini_hochberg(res["p_value"].to_numpy())
        res["p_bh"] = padj
        res["significant_fdr05"] = res["p_bh"] <= 0.05
        res = res.reindex(res["spearman"].abs().sort_values(ascending=False).index)
    res.to_csv(config.TABLES_DIR / f"motifs_composition_{key}.csv", index=False)
    return res.reset_index(drop=True)


# --------------------------------------------------------------------------
# 3. IMGT position-wise enrichment
# --------------------------------------------------------------------------

def _build_msa(seqs: list[str], chain_set: list[str]):
    from antpack import SingleChainAnnotator
    ann = SingleChainAnnotator(chains=chain_set, scheme="imgt")
    alns = [ann.analyze_seq(s) for s in seqs]
    positions, aligned = ann.build_msa(seqs, alns)
    return list(positions), list(aligned)


def positional_enrichment(key: str, chain: str = "H") -> pd.DataFrame:
    ds, ann = _load(key)
    y = ds["value"].to_numpy(dtype=float)
    col = "heavy_chain" if chain == "H" else "light_chain"
    seqs = ds[col].fillna("").astype(str).tolist()
    if not all(seqs):
        return pd.DataFrame()

    chain_set = ["H"] if chain == "H" else ["K", "L"]
    positions, aligned = _build_msa(seqs, chain_set)
    mat = np.array([list(a) for a in aligned])

    rows = []
    for j, code in enumerate(positions):
        column = mat[:, j]
        region = _region_of(code)
        for aa in sorted(set(column) - {"-"}):
            present = (column == aa).astype(int)
            n = int(present.sum())
            if n < 8 or n > len(present) - 8:
                continue   # too rare or too universal to test meaningfully
            with_aa, without = y[present == 1], y[present == 0]
            mw = stats.mannwhitneyu(with_aa, without, alternative="two-sided")
            rows.append({
                "chain": chain, "imgt_position": code, "region": region,
                "amino_acid": aa, "n_with": n, "n_without": int(len(present) - n),
                "mean_with": float(with_aa.mean()),
                "mean_without": float(without.mean()),
                "delta": float(with_aa.mean() - without.mean()),
                "p_value": float(mw.pvalue),
            })

    res = pd.DataFrame(rows)
    if len(res):
        _, padj = benjamini_hochberg(res["p_value"].to_numpy())
        res["p_bh"] = padj
        res["significant_fdr05"] = res["p_bh"] <= 0.05
        res = res.sort_values("p_value").reset_index(drop=True)
    res.to_csv(config.TABLES_DIR / f"motifs_positional_{chain}_{key}.csv", index=False)
    return res


# --------------------------------------------------------------------------
# 4. Model attribution
# --------------------------------------------------------------------------

def model_attribution(key: str, max_display: int = 22) -> dict:
    """
    Global feature attribution for an interpretable descriptor model.

    We deliberately attribute a DESCRIPTOR model rather than the best overall
    model: attributing 480 anonymous language-model dimensions would produce a
    ranking nobody can act on, whereas "CDR aromatic fraction" is a statement a
    protein engineer can test.  If the language model wins on accuracy, the
    report says so and uses the descriptor model only to explain the signal.
    """
    import matplotlib.pyplot as plt
    from sklearn.model_selection import GroupKFold
    from src.models import zoo as Z

    ds_all = pd.read_csv(config.PROCESSED_DIR / f"dataset_{key}.csv")
    npz = np.load(config.PROCESSED_DIR / f"features_{key}.npz", allow_pickle=True)
    splits = json.loads((config.PROCESSED_DIR / f"splits_{key}.json").read_text(encoding="utf-8"))
    is_train = (ds_all["role"] == "train").to_numpy()

    X = npz["X__desc_all"][is_train]
    cols = [str(c) for c in npz["cols__desc_all"]]
    y = ds_all.loc[is_train, "value"].to_numpy(dtype=float)
    groups = np.asarray(splits["cluster_labels"])

    est, grid = Z.regression_zoo()["random_forest"]
    from sklearn.model_selection import GridSearchCV
    search = GridSearchCV(est, grid, scoring="neg_mean_absolute_error",
                          cv=GroupKFold(n_splits=4), n_jobs=-1, refit=True)
    search.fit(X, y, groups=groups)
    model = search.best_estimator_

    out: dict = {"model": "random_forest", "feature_set": "desc_all",
                 "best_params": {str(k): str(v) for k, v in search.best_params_.items()}}

    # SHAP where available; permutation importance as the documented fallback.
    importance, method = None, None
    try:
        import shap
        Xs = model.named_steps["var"].transform(X)
        Xs = model.named_steps["scale"].transform(Xs)
        kept = model.named_steps["var"].get_support()
        kept_cols = [c for c, k in zip(cols, kept) if k]
        explainer = shap.TreeExplainer(model.named_steps["model"])
        sv = explainer.shap_values(Xs, check_additivity=False)
        importance = np.abs(sv).mean(axis=0)
        signed = np.array([
            stats.spearmanr(Xs[:, i], sv[:, i]).statistic for i in range(Xs.shape[1])
        ])
        method = "shap_tree"
        names = kept_cols
    except Exception as exc:
        print(f"    [motifs] SHAP unavailable ({type(exc).__name__}: {exc}); "
              "using permutation importance")
        from sklearn.inspection import permutation_importance
        pi = permutation_importance(model, X, y, n_repeats=20,
                                    random_state=config.SEED, n_jobs=-1,
                                    scoring="neg_mean_absolute_error")
        importance = pi.importances_mean
        signed = np.array([stats.spearmanr(X[:, i], y).statistic
                           for i in range(X.shape[1])])
        method = "permutation"
        names = cols

    out["attribution_method"] = method
    imp = pd.DataFrame({"feature": names, "importance": importance,
                        "direction": np.sign(signed), "signed_strength": signed})
    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
    imp.to_csv(config.TABLES_DIR / f"attribution_{key}.csv", index=False)

    top = imp.head(max_display).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.6, 0.33 * max_display + 1.6))
    colours = [P.ORANGE if d > 0 else P.BLUE for d in top["direction"]]
    ax.barh(range(len(top)), top["importance"], color=colours, height=0.7)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"], fontsize=7.5)
    ax.set_xlabel(f"mean |{'SHAP' if method == 'shap_tree' else 'permutation'}| importance")
    ep = config.ENDPOINTS[key]
    ax.set_title(
        f"What the descriptor model uses to predict {ep.pretty_name}\n"
        f"orange = raises the measurement, blue = lowers it", fontsize=9.5)
    ax.grid(axis="y", visible=False)
    P.despine(ax)
    out["figure"] = P.save(fig, f"attribution_{key}")
    out["top_features"] = imp.head(20).to_dict("records")
    return out


# --------------------------------------------------------------------------
# 5. Figures for the motif analyses
# --------------------------------------------------------------------------

def fig_liability_and_composition(key: str, liab: pd.DataFrame,
                                  comp: pd.DataFrame) -> list[str]:
    import matplotlib.pyplot as plt
    ep = config.ENDPOINTS[key]
    paths = []

    if len(liab):
        top = liab.head(16).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8.6, 0.35 * len(top) + 1.6))
        colours = [P.ORANGE if v > 0 else P.BLUE for v in top["spearman"]]
        ax.barh(range(len(top)), top["spearman"], color=colours, height=0.7)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top["feature"], fontsize=7.5)
        ax.axvline(0, color=P.INK_SECONDARY, linewidth=1.0)
        for i, (v, s) in enumerate(zip(top["spearman"], top["significant_fdr05"])):
            ax.text(v + (0.012 if v > 0 else -0.012), i, f"{v:+.2f}{'*' if s else ''}",
                    va="center", ha="left" if v > 0 else "right", fontsize=7.5,
                    color=P.INK_SECONDARY)
        ax.margins(x=0.18)
        ax.set_xlabel(f"Spearman rho with {ep.pretty_name}")
        ax.set_title(f"Chemical liability motifs vs {ep.pretty_name}\n"
                     f"* survives Benjamini-Hochberg FDR 5%", fontsize=9.5)
        ax.grid(axis="y", visible=False)
        P.despine(ax)
        paths.append(P.save(fig, f"motifs_{key}_liabilities"))

    if len(comp):
        pivot = comp.pivot_table(index="amino_acid",
                                 columns=["chain", "region"],
                                 values="spearman", aggfunc="first")
        pivot = pivot.reindex(list(config.STANDARD_AA))
        fig, ax = plt.subplots(figsize=(8.0, 6.2))
        vmax = float(np.nanmax(np.abs(pivot.to_numpy(dtype=float))))
        im = ax.imshow(pivot.to_numpy(dtype=float), cmap=P.DIVERGING,
                       vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels([f"{c}-{r.upper()}" for c, r in pivot.columns],
                           rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(pivot.shape[0]))
        ax.set_yticklabels(pivot.index, fontsize=8)
        sig = comp.set_index(["amino_acid", "chain", "region"])["significant_fdr05"]
        for i, aa in enumerate(pivot.index):
            for j, (c, r) in enumerate(pivot.columns):
                v = pivot.iloc[i, j]
                if np.isfinite(v) and sig.get((aa, c, r), False):
                    ax.text(j, i, "*", ha="center", va="center", fontsize=11,
                            color="white" if abs(v) > vmax * 0.55 else P.INK)
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, shrink=0.8)
        cb.set_label(f"Spearman rho with {ep.pretty_name}", fontsize=8.5,
                     color=P.INK_SECONDARY)
        ax.set_title(f"Amino-acid composition of each CDR vs {ep.pretty_name}\n"
                     f"* survives FDR 5%", fontsize=10)
        paths.append(P.save(fig, f"motifs_{key}_composition"))

    return paths


def fig_sequence_logo(key: str, chain: str = "H") -> str | None:
    """Difference logo: CDR composition of the risky tail minus the safe tail."""
    import matplotlib.pyplot as plt
    try:
        import logomaker
    except ImportError:
        return None

    ds, _ = _load(key)
    ep = config.ENDPOINTS[key]
    col = "heavy_chain" if chain == "H" else "light_chain"
    seqs = ds[col].fillna("").astype(str).tolist()
    if not all(seqs):
        return None
    y = ds["value"].to_numpy(dtype=float)

    positions, aligned = _build_msa(seqs, ["H"] if chain == "H" else ["K", "L"])
    mat = np.array([list(a) for a in aligned])
    cdr_idx = [j for j, c in enumerate(positions) if _region_of(c) != "framework"]
    if not cdr_idx:
        return None

    worst = y >= np.quantile(y, 0.75)
    best = y <= np.quantile(y, 0.25)
    if ep.favourable_direction == "higher":
        worst, best = best, worst

    def freq(rows) -> pd.DataFrame:
        out = np.zeros((len(cdr_idx), len(config.STANDARD_AA)))
        for ri, j in enumerate(cdr_idx):
            colv = mat[rows, j]
            colv = colv[colv != "-"]
            if len(colv) == 0:
                continue
            for ai, aa in enumerate(config.STANDARD_AA):
                out[ri, ai] = np.mean(colv == aa)
        return pd.DataFrame(out, columns=list(config.STANDARD_AA))

    diff = freq(worst) - freq(best)
    diff.index = range(len(cdr_idx))

    fig, ax = plt.subplots(figsize=(max(10, len(cdr_idx) * 0.22), 3.4))
    logomaker.Logo(diff, ax=ax, color_scheme="chemistry", flip_below=True)
    ax.set_xticks(range(0, len(cdr_idx), max(1, len(cdr_idx) // 28)))
    ax.set_xticklabels(
        [positions[cdr_idx[i]] for i in range(0, len(cdr_idx), max(1, len(cdr_idx) // 28))],
        rotation=90, fontsize=6.5)
    ax.set_xlabel(f"IMGT position ({chain} chain, CDR positions only)")
    ax.set_ylabel("frequency difference")
    ax.set_title(
        f"Residues enriched in the worst quartile of {ep.pretty_name}\n"
        f"upward = more common in the risky tail, downward = more common in the safe tail",
        fontsize=9.5)
    ax.grid(False)
    P.despine(ax)
    return P.save(fig, f"motifs_{key}_logo_{chain}")


def main() -> int:  # pragma: no cover
    config.ensure_dirs()
    P.apply_style()
    summary: dict = {}
    for key in config.ENDPOINTS:
        print(f"\n[motifs] === {key} ===")
        liab = liability_associations(key)
        comp = composition_associations(key)
        pos_h = positional_enrichment(key, "H")
        pos_l = positional_enrichment(key, "L")
        figs = fig_liability_and_composition(key, liab, comp)
        logo = fig_sequence_logo(key, "H")

        attribution = {}
        if (config.PROCESSED_DIR / f"features_{key}.npz").exists():
            attribution = model_attribution(key)

        summary[key] = {
            "n_liability_features": int(len(liab)),
            "n_liability_fdr05": int(liab["significant_fdr05"].sum()) if len(liab) else 0,
            "top_liabilities": liab.head(8).to_dict("records") if len(liab) else [],
            "n_composition_fdr05": int(comp["significant_fdr05"].sum()) if len(comp) else 0,
            "top_composition": comp.head(10).to_dict("records") if len(comp) else [],
            "n_positional_tests_H": int(len(pos_h)),
            "n_positional_fdr05_H": int(pos_h["significant_fdr05"].sum()) if len(pos_h) else 0,
            "top_positional_H": pos_h.head(10).to_dict("records") if len(pos_h) else [],
            "n_positional_fdr05_L": int(pos_l["significant_fdr05"].sum()) if len(pos_l) else 0,
            "top_positional_L": pos_l.head(10).to_dict("records") if len(pos_l) else [],
            "figures": figs + ([logo] if logo else []),
            "attribution": attribution,
        }
        print(f"  liabilities: {summary[key]['n_liability_fdr05']}/{len(liab)} pass FDR")
        print(f"  composition: {summary[key]['n_composition_fdr05']}/{len(comp)} pass FDR")
        print(f"  positional H: {summary[key]['n_positional_fdr05_H']}/{len(pos_h)} pass FDR")

    (config.RESULTS_DIR / "motif_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
