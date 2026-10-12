"""
Leakage-safe split generation.

The problem with a random split on antibodies
---------------------------------------------
Every antibody variable domain shares 70-85% framework identity with every
other one, so whole-sequence identity is almost useless for deciding whether
two antibodies are "the same".  What distinguishes them is the CDR loops.  Two
antibodies with near-identical CDRs are effectively one data point, and if one
lands in train and the other in test the reported performance is partly
memorisation.

So we cluster on **CDR identity only** and split by cluster.

Concretely:
  1. Concatenate the six IMGT CDRs of each antibody.
  2. Compute pairwise identity by aligning the CDRs region by region, summing
     matches and dividing by the summed longer-of-the-pair lengths.  Doing it
     per region rather than on the concatenation stops a long CDR-H3 from
     dominating, and stops CDR-L1 of one antibody being aligned against
     CDR-H2 of another.
  3. Single-linkage agglomerative clustering at a chosen identity threshold.
     Single linkage is the right choice here because the property we need is
     transitive: if A is a near-duplicate of B and B of C, all three must stay
     on the same side of the split.
  4. StratifiedGroupKFold over those clusters, stratifying on quantile bins of
     the target so folds stay comparable in difficulty.

The same machinery provides the applicability-domain measure: an external
antibody's maximum CDR identity to anything in the training set.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import StratifiedGroupKFold

from src import config
from src.features import numbering as nb

CDR_FIELDS = [f"{c}_{r}" for c in ("H", "L") for r in nb.CDR_REGIONS]


# --------------------------------------------------------------------------
# Pairwise CDR identity
# --------------------------------------------------------------------------

def _aligner():
    from Bio import Align
    a = Align.PairwiseAligner()
    a.mode = "global"
    a.match_score = 1.0
    a.mismatch_score = 0.0
    a.open_gap_score = -1.0
    a.extend_gap_score = -0.5
    return a


def _identity(a: str, b: str, aligner) -> tuple[float, int]:
    """Return (matches, denominator) for one pair of region sequences."""
    if not a and not b:
        return 0.0, 0
    if not a or not b:
        return 0.0, max(len(a), len(b))
    aln = aligner.align(a, b)[0]
    matches = 0
    for (s1, e1), (s2, e2) in zip(aln.aligned[0], aln.aligned[1]):
        matches += sum(1 for x, y in zip(a[s1:e1], b[s2:e2]) if x == y)
    return float(matches), max(len(a), len(b))


def cdr_identity_matrix(ann: pd.DataFrame, cache_tag: str | None = None,
                        use_cache: bool = True) -> np.ndarray:
    """
    Symmetric (N, N) matrix of CDR identity in [0, 1].

    O(N^2) alignments. With N <= 300 and ~60 residues of CDR per antibody this
    is a few seconds, so no approximation is needed.
    """
    regions = [[str(ann.iloc[i].get(f, "") or "") for f in CDR_FIELDS]
               for i in range(len(ann))]
    n = len(regions)

    cache = None
    if use_cache and cache_tag:
        h = hashlib.sha256(
            json.dumps(regions, sort_keys=True).encode()
        ).hexdigest()[:16]
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache = config.CACHE_DIR / f"cdr_identity_{cache_tag}_{h}.npy"
        if cache.exists():
            print(f"[splits] CDR identity cache hit: {cache.name}")
            return np.load(cache)

    aligner = _aligner()
    ident = np.eye(n, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            tot_m = tot_d = 0.0
            for ri in range(len(CDR_FIELDS)):
                m, d = _identity(regions[i][ri], regions[j][ri], aligner)
                tot_m += m
                tot_d += d
            val = float(tot_m / tot_d) if tot_d else 0.0
            ident[i, j] = ident[j, i] = val
    if cache is not None:
        np.save(cache, ident)
    return ident


# --------------------------------------------------------------------------
# Clustering
# --------------------------------------------------------------------------

def cluster_by_identity(identity: np.ndarray, threshold: float) -> np.ndarray:
    """Single-linkage clusters: antibodies linked above ``threshold`` merge."""
    distance = 1.0 - identity
    np.fill_diagonal(distance, 0.0)
    distance = np.clip(distance, 0.0, None)
    model = AgglomerativeClustering(
        n_clusters=None, metric="precomputed", linkage="single",
        distance_threshold=1.0 - threshold,
    )
    return model.fit_predict(distance)


def summarise_clusters(labels: np.ndarray) -> dict:
    _, counts = np.unique(labels, return_counts=True)
    return {
        "n_clusters": int(len(counts)),
        "n_items": int(labels.size),
        "largest_cluster": int(counts.max()),
        "n_singletons": int((counts == 1).sum()),
        "mean_cluster_size": float(counts.mean()),
    }


# --------------------------------------------------------------------------
# Fold generation
# --------------------------------------------------------------------------

def quantile_bins(y: np.ndarray, n_bins: int = 4) -> np.ndarray:
    """Stratification labels from quantile bins of a continuous target."""
    edges = np.quantile(y, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.digitize(y, edges)


def make_cv_folds(y: np.ndarray, groups: np.ndarray, n_splits: int,
                  n_repeats: int, seed: int) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    """
    Repeated grouped, target-stratified folds.

    Returns a list of repeats; each repeat is a list of (train_idx, test_idx).
    Stratification is on quantile bins of y so that no fold is accidentally
    made of only fast- or only slow-eluting antibodies -- with n=137 that
    happens easily by chance and makes fold-level metrics incomparable.
    """
    strata = quantile_bins(y, n_bins=4)
    repeats = []
    for r in range(n_repeats):
        rng = np.random.default_rng(seed + r)
        # Relabel groups in a random order so the group->fold assignment
        # changes between repeats (StratifiedGroupKFold is deterministic
        # given the group labels).
        uniq = np.unique(groups)
        perm = rng.permutation(len(uniq))
        remap = {g: perm[i] for i, g in enumerate(uniq)}
        shuffled = np.array([remap[g] for g in groups])

        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed + r
        )
        folds = [(tr, te) for tr, te in splitter.split(np.zeros(len(y)), strata, shuffled)]
        repeats.append(folds)
    return repeats


def make_random_folds(y: np.ndarray, n_splits: int, n_repeats: int,
                      seed: int) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    """Ungrouped stratified folds, for the 'does clustering matter?' ablation."""
    from sklearn.model_selection import StratifiedKFold
    strata = quantile_bins(y, n_bins=4)
    repeats = []
    for r in range(n_repeats):
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                   random_state=seed + r)
        repeats.append(list(splitter.split(np.zeros(len(y)), strata)))
    return repeats


# --------------------------------------------------------------------------
# Applicability domain
# --------------------------------------------------------------------------

def nearest_train_identity(ann_query: pd.DataFrame,
                           ann_train: pd.DataFrame) -> np.ndarray:
    """
    For each query antibody, its maximum CDR identity to any training antibody.

    This is the applicability-domain coordinate: a prediction for an antibody
    whose closest training neighbour shares 30% of its CDRs is an
    extrapolation, and the report says so instead of quoting one pooled error.
    """
    aligner = _aligner()
    q = [[str(ann_query.iloc[i].get(f, "") or "") for f in CDR_FIELDS]
         for i in range(len(ann_query))]
    t = [[str(ann_train.iloc[i].get(f, "") or "") for f in CDR_FIELDS]
         for i in range(len(ann_train))]

    out = np.zeros(len(q), dtype=np.float32)
    for i in range(len(q)):
        best = 0.0
        for j in range(len(t)):
            tot_m = tot_d = 0.0
            for ri in range(len(CDR_FIELDS)):
                m, d = _identity(q[i][ri], t[j][ri], aligner)
                tot_m += m
                tot_d += d
            if tot_d:
                best = max(best, tot_m / tot_d)
        out[i] = best
    return out


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def build_splits_for_endpoint(key: str) -> dict:
    """Generate and persist the split definition for one endpoint."""
    ep = config.ENDPOINTS[key]
    ds = pd.read_csv(config.PROCESSED_DIR / f"dataset_{key}.csv")
    ann = pd.read_csv(config.PROCESSED_DIR / f"annotations_{key}.csv",
                      keep_default_na=False, na_values=["__NA__"])
    ann = ann.set_index("antibody_id").reindex(ds["antibody_id"]).reset_index()

    is_train = ds["role"].to_numpy() == "train"
    train_ds = ds.loc[is_train].reset_index(drop=True)
    train_ann = ann.loc[is_train].reset_index(drop=True)
    y = train_ds["value"].to_numpy(dtype=float)

    ident = cdr_identity_matrix(train_ann, cache_tag=f"{key}_train")

    report: dict = {"endpoint": key, "n_train": int(len(train_ds)),
                    "clustering": {}, "identity_summary": {}}

    off_diag = ident[~np.eye(len(ident), dtype=bool)]
    report["identity_summary"] = {
        "mean_pairwise_cdr_identity": float(off_diag.mean()),
        "max_pairwise_cdr_identity": float(off_diag.max()),
        "n_pairs_above_0.8": int((off_diag > 0.8).sum() // 2),
        "n_pairs_above_0.9": int((off_diag > 0.9).sum() // 2),
    }

    chosen_labels = None
    for thr in config.CDR_IDENTITY_SENSITIVITY:
        labels = cluster_by_identity(ident, thr)
        report["clustering"][f"threshold_{thr}"] = summarise_clusters(labels)
        if abs(thr - config.CDR_IDENTITY_THRESHOLD) < 1e-9:
            chosen_labels = labels

    assert chosen_labels is not None
    folds = make_cv_folds(y, chosen_labels, config.CV_N_SPLITS,
                          config.CV_N_REPEATS, config.SEED)
    rand_folds = make_random_folds(y, config.CV_N_SPLITS,
                                   config.CV_N_REPEATS, config.SEED)

    # Applicability domain for each external cohort.  Computed BEFORE the
    # split payload is serialised: json.dumps snapshots the dict, so adding
    # keys to `report` afterwards would leave them out of splits_<key>.json
    # and any consumer reading that file would see an empty table.
    ad = {}
    for cohort in ep.externals:
        mask = (ds["cohort"] == cohort.key).to_numpy()
        if not mask.any():
            continue
        ext_ann = ann.loc[mask].reset_index(drop=True)
        sim = nearest_train_identity(ext_ann, train_ann)
        ad[cohort.key] = {
            "n": int(mask.sum()),
            "max_identity_to_train": {
                "min": float(sim.min()), "median": float(np.median(sim)),
                "max": float(sim.max()), "mean": float(sim.mean()),
            },
            "per_antibody": dict(zip(ds.loc[mask, "antibody_id"],
                                     sim.astype(float).round(4).tolist())),
        }
    (config.PROCESSED_DIR / f"applicability_{key}.json").write_text(
        json.dumps(ad, indent=2), encoding="utf-8")

    report["applicability_domain"] = {
        k: v["max_identity_to_train"] for k, v in ad.items()
    }

    payload = {
        "endpoint": key,
        "train_ids": train_ds["antibody_id"].tolist(),
        "cluster_labels": chosen_labels.tolist(),
        "cluster_threshold": config.CDR_IDENTITY_THRESHOLD,
        "clustered_folds": [[[tr.tolist(), te.tolist()] for tr, te in rep]
                            for rep in folds],
        "random_folds": [[[tr.tolist(), te.tolist()] for tr, te in rep]
                         for rep in rand_folds],
        "report": report,
    }
    out = config.PROCESSED_DIR / f"splits_{key}.json"
    out.write_text(json.dumps(payload), encoding="utf-8")

    return report


def main() -> int:  # pragma: no cover
    config.ensure_dirs()
    all_reports = {}
    for key in config.ENDPOINTS:
        print(f"\n[splits] === {key} ===")
        rep = build_splits_for_endpoint(key)
        all_reports[key] = rep
        print(f"  n_train = {rep['n_train']}")
        print(f"  mean pairwise CDR identity = "
              f"{rep['identity_summary']['mean_pairwise_cdr_identity']:.3f}")
        print(f"  pairs >0.8 identity = {rep['identity_summary']['n_pairs_above_0.8']}")
        for thr, s in rep["clustering"].items():
            print(f"  {thr}: {s['n_clusters']} clusters, "
                  f"largest={s['largest_cluster']}, singletons={s['n_singletons']}")
        for cohort, s in rep.get("applicability_domain", {}).items():
            print(f"  AD {cohort}: max-identity-to-train "
                  f"median={s['median']:.3f} range=[{s['min']:.3f}, {s['max']:.3f}]")
    (config.PROCESSED_DIR / "splits_report.json").write_text(
        json.dumps(all_reports, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
