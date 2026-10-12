"""
Assemble the feature matrices that the model comparison consumes.

Each named feature set maps to a matrix with one row per antibody, aligned to
the curated dataset's ``antibody_id`` order.  All of them are deterministic
functions of sequence alone -- no target information is used anywhere -- so
they are safe to compute once, before any split is drawn.

Feature sets
------------
baseline_trivial    4 columns: chain lengths and mean hydropathy.  The floor.
desc_global         whole-chain physicochemical descriptors + liabilities
desc_cdr            CDR-restricted descriptors + TAP-style aggregates
desc_all            desc_global + desc_cdr
esm2_35m            ESM-2 35M mean-pooled, heavy and light concatenated (960d)
ablang2             AbLang2 paired seqcoding (480d)
desc_all+ablang2    late fusion of the interpretable and learned views
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.features import descriptors as desc
from src.features import plm


def _load(key: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ds = pd.read_csv(config.PROCESSED_DIR / f"dataset_{key}.csv")
    ann = pd.read_csv(config.PROCESSED_DIR / f"annotations_{key}.csv",
                      keep_default_na=False, na_values=["__NA__"])
    dsc = pd.read_csv(config.PROCESSED_DIR / f"descriptors_{key}.csv")
    return ds, ann, dsc


def _align(df: pd.DataFrame, ids: pd.Series) -> pd.DataFrame:
    """Reindex a feature frame onto the dataset's antibody_id order."""
    indexed = df.set_index("antibody_id")
    # Check membership directly. Testing for an all-NaN row AFTER reset_index
    # can never fire, because reset_index puts the (non-null) antibody_id back
    # as a column -- so a genuinely missing row would sail through as a row of
    # NaN features and be written into the npz.
    missing = [a for a in ids if a not in indexed.index]
    if missing:
        raise ValueError(
            f"{len(missing)} antibodies have no feature row, e.g. {missing[:5]}. "
            "Re-run the descriptor/annotation steps before building features."
        )
    return indexed.reindex(ids).reset_index()


def _split_descriptor_blocks(dsc: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Partition descriptor columns into the global and CDR blocks."""
    cdr_prefixes = tuple(
        f"{c}_{r}_" for c in ("H", "L") for r in ("cdr1", "cdr2", "cdr3")
    ) + ("CDRall_",)
    cdr_cols, global_cols = [], []
    for c in dsc.columns:
        if c == "antibody_id":
            continue
        (cdr_cols if c.startswith(cdr_prefixes) else global_cols).append(c)
    return global_cols, cdr_cols


def build_feature_sets(key: str, device: str | None = None
                       ) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, list[str]]]:
    """
    Returns (dataset, {feature_set_name: matrix}, {feature_set_name: column names}).
    """
    ds, ann, dsc = _load(key)
    ids = ds["antibody_id"]
    dsc = _align(dsc, ids)
    global_cols, cdr_cols = _split_descriptor_blocks(dsc)

    trivial = _align(desc.build_trivial_baseline(ds), ids)
    trivial_cols = [c for c in trivial.columns if c != "antibody_id"]

    seqs = [
        (h, (l if isinstance(l, str) else ""))
        for h, l in zip(ds["heavy_chain"], ds["light_chain"])
    ]

    matrices: dict[str, np.ndarray] = {}
    names: dict[str, list[str]] = {}

    matrices["baseline_trivial"] = trivial[trivial_cols].to_numpy(dtype=np.float64)
    names["baseline_trivial"] = trivial_cols

    matrices["desc_global"] = dsc[global_cols].to_numpy(dtype=np.float64)
    names["desc_global"] = global_cols

    matrices["desc_cdr"] = dsc[cdr_cols].to_numpy(dtype=np.float64)
    names["desc_cdr"] = cdr_cols

    all_cols = global_cols + cdr_cols
    matrices["desc_all"] = dsc[all_cols].to_numpy(dtype=np.float64)
    names["desc_all"] = all_cols

    ablang = plm.embed_ablang2(seqs, device=device)
    matrices["ablang2"] = ablang.astype(np.float64)
    names["ablang2"] = [f"ablang2_{i}" for i in range(ablang.shape[1])]

    esm = plm.embed_esm2(seqs, device=device)
    matrices["esm2_35m"] = esm.astype(np.float64)
    names["esm2_35m"] = [f"esm2_{i}" for i in range(esm.shape[1])]

    matrices["desc_all+ablang2"] = np.concatenate(
        [matrices["desc_all"], matrices["ablang2"]], axis=1
    )
    names["desc_all+ablang2"] = names["desc_all"] + names["ablang2"]

    return ds, matrices, names


def main() -> int:  # pragma: no cover
    config.ensure_dirs()
    for key in config.ENDPOINTS:
        print(f"\n[build] === {key} ===")
        ds, mats, names = build_feature_sets(key)
        for name in config.FEATURE_SETS:
            m = mats[name]
            if not np.all(np.isfinite(m)):
                n_bad = int((~np.isfinite(m)).sum())
                print(f"  WARNING {name}: {n_bad} non-finite values present")
            print(f"  {name:<20} {m.shape[0]:>4} x {m.shape[1]:<5}")
        np.savez_compressed(
            config.PROCESSED_DIR / f"features_{key}.npz",
            antibody_id=ds["antibody_id"].to_numpy(dtype=object),
            **{f"X__{k}": v for k, v in mats.items()},
            **{f"cols__{k}": np.array(v, dtype=object) for k, v in names.items()},
        )
        print(f"  -> features_{key}.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
