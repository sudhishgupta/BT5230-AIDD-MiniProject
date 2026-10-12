"""
Interpretable sequence descriptors for antibody variable domains.

These are the "classical" arm of the feature comparison: every number here is
a deterministic, explainable function of the amino-acid sequence, so when the
model leans on one we can say what it means biophysically.

Descriptor families
-------------------
composition   fraction of each amino acid, plus grouped fractions
hydropathy    Kyte-Doolittle, Eisenberg and Black-Mould means, plus the
              maximum windowed hydrophobicity -- a sequence-only proxy for the
              solvent-exposed hydrophobic patch that actually drives HIC
              retention (the real quantity is structural; this is its
              one-dimensional shadow)
charge        net charge at pH 5.5 and pH 7.4 and theoretical pI.  Those two
              pH values are the ones formulation science cares about: ~5.5 is
              a typical high-concentration formulation buffer and 7.4 is
              physiological
stability     Biopython's instability index, aromaticity, molecular weight
liability     counts of chemical liability motifs
pairing       TAP-style whole-Fv terms: total CDR length, CDR charge, and a
              charge-symmetry term (product of VH and VL net charge), which is
              the sequence analogue of TAP's SFvCSP

Nothing here is fitted to data, so computing descriptors before splitting
introduces no leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.features import numbering as nb

AA = config.STANDARD_AA

# --------------------------------------------------------------------------
# Scales
# --------------------------------------------------------------------------

KYTE_DOOLITTLE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

EISENBERG = {
    "A": 0.62, "R": -2.53, "N": -0.78, "D": -0.90, "C": 0.29, "Q": -0.85,
    "E": -0.74, "G": 0.48, "H": -0.40, "I": 1.38, "L": 1.06, "K": -1.50,
    "M": 0.64, "F": 1.19, "P": 0.12, "S": -0.18, "T": -0.05, "W": 0.81,
    "Y": 0.26, "V": 1.08,
}

# Black & Mould (1991) normalised hydrophobicity -- the scale most often
# quoted in antibody hydrophobicity / HIC work.
BLACK_MOULD = {
    "A": 0.616, "R": 0.000, "N": 0.236, "D": 0.028, "C": 0.680, "Q": 0.251,
    "E": 0.043, "G": 0.501, "H": 0.165, "I": 0.943, "L": 0.943, "K": 0.283,
    "M": 0.738, "F": 1.000, "P": 0.711, "S": 0.359, "T": 0.450, "W": 0.878,
    "Y": 0.880, "V": 0.825,
}

# Side-chain pKa values used for the Henderson-Hasselbalch charge sum.
PKA_SIDE = {"D": 3.65, "E": 4.25, "C": 8.18, "Y": 10.07, "H": 6.00,
            "K": 10.53, "R": 12.48}
PKA_N_TERM, PKA_C_TERM = 9.69, 2.34
POSITIVE = {"K", "R", "H"}

GROUPS = {
    "hydrophobic": set("AILMFVWY"),
    "aromatic": set("FWY"),
    "positive": set("KR"),
    "negative": set("DE"),
    "polar": set("STNQ"),
    "tiny": set("AGS"),
    "proline": set("P"),
    "cysteine": set("C"),
    "glycine": set("G"),
}


# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------

def net_charge(seq: str, ph: float) -> float:
    """Net charge from the Henderson-Hasselbalch equation."""
    if not seq:
        return 0.0
    charge = 1.0 / (1.0 + 10 ** (ph - PKA_N_TERM))        # N-terminus
    charge -= 1.0 / (1.0 + 10 ** (PKA_C_TERM - ph))       # C-terminus
    for aa, pka in PKA_SIDE.items():
        n = seq.count(aa)
        if not n:
            continue
        if aa in POSITIVE:
            charge += n / (1.0 + 10 ** (ph - pka))
        else:
            charge -= n / (1.0 + 10 ** (pka - ph))
    return float(charge)


def isoelectric_point(seq: str) -> float:
    """Theoretical pI by bisection on the net-charge curve."""
    if not seq:
        return float("nan")
    lo, hi = 0.0, 14.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if net_charge(seq, mid) > 0:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2.0, 4)


def _scale_mean(seq: str, scale: dict[str, float]) -> float:
    vals = [scale[a] for a in seq if a in scale]
    return float(np.mean(vals)) if vals else float("nan")


def max_window_hydrophobicity(seq: str, scale: dict[str, float],
                              window: int = 5) -> float:
    """
    Highest mean hydrophobicity over any contiguous window.

    A hydrophobic patch on the folded surface is what HIC actually binds.
    Without a structure we cannot find real patches, so we use the strongest
    local hydrophobic stretch in sequence space as a stand-in.  It is a
    deliberately crude proxy and is labelled as such in the report.
    """
    if len(seq) < window:
        return _scale_mean(seq, scale)
    vals = np.array([scale.get(a, 0.0) for a in seq], dtype=float)
    kernel = np.ones(window) / window
    return float(np.convolve(vals, kernel, mode="valid").max())


def instability_index(seq: str) -> float:
    try:
        from Bio.SeqUtils.ProtParam import ProteinAnalysis
        return float(ProteinAnalysis(seq).instability_index())
    except Exception:
        return float("nan")


def molecular_weight(seq: str) -> float:
    try:
        from Bio.SeqUtils.ProtParam import ProteinAnalysis
        return float(ProteinAnalysis(seq).molecular_weight())
    except Exception:
        return float("nan")


# --------------------------------------------------------------------------
# Descriptor blocks
# --------------------------------------------------------------------------

def region_descriptors(seq: str, prefix: str,
                       full_composition: bool = False) -> dict[str, float]:
    """Descriptors for one sequence region."""
    d: dict[str, float] = {}
    n = len(seq)
    d[f"{prefix}_len"] = float(n)
    if n == 0:
        for g in GROUPS:
            d[f"{prefix}_frac_{g}"] = 0.0
        for s in ("kd", "eis", "bm"):
            d[f"{prefix}_{s}_mean"] = 0.0
            d[f"{prefix}_{s}_maxwin"] = 0.0
        d[f"{prefix}_charge_ph55"] = 0.0
        d[f"{prefix}_charge_ph74"] = 0.0
        d[f"{prefix}_pI"] = 0.0
        if full_composition:
            for a in AA:
                d[f"{prefix}_frac_{a}"] = 0.0
        return d

    for g, members in GROUPS.items():
        d[f"{prefix}_frac_{g}"] = sum(seq.count(a) for a in members) / n

    for tag, scale in (("kd", KYTE_DOOLITTLE), ("eis", EISENBERG),
                       ("bm", BLACK_MOULD)):
        d[f"{prefix}_{tag}_mean"] = _scale_mean(seq, scale)
        d[f"{prefix}_{tag}_maxwin"] = max_window_hydrophobicity(seq, scale, 5)

    d[f"{prefix}_charge_ph55"] = net_charge(seq, 5.5)
    d[f"{prefix}_charge_ph74"] = net_charge(seq, 7.4)
    d[f"{prefix}_pI"] = isoelectric_point(seq)

    if full_composition:
        for a in AA:
            d[f"{prefix}_frac_{a}"] = seq.count(a) / n

    return d


def chain_descriptors(seq: str, prefix: str) -> dict[str, float]:
    """Whole-chain descriptors, including the heavier Biopython terms."""
    d = region_descriptors(seq, prefix, full_composition=True)
    if seq:
        d[f"{prefix}_instability"] = instability_index(seq)
        d[f"{prefix}_mw"] = molecular_weight(seq)
        d[f"{prefix}_aromaticity"] = sum(seq.count(a) for a in "FWY") / len(seq)
        d[f"{prefix}_n_cys"] = float(seq.count("C"))
        d[f"{prefix}_cys_odd"] = float(seq.count("C") % 2)
    else:
        for k in ("instability", "mw", "aromaticity", "n_cys", "cys_odd"):
            d[f"{prefix}_{k}"] = 0.0
    return d


def liability_descriptors(seq: str, prefix: str) -> dict[str, float]:
    return {f"{prefix}_liab_{k}": float(v)
            for k, v in nb.scan_liabilities_regex(seq).items()}


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------

def build_global_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    """Whole-chain descriptors for heavy and light, plus pairing terms."""
    rows = []
    for rec in df.itertuples(index=False):
        heavy = rec.heavy_chain
        light = rec.light_chain if isinstance(rec.light_chain, str) else ""
        d: dict[str, float] = {"antibody_id": rec.antibody_id}
        d.update(chain_descriptors(heavy, "H"))
        d.update(chain_descriptors(light, "L"))
        d.update(liability_descriptors(heavy, "H"))
        d.update(liability_descriptors(light, "L"))

        qh55, ql55 = net_charge(heavy, 5.5), net_charge(light, 5.5)
        qh74, ql74 = net_charge(heavy, 7.4), net_charge(light, 7.4)
        # Sequence analogue of TAP's structural Fv charge symmetry parameter:
        # a large positive product means both domains carry the same sign of
        # charge, which is associated with better behaviour; a large negative
        # product means opposite charges that can drive self-association.
        d["Fv_charge_symmetry_ph55"] = qh55 * ql55
        d["Fv_charge_symmetry_ph74"] = qh74 * ql74
        d["Fv_net_charge_ph55"] = qh55 + ql55
        d["Fv_net_charge_ph74"] = qh74 + ql74
        d["Fv_pI_gap"] = abs(isoelectric_point(heavy) - isoelectric_point(light)) \
            if light else 0.0
        d["Fv_len"] = float(len(heavy) + len(light))
        rows.append(d)
    return pd.DataFrame(rows)


def build_cdr_descriptors(df: pd.DataFrame, ann: pd.DataFrame) -> pd.DataFrame:
    """CDR-restricted descriptors plus TAP-style whole-CDR aggregates."""
    merged = df[["antibody_id"]].merge(ann, on="antibody_id", how="left")
    rows = []
    for rec in merged.itertuples(index=False):
        d: dict[str, float] = {"antibody_id": rec.antibody_id}
        cdr_concat = ""
        for chain in ("H", "L"):
            for reg in nb.CDR_REGIONS:
                seq = getattr(rec, f"{chain}_{reg}", "") or ""
                if not isinstance(seq, str):
                    seq = ""
                d.update(region_descriptors(seq, f"{chain}_{reg}"))
                d.update(liability_descriptors(seq, f"{chain}_{reg}"))
                cdr_concat += seq

        # TAP-style aggregates over the whole CDR set.
        d.update(region_descriptors(cdr_concat, "CDRall", full_composition=True))
        d["CDRall_total_length"] = float(len(cdr_concat))
        d["CDRall_n_aromatic"] = float(sum(cdr_concat.count(a) for a in "FWY"))
        d["CDRall_n_hydrophobic"] = float(
            sum(cdr_concat.count(a) for a in "AILMFVWY")
        )
        d["CDRall_charge_ph55"] = net_charge(cdr_concat, 5.5)
        d["CDRall_charge_ph74"] = net_charge(cdr_concat, 7.4)
        d["CDRall_n_positive"] = float(sum(cdr_concat.count(a) for a in "KR"))
        d["CDRall_n_negative"] = float(sum(cdr_concat.count(a) for a in "DE"))
        rows.append(d)
    return pd.DataFrame(rows)


def build_trivial_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    The minimum-effort model that any real model must beat.

    Chain lengths and a single mean-hydropathy number per chain.  If a model
    built on 480-dimensional language-model embeddings cannot beat this, the
    embeddings are not contributing anything.
    """
    rows = []
    for rec in df.itertuples(index=False):
        heavy = rec.heavy_chain
        light = rec.light_chain if isinstance(rec.light_chain, str) else ""
        rows.append({
            "antibody_id": rec.antibody_id,
            "H_len": float(len(heavy)),
            "L_len": float(len(light)),
            "H_kd_mean": _scale_mean(heavy, KYTE_DOOLITTLE),
            "L_kd_mean": _scale_mean(light, KYTE_DOOLITTLE) if light else 0.0,
        })
    return pd.DataFrame(rows)


def main() -> int:  # pragma: no cover
    config.ensure_dirs()
    for key in config.ENDPOINTS:
        ds = config.PROCESSED_DIR / f"dataset_{key}.csv"
        an = config.PROCESSED_DIR / f"annotations_{key}.csv"
        if not (ds.exists() and an.exists()):
            print(f"[descriptors] skipping {key}: missing inputs")
            continue
        df = pd.read_csv(ds)
        ann = pd.read_csv(an, keep_default_na=False, na_values=["__NA__"])
        g = build_global_descriptors(df)
        c = build_cdr_descriptors(df, ann)
        out = g.merge(c, on="antibody_id")
        path = config.PROCESSED_DIR / f"descriptors_{key}.csv"
        out.to_csv(path, index=False, encoding="utf-8")
        print(f"[descriptors] {key}: {out.shape[0]} x {out.shape[1] - 1} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
