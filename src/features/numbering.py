"""
Antibody numbering, region extraction, germline assignment and chemical
liability annotation -- all via AntPack.

Why AntPack and not ANARCI: ANARCI is the field standard but shells out to
HMMER, which has no supported Windows build and cannot be pip-installed.
AntPack 0.3.8.6.2 is a pure pip install (one dependency, numpy), ships a
cp313 Windows wheel, needs no licence key, and implements IMGT/Kabat/Martin/
Aho numbering natively.  Verified on this machine against published
trastuzumab CDRs before use.

What this module provides per chain:
  * IMGT numbering and the seven regions (FR1-3 + CDR1-3 + FR4)
  * Kabat regions as well, because the aggregation literature quotes Kabat
  * germline V and J gene calls with identity fractions
  * chemical liability motifs with severity, from AntPack's curated set,
    cross-checked against our own independent regex implementation
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

from src import config

# Bump this whenever the annotation logic changes (regions, liability
# patterns, germline calls, the ``ok`` criterion). It is part of the cache
# key, so bumping it forces every cached annotation to be recomputed.
ANNOTATION_VERSION = "2026-10-11.1"

# --------------------------------------------------------------------------
# Independent liability regexes.
#
# These duplicate AntPack's curated set on purpose.  Having two independent
# implementations lets us report an agreement rate, which is a cheap but real
# check that neither is silently broken.  Patterns follow the conventions used
# in antibody developability triage (Lu et al. 2019; Jarasch et al. 2015).
# --------------------------------------------------------------------------

LIABILITY_PATTERNS: dict[str, tuple[str, str]] = {
    # name:                 (regex,            severity)
    "n_glycosylation":      (r"N[^P][ST]",     "high"),
    "deamidation_NG":       (r"NG",            "high"),
    "deamidation_NS":       (r"NS",            "medium"),
    "deamidation_NT":       (r"NT",            "low"),
    "deamidation_NN":       (r"NN",            "low"),
    "deamidation_NH":       (r"NH",            "low"),
    "isomerisation_DG":     (r"DG",            "high"),
    "isomerisation_DS":     (r"DS",            "medium"),
    "isomerisation_DD":     (r"DD",            "medium"),
    "isomerisation_DT":     (r"DT",            "low"),
    "fragmentation_DP":     (r"DP",            "medium"),
    "oxidation_M":          (r"M",             "medium"),
    "oxidation_W":          (r"W",             "medium"),
    # Named "total", not "free". Every variable domain carries the two
    # canonical IMGT Cys23/Cys104 residues forming the intradomain disulfide,
    # so a raw cysteine count is ~2 for almost every antibody and carries
    # almost no information. The informative quantities are the EXCESS over
    # that canonical pair and the parity, both derived in
    # scan_liabilities_regex below; a genuinely unpaired cysteine is what
    # creates the liability.
    "cysteine_total":       (r"C",             "context"),
    "glycation_KE":         (r"K[DE]",         "low"),
}

REGION_ORDER = ["fmwk1", "cdr1", "fmwk2", "cdr2", "fmwk3", "cdr3", "fmwk4"]
CDR_REGIONS = ["cdr1", "cdr2", "cdr3"]


@dataclass
class ChainAnnotation:
    """Everything we know about one variable domain."""

    sequence: str
    chain_type: str            # 'H', 'K' or 'L' as called by AntPack
    percent_identity: float
    error: str
    numbering: tuple[str, ...] = ()
    regions_imgt: dict[str, str] = field(default_factory=dict)
    regions_kabat: dict[str, str] = field(default_factory=dict)
    v_gene: str = ""
    j_gene: str = ""
    v_identity: float = float("nan")
    j_identity: float = float("nan")
    liabilities: list[dict] = field(default_factory=list)

    # Minimum fraction of positions matching the scheme's consensus before we
    # distrust the alignment outright.  AntPack returns ~0.92-1.00 for genuine
    # variable domains; anything far below that is not an Fv.
    MIN_IDENTITY = 0.70

    @property
    def ok(self) -> bool:
        """
        Whether the numbering is usable.

        Deliberately NOT ``not self.error``.  AntPack emits messages such as
        "Unexpected AA: the FWGxG conserved region has been modified" for
        antibodies whose framework differs from the consensus at a conserved
        position -- for example a natural WDQGT instead of WGQGT in FR4.
        Those alignments are still correct: the numbering covers every residue
        and the regions come out sensibly.  Treating the message as a failure
        would silently discard real antibodies for having an unusual but
        perfectly valid framework, which is exactly the kind of antibody a
        developability study should not be throwing away.

        We therefore require structural soundness -- a recognised chain type,
        a numbering that covers the whole sequence, and a plausible identity --
        and keep the message as a warning.
        """
        return (
            self.chain_type in {"H", "K", "L"}
            and len(self.numbering) == len(self.sequence)
            and self.percent_identity >= self.MIN_IDENTITY
        )

    @property
    def has_warning(self) -> bool:
        return bool(self.error)


# --------------------------------------------------------------------------
# AntPack handles (built once; the C++ objects are not cheap to construct)
# --------------------------------------------------------------------------

_TOOLS: dict | None = None


def _tools() -> dict:
    global _TOOLS
    if _TOOLS is None:
        from antpack import SingleChainAnnotator, LiabilitySearchTool, VJGeneTool
        _TOOLS = {
            "imgt": SingleChainAnnotator(chains=["H", "K", "L"], scheme="imgt"),
            "kabat": SingleChainAnnotator(chains=["H", "K", "L"], scheme="kabat"),
            "liability": LiabilitySearchTool(),
            "vj": VJGeneTool(),
        }
    return _TOOLS


def _regions_from_labels(seq: str, labels) -> dict[str, str]:
    out: dict[str, str] = {k: "" for k in REGION_ORDER}
    for aa, lab in zip(seq, labels):
        if lab in out:
            out[lab] += aa
    return out


def annotate_chain(sequence: str, with_germline: bool = True) -> ChainAnnotation:
    """Fully annotate one antibody variable domain."""
    t = _tools()
    seq = sequence.strip().upper()

    aln_imgt = t["imgt"].analyze_seq(seq)
    numbering, pct_id, chain, err = aln_imgt

    ann = ChainAnnotation(
        sequence=seq, chain_type=chain, percent_identity=float(pct_id),
        error=str(err or ""), numbering=tuple(numbering),
    )
    if not ann.ok:
        return ann

    ann.regions_imgt = _regions_from_labels(
        seq, t["imgt"].assign_cdr_labels(numbering, chain)
    )

    # Kabat regions come from a Kabat-scheme alignment of the same sequence.
    try:
        aln_kab = t["kabat"].analyze_seq(seq)
        ann.regions_kabat = _regions_from_labels(
            seq, t["kabat"].assign_cdr_labels(aln_kab[0], aln_kab[2])
        )
    except Exception:
        ann.regions_kabat = {}

    # Chemical liabilities, with their position ranges and severity text.
    try:
        hits = t["liability"].analyze_seq(seq, aln_imgt, "imgt", "imgt")
        for (lo, hi), desc in hits:
            motif = seq[lo:hi + 1]
            ann.liabilities.append({
                "start": int(lo), "end": int(hi), "motif": motif,
                "description": desc,
                "in_cdr": _position_in_cdr(lo, hi, seq, ann.regions_imgt),
            })
    except Exception as exc:  # pragma: no cover
        ann.liabilities = [{"error": f"{type(exc).__name__}: {exc}"}]

    if with_germline:
        try:
            v, j, vid, jid, _species = t["vj"].assign_vj_genes(
                aln_imgt, seq, "human", "identity"
            )
            ann.v_gene, ann.j_gene = str(v), str(j)
            ann.v_identity, ann.j_identity = float(vid), float(jid)
        except Exception:
            pass

    return ann


def _position_in_cdr(lo: int, hi: int, seq: str, regions: dict[str, str]) -> bool:
    """Is any part of [lo, hi] inside an IMGT CDR?"""
    cursor = 0
    spans: list[tuple[int, int]] = []
    for name in REGION_ORDER:
        seg = regions.get(name, "")
        if not seg:
            continue
        start = seq.find(seg, cursor)
        if start < 0:
            continue
        if name in CDR_REGIONS:
            spans.append((start, start + len(seg) - 1))
        cursor = start + len(seg)
    return any(lo <= e and hi >= s for s, e in spans)


# --------------------------------------------------------------------------
# Independent regex liability scan (cross-check)
# --------------------------------------------------------------------------

def scan_liabilities_regex(seq: str) -> dict[str, int]:
    """Count liability motifs with our own regexes, independent of AntPack."""
    counts = {}
    for name, (pattern, _sev) in LIABILITY_PATTERNS.items():
        counts[name] = len(re.findall(f"(?={pattern})", seq))
    n_cys = seq.count("C")
    # An odd cysteine count guarantees at least one unpaired thiol, which is a
    # real aggregation and heterogeneity liability. The excess over the two
    # canonical framework cysteines is the part of the count that varies
    # between antibodies and therefore the part that can carry signal.
    counts["cysteine_parity_odd"] = int(n_cys % 2 == 1)
    counts["cysteine_excess_over_canonical"] = max(0, n_cys - 2)
    return counts


# --------------------------------------------------------------------------
# Batch annotation with on-disk caching
# --------------------------------------------------------------------------

def annotate_dataset(df: pd.DataFrame, cache_name: str,
                     use_cache: bool = True) -> pd.DataFrame:
    """
    Annotate every antibody in a curated dataset.

    Returns one row per antibody with heavy/light regions, germline calls and
    liability counts, keyed by antibody_id.
    """
    # The cache key hashes the actual SEQUENCES plus a version constant, not
    # the antibody ids.  Ids are positional (``<cohort>_<row index>``), so two
    # different curations of the same cohort produce identical id sets; keying
    # on ids would happily hand back annotations computed for different
    # sequences.  Bumping ANNOTATION_VERSION invalidates every cache when the
    # annotation logic changes.
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        [[str(a), str(h), str(l or "")] for a, h, l in
         zip(df["antibody_id"], df["heavy_chain"], df["light_chain"])],
        sort_keys=True,
    )
    digest = hashlib.sha256(
        (ANNOTATION_VERSION + payload).encode("utf-8")
    ).hexdigest()[:16]
    cache = config.CACHE_DIR / f"annotations_{cache_name}_{digest}.csv.gz"
    if use_cache and cache.exists():
        cached = pd.read_csv(cache, keep_default_na=False, na_values=["__NA__"])
        if len(cached) == len(df) and set(cached["antibody_id"]) == set(df["antibody_id"]):
            print(f"[numbering] annotation cache hit: {cache.name}")
            return cached

    rows = []
    n_fail_h = n_fail_l = 0
    n_warn_h = n_warn_l = 0
    for rec in df.itertuples(index=False):
        row: dict = {"antibody_id": rec.antibody_id}

        h = annotate_chain(rec.heavy_chain)
        if not h.ok:
            n_fail_h += 1
        if h.has_warning:
            n_warn_h += 1
        row["heavy_ok"] = h.ok
        row["heavy_warning"] = h.error
        row["heavy_chain_type"] = h.chain_type
        row["heavy_numbering_identity"] = h.percent_identity
        row["heavy_v_gene"] = h.v_gene
        row["heavy_j_gene"] = h.j_gene
        row["heavy_v_identity"] = h.v_identity
        for reg in REGION_ORDER:
            row[f"H_{reg}"] = h.regions_imgt.get(reg, "")
            row[f"H_{reg}_kabat"] = h.regions_kabat.get(reg, "")
        row["H_liabilities"] = json.dumps(h.liabilities)

        light = (rec.light_chain or "")
        if light:
            l = annotate_chain(light)
            if not l.ok:
                n_fail_l += 1
            if l.has_warning:
                n_warn_l += 1
            row["light_ok"] = l.ok
            row["light_warning"] = l.error
            row["light_chain_type"] = l.chain_type
            row["light_numbering_identity"] = l.percent_identity
            row["light_v_gene"] = l.v_gene
            row["light_j_gene"] = l.j_gene
            row["light_v_identity"] = l.v_identity
            for reg in REGION_ORDER:
                row[f"L_{reg}"] = l.regions_imgt.get(reg, "")
                row[f"L_{reg}_kabat"] = l.regions_kabat.get(reg, "")
            row["L_liabilities"] = json.dumps(l.liabilities)
        else:
            row["light_ok"] = False
            row["light_warning"] = "no light chain"
            row["light_chain_type"] = ""
            row["light_numbering_identity"] = float("nan")
            row["light_v_gene"] = row["light_j_gene"] = ""
            row["light_v_identity"] = float("nan")
            for reg in REGION_ORDER:
                row[f"L_{reg}"] = ""
                row[f"L_{reg}_kabat"] = ""
            row["L_liabilities"] = "[]"

        for ch, seq in (("H", rec.heavy_chain), ("L", light)):
            for k, v in scan_liabilities_regex(seq).items():
                row[f"{ch}_regex_{k}"] = v

        rows.append(row)

    out = pd.DataFrame(rows)
    print(f"[numbering] annotated {len(out)} antibodies | "
          f"unusable H/L: {n_fail_h}/{n_fail_l} | "
          f"framework warnings H/L: {n_warn_h}/{n_warn_l}")

    if use_cache:
        out.to_csv(cache, index=False, encoding="utf-8")
    return out


def concat_cdrs(row: pd.Series, chains: tuple[str, ...] = ("H", "L")) -> str:
    """Concatenated CDR string, used for identity-based clustering."""
    return "".join(row.get(f"{c}_{r}", "") or "" for c in chains for r in CDR_REGIONS)


def main() -> int:  # pragma: no cover - CLI convenience
    config.ensure_dirs()
    for key in config.ENDPOINTS:
        path = config.PROCESSED_DIR / f"dataset_{key}.csv"
        if not path.exists():
            print(f"[numbering] skipping {key}: run src.data.curate first")
            continue
        df = pd.read_csv(path)
        ann = annotate_dataset(df, cache_name=key)
        out = config.PROCESSED_DIR / f"annotations_{key}.csv"
        ann.to_csv(out, index=False, encoding="utf-8")
        print(f"[numbering] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
