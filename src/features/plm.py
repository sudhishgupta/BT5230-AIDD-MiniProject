"""
Protein-language-model embeddings for paired antibody variable domains.

Two models are supported:

  AbLang2 (``ablang2-paired``)  -- the primary model.  It is an antibody
      specific language model trained on paired heavy/light OAS data, so it
      sees the two chains jointly and its representation already encodes
      antibody-specific structure (framework vs CDR, germline context).
      Verified API against the installed source (ablang2 0.2.1):
          model = ablang2.pretrained(model_to_use='ablang2-paired', device='cpu')
          out   = model([[heavy, light], ...], mode='seqcoding')
      ``seqcoding`` mean-pools the residue states of the formatted string
      ``<heavy>|<light>`` over its non-padding positions, giving one vector
      per antibody.  ``align=True`` is deliberately never used because it
      imports ANARCI, which needs HMMER and does not pip-install on Windows.

  ESM-2 35M (``facebook/esm2_t12_35M_UR50D``) -- the general-protein-PLM
      baseline.  It has no antibody-specific training signal, which is
      exactly the point: the comparison answers "does antibody-specific
      pre-training buy anything over a generic protein model of the same
      embedding width?".  Both models output 480-dimensional states, so the
      comparison is not confounded by embedding size.

All embeddings are cached to .npz keyed by a hash of (model, sequences), so
re-running the pipeline does not re-run the forward passes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src import config


# --------------------------------------------------------------------------
# Caching
# --------------------------------------------------------------------------

# Part of every embedding cache key. Bump it whenever the extraction changes
# -- pooling convention, which layer is read, how chains are combined --
# otherwise a cached vector computed under the old convention is silently
# reused under the new one.
EMBEDDING_VERSION = "2026-10-11.1"


def _cache_key(model_tag: str, seqs: list[tuple[str, str]], extra: str = "") -> str:
    h = hashlib.sha256()
    h.update(EMBEDDING_VERSION.encode())
    h.update(model_tag.encode())
    h.update(extra.encode())
    for heavy, light in seqs:
        h.update(heavy.encode())
        h.update(b"|")
        h.update((light or "").encode())
        h.update(b"\n")
    return h.hexdigest()[:24]


def _cache_path(model_tag: str, key: str) -> Path:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return config.CACHE_DIR / f"plm_{model_tag}_{key}.npz"


def _load_cache(path: Path) -> np.ndarray | None:
    if path.exists():
        with np.load(path) as z:
            return z["embeddings"]
    return None


def _save_cache(path: Path, emb: np.ndarray, meta: dict) -> None:
    np.savez_compressed(path, embeddings=emb, meta=json.dumps(meta))


# --------------------------------------------------------------------------
# AbLang2
# --------------------------------------------------------------------------

_ABLANG_MODEL = None


def _get_ablang2(device: str = "cpu"):
    """Load AbLang2 once per process. Weights download from Zenodo on first use."""
    global _ABLANG_MODEL
    if _ABLANG_MODEL is None:
        import ablang2
        print(f"[plm] Loading AbLang2 ({config.ABLANG2_MODEL}) on {device} ...")
        _ABLANG_MODEL = ablang2.pretrained(
            model_to_use=config.ABLANG2_MODEL, random_init=False, ncpu=1, device=device
        )
        _ABLANG_MODEL.freeze()
    return _ABLANG_MODEL


def embed_ablang2(seqs: list[tuple[str, str]], device: str | None = None,
                  batch_size: int = 32, use_cache: bool = True) -> np.ndarray:
    """
    Sequence-level AbLang2 embeddings.

    Parameters
    ----------
    seqs : list of (heavy, light) tuples. ``light`` may be "" for heavy-only.

    Returns
    -------
    (N, D) float32 array, D = 480 for ablang2-paired.
    """
    device = device or config.TORCH_DEVICE
    key = _cache_key("ablang2", seqs, extra=config.ABLANG2_MODEL)
    path = _cache_path("ablang2", key)
    if use_cache:
        cached = _load_cache(path)
        if cached is not None:
            print(f"[plm] AbLang2 cache hit: {cached.shape} <- {path.name}")
            return cached

    model = _get_ablang2(device)
    pairs = [[h, (l or "")] for h, l in seqs]

    out = model(pairs, mode="seqcoding", align=False, batch_size=batch_size)
    emb = np.asarray(out, dtype=np.float32)
    if emb.ndim != 2 or emb.shape[0] != len(seqs):
        raise RuntimeError(
            f"Unexpected AbLang2 seqcoding shape {emb.shape} for {len(seqs)} sequences."
        )

    print(f"[plm] AbLang2 embeddings: {emb.shape}")
    if use_cache:
        _save_cache(path, emb, {"model": config.ABLANG2_MODEL, "n": len(seqs),
                                "dim": int(emb.shape[1]), "mode": "seqcoding"})
    return emb


def embed_ablang2_residue(seqs: list[tuple[str, str]], device: str | None = None,
                          batch_size: int = 16) -> list[np.ndarray]:
    """
    Per-residue AbLang2 states, used for the residue-attribution analysis.

    Returns one (L_i, D) array per antibody, where L_i is the length of the
    formatted string ``<heavy>|<light>`` -- i.e. it includes the two chain
    delimiters and the separator.  ``index_map_for_formatted`` converts
    positions back to (chain, residue index).
    """
    device = device or config.TORCH_DEVICE
    model = _get_ablang2(device)
    pairs = [[h, (l or "")] for h, l in seqs]
    out = model(pairs, mode="rescoding", align=False, batch_size=batch_size)
    return [np.asarray(x, dtype=np.float32) for x in out]


def formatted_pair_string(heavy: str, light: str) -> str:
    """Reproduce AbLang2's internal input formatting, for index bookkeeping."""
    return f"<{heavy}>|<{light}>".replace("<>", "")


def index_map_for_formatted(heavy: str, light: str) -> list[tuple[str, int | None]]:
    """
    Map each position of the formatted AbLang2 input string to its origin.

    Returns a list the same length as ``formatted_pair_string(heavy, light)``
    where each element is ("H", i) / ("L", i) for a real residue, or
    ("special", None) for the ``<``, ``>`` and ``|`` tokens.
    """
    s = formatted_pair_string(heavy, light)
    mapping: list[tuple[str, int | None]] = []
    chain = "H"
    idx = 0
    for ch in s:
        if ch in "<>":
            mapping.append(("special", None))
        elif ch == "|":
            mapping.append(("special", None))
            chain = "L"
            idx = 0
        else:
            mapping.append((chain, idx))
            idx += 1
    return mapping


# --------------------------------------------------------------------------
# ESM-2
# --------------------------------------------------------------------------

_ESM_BUNDLE = None


def _get_esm2(device: str = "cpu"):
    """
    Load ESM-2 35M. Prefers HuggingFace transformers; falls back to fair-esm.

    Returns (backend, tokenizer_or_alphabet, model).
    """
    global _ESM_BUNDLE
    if _ESM_BUNDLE is not None:
        return _ESM_BUNDLE

    import torch

    try:
        from transformers import AutoTokenizer, AutoModel
        print(f"[plm] Loading ESM-2 via transformers ({config.ESM2_HF_ID}) ...")
        tok = AutoTokenizer.from_pretrained(config.ESM2_HF_ID)
        mdl = AutoModel.from_pretrained(config.ESM2_HF_ID)
        mdl.eval().to(torch.device(device))
        _ESM_BUNDLE = ("hf", tok, mdl)
        return _ESM_BUNDLE
    except Exception as exc:  # pragma: no cover - network/offline fallback
        print(f"[plm] transformers route failed ({type(exc).__name__}: {exc}); "
              "falling back to fair-esm.")

    import esm
    print(f"[plm] Loading ESM-2 via fair-esm ({config.ESM2_FAIR_NAME}) ...")
    mdl, alphabet = esm.pretrained.load_model_and_alphabet_hub(config.ESM2_FAIR_NAME)
    mdl.eval().to(torch.device(device))
    _ESM_BUNDLE = ("fair", alphabet, mdl)
    return _ESM_BUNDLE


def _esm2_embed_chains(chains: list[str], device: str, batch_size: int) -> np.ndarray:
    """
    Mean-pool ESM-2 residue states over *real residues only*.

    The subtlety: the attention mask marks <cls>, <eos> and (for HF) padding.
    Averaging over the raw attention mask therefore folds the <cls> and <eos>
    states into the mean.  Those two states are sequence-length dependent and
    carry no residue identity, so they add a systematic, length-correlated
    offset to every embedding -- a spurious feature when the target itself
    correlates with chain length.  We strip them explicitly.
    """
    import torch

    backend, tok, mdl = _get_esm2(device)
    dev = torch.device(device)
    vectors: list[np.ndarray] = []

    for start in range(0, len(chains), batch_size):
        batch = chains[start:start + batch_size]

        if backend == "hf":
            enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=True)
            enc = {k: v.to(dev) for k, v in enc.items()}
            with torch.no_grad():
                out = mdl(**enc)
            hidden = out.last_hidden_state                      # (B, T, D)
            for i, seq in enumerate(batch):
                # token 0 is <cls>; residues occupy 1 .. len(seq)
                reps = hidden[i, 1:len(seq) + 1, :]
                vectors.append(reps.mean(dim=0).cpu().numpy())
        else:
            alphabet = tok
            converter = alphabet.get_batch_converter()
            data = [(f"s{i}", s) for i, s in enumerate(batch)]
            _, _, toks = converter(data)
            toks = toks.to(dev)
            with torch.no_grad():
                out = mdl(toks, repr_layers=[config.ESM2_REPR_LAYER],
                          return_contacts=False)
            hidden = out["representations"][config.ESM2_REPR_LAYER]
            for i, seq in enumerate(batch):
                reps = hidden[i, 1:len(seq) + 1, :]
                vectors.append(reps.mean(dim=0).cpu().numpy())

    return np.asarray(vectors, dtype=np.float32)


def embed_esm2(seqs: list[tuple[str, str]], device: str | None = None,
               batch_size: int | None = None, use_cache: bool = True) -> np.ndarray:
    """
    ESM-2 35M embeddings for paired antibodies.

    Heavy and light chains are embedded *separately* and concatenated.  ESM-2
    has no paired-chain separator token in its vocabulary, so gluing the two
    chains into one string with an artificial linker would put the model
    firmly out of distribution; embedding each chain in its own forward pass
    is the standard treatment and keeps the comparison with AbLang2 fair
    (AbLang2 is the one that genuinely models the pairing).

    Returns (N, 960) for paired input: 480 heavy + 480 light.
    """
    device = device or config.TORCH_DEVICE
    batch_size = batch_size or config.PLM_BATCH_SIZE
    # The layer and the backend both change the numbers, so both belong in
    # the key: an HF->fair-esm fallback or a layer change must not silently
    # reuse vectors produced under the other setting.
    backend = _ESM_BUNDLE[0] if _ESM_BUNDLE else "unloaded"
    key = _cache_key("esm2", seqs,
                     extra=f"{config.ESM2_HF_ID}|layer{config.ESM2_REPR_LAYER}|{backend}")
    path = _cache_path("esm2", key)
    if use_cache:
        cached = _load_cache(path)
        if cached is not None:
            print(f"[plm] ESM-2 cache hit: {cached.shape} <- {path.name}")
            return cached

    heavies = [h for h, _ in seqs]
    lights = [l for _, l in seqs]
    has_light = any(bool(l) for l in lights)

    h_emb = _esm2_embed_chains(heavies, device, batch_size)
    if has_light:
        # Guard against empty strings in a mostly-paired dataset.
        safe_lights = [l if l else "G" for l in lights]
        l_emb = _esm2_embed_chains(safe_lights, device, batch_size)
        for i, l in enumerate(lights):
            if not l:
                l_emb[i] = 0.0
        emb = np.concatenate([h_emb, l_emb], axis=1)
    else:
        emb = h_emb

    print(f"[plm] ESM-2 embeddings: {emb.shape}")
    if use_cache:
        _save_cache(path, emb, {"model": config.ESM2_HF_ID, "n": len(seqs),
                                "dim": int(emb.shape[1]), "pooling": "mean_residues_only"})
    return emb


# --------------------------------------------------------------------------
# Likelihood-style scores (zero-shot comparison)
# --------------------------------------------------------------------------

def ablang2_pseudo_log_likelihood(seqs: list[tuple[str, str]],
                                  device: str | None = None,
                                  batch_size: int = 8) -> np.ndarray:
    """
    AbLang2 pseudo-log-likelihood per antibody.

    This is the kind of score the FLAb benchmark evaluates zero-shot: a
    "how antibody-like is this sequence" number with no supervision.  We
    include it as a zero-shot baseline so the report can state whether
    supervised learning on 137 examples actually beats an unsupervised
    language-model score.
    """
    device = device or config.TORCH_DEVICE
    model = _get_ablang2(device)
    pairs = [[h, (l or "")] for h, l in seqs]
    out = model(pairs, mode="pseudo_log_likelihood", align=False, batch_size=batch_size)
    return np.asarray(out, dtype=np.float32).reshape(-1)
