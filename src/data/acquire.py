"""
Acquire the FLAb antibody fitness dataset at a pinned revision.

The full FLAb tree does not check out cleanly on Windows (some paths under
score/ exceed the default path limit), and we only need the measurement
CSVs anyway.  So we do a blobless, sparse clone of the data/ subtree at an
exact commit and record a manifest with file hashes for provenance.

Run:
    python -m src.data.acquire
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from src import config


def _run(cmd: list[str], cwd: Path | None = None) -> str:
    """Run a git command, raising with useful output on failure."""
    proc = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc.stdout.strip()


def clone_flab(force: bool = False) -> Path:
    """Clone (or reuse) the FLAb repository at the pinned commit."""
    dest = config.FLAB_DIR

    if dest.exists() and not force:
        try:
            head = _run(["git", "rev-parse", "HEAD"], cwd=dest)
            if head == config.FLAB_PINNED_COMMIT:
                print(f"[acquire] FLAb already present at pinned commit {head[:10]}")
                return dest
            print(f"[acquire] FLAb present but at {head[:10]}, re-pinning...")
        except Exception:
            print("[acquire] Existing FLAb directory is not a usable git repo; re-cloning.")
            force = True

    if force and dest.exists():
        import shutil
        shutil.rmtree(dest, ignore_errors=True)

    dest.parent.mkdir(parents=True, exist_ok=True)

    if not dest.exists():
        print(f"[acquire] Cloning {config.FLAB_REPO_URL} (blobless, sparse)...")
        _run([
            "git", "-c", "core.longpaths=true",
            "clone", "--filter=blob:none", "--no-checkout", "--sparse",
            config.FLAB_REPO_URL, str(dest),
        ])

    _run(["git", "-c", "core.longpaths=true", "sparse-checkout", "set",
          *config.FLAB_SPARSE_PATHS], cwd=dest)
    _run(["git", "-c", "core.longpaths=true", "fetch", "--depth", "1",
          "origin", config.FLAB_PINNED_COMMIT], cwd=dest)
    _run(["git", "-c", "core.longpaths=true", "checkout", config.FLAB_PINNED_COMMIT],
         cwd=dest)

    head = _run(["git", "rev-parse", "HEAD"], cwd=dest)
    if head != config.FLAB_PINNED_COMMIT:
        raise RuntimeError(f"Checked out {head}, expected {config.FLAB_PINNED_COMMIT}")

    print(f"[acquire] FLAb checked out at {head}")
    return dest


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest() -> Path:
    """Record exactly which files this study consumed, with content hashes."""
    entries = []
    for ep in config.ENDPOINTS.values():
        for cohort in (ep.train, *ep.externals):
            path = config.FLAB_DIR / cohort.relpath
            if not path.exists():
                raise FileNotFoundError(
                    f"Expected FLAb file missing: {cohort.relpath}. "
                    "Has the upstream layout changed? Re-check FLAB_PINNED_COMMIT."
                )
            with path.open("r", encoding="utf-8") as fh:
                n_rows = sum(1 for _ in fh) - 1
            entries.append({
                "endpoint": ep.key,
                "cohort": cohort.key,
                "relpath": cohort.relpath,
                "value_column": cohort.value_column,
                "units": cohort.units,
                "cohort_name": cohort.cohort_name,
                "publication": cohort.publication,
                "year": cohort.year,
                "favourable_direction": cohort.favourable_direction,
                "n_rows": n_rows,
                "sha256": _sha256(path),
            })

    manifest = {
        "acquired_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_repo": config.FLAB_REPO_URL,
        "pinned_commit": config.FLAB_PINNED_COMMIT,
        "license_note": (
            "FLAb is distributed by the Gray Lab (Johns Hopkins). The underlying "
            "measurements belong to the cited primary publications. This project "
            "uses the data for non-commercial academic coursework; consult the "
            "upstream repository and the primary papers for licence terms."
        ),
        "files": entries,
    }

    config.ensure_dirs()
    out = config.PROCESSED_DIR / "data_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[acquire] Wrote manifest with {len(entries)} files -> {out}")
    return out


def main() -> int:
    config.ensure_dirs()
    clone_flab(force="--force" in sys.argv)
    write_manifest()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
