"""
Fast pre-flight checks. No external data, no model fitting, a few seconds.

Run as:  python -m tests.smoke

These exist because of two failures that actually happened while building this
project: a mis-indented line left an analysis module unimportable (and the
pipeline only discovered it after an hour of modelling compute), and a metric
helper silently formatted booleans as "1.000". Both are the kind of thing a
cheap check catches instantly and a long pipeline run catches expensively.
"""

from __future__ import annotations

import importlib
import sys
import traceback

import numpy as np

FAILURES: list[str] = []

print("Pre-flight smoke tests")
print("=" * 60)


def check(name: str):
    """Decorator that records a pass/fail line instead of aborting the run."""
    def wrap(fn):
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:
            FAILURES.append(name)
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=2)
        return fn
    return wrap


MODULES = [
    "src.config", "src.data.acquire", "src.data.curate", "src.data.splits",
    "src.features.numbering", "src.features.descriptors", "src.features.plm",
    "src.features.build", "src.models.zoo", "src.models.train",
    "src.models.metrics", "src.models.experiment", "src.analysis.plotting",
    "src.analysis.eda", "src.analysis.motifs", "src.tool.screen",
    "src.report.build_report",
]


@check("every module imports")
def _imports():
    for m in MODULES:
        importlib.import_module(m)


@check("sequence canonicalisation rejects non-standard residues")
def _canon():
    from src.data.curate import canonicalise_sequence
    assert canonicalise_sequence(" acdef ") == "ACDEF"
    assert canonicalise_sequence("ACD-EF") == "ACDEF"
    assert canonicalise_sequence("ACDXEF") is None, "X must be rejected"
    assert canonicalise_sequence("") is None
    assert canonicalise_sequence(None) is None


@check("net charge has the right sign and ordering")
def _charge():
    from src.features.descriptors import net_charge, isoelectric_point
    assert net_charge("KKKKK", 7.4) > 3, "poly-lysine must be strongly positive"
    assert net_charge("DDDDD", 7.4) < -3, "poly-aspartate must be strongly negative"
    assert net_charge("KKKKK", 7.4) > net_charge("KKKKK", 12.0), "charge falls with pH"
    assert isoelectric_point("KKKKK") > isoelectric_point("DDDDD")
    # pI is the pH where net charge crosses zero.
    for seq in ("KKKDDD", "ACDEFGHIKLMNPQRSTVWY"):
        assert abs(net_charge(seq, isoelectric_point(seq))) < 0.05


@check("hydrophobicity scales are complete and correctly ordered")
def _scales():
    from src.features import descriptors as D
    from src.config import STANDARD_AA
    for name, scale in (("KD", D.KYTE_DOOLITTLE), ("Eisenberg", D.EISENBERG),
                        ("Black-Mould", D.BLACK_MOULD)):
        assert set(scale) == set(STANDARD_AA), f"{name} missing residues"
        assert scale["I"] > scale["R"], f"{name}: Ile must exceed Arg"
    assert D.BLACK_MOULD["F"] == 1.000
    assert D.BLACK_MOULD["R"] == 0.000


@check("liability regexes match the canonical motifs")
def _liabilities():
    from src.features.numbering import scan_liabilities_regex
    c = scan_liabilities_regex("NASNPSNGTDGDPMWC")
    assert c["n_glycosylation"] >= 1, "NAS is a sequon"
    assert c["deamidation_NG"] >= 1
    assert c["isomerisation_DG"] >= 1
    assert c["fragmentation_DP"] >= 1
    # N-P-S must NOT count: proline blocks glycosylation.
    assert scan_liabilities_regex("NPS")["n_glycosylation"] == 0
    assert scan_liabilities_regex("CC")["cysteine_parity_odd"] == 0
    assert scan_liabilities_regex("CCC")["cysteine_parity_odd"] == 1
    assert scan_liabilities_regex("CC")["cysteine_excess_over_canonical"] == 0


@check("regression metrics are correct and argument order matters")
def _reg_metrics():
    from src.models.metrics import regression_metrics
    y = np.array([1.0, 2, 3, 4, 5])
    m = regression_metrics(y, y)
    assert abs(m["r2"] - 1.0) < 1e-9 and abs(m["mae"]) < 1e-9
    assert abs(m["spearman"] - 1.0) < 1e-9
    m2 = regression_metrics(y, y + 1.0)
    assert abs(m2["mae"] - 1.0) < 1e-9
    assert abs(m2["rmse"] - 1.0) < 1e-9
    # R-squared is asymmetric: swapping the arguments must change it.
    a, b = np.array([1.0, 2, 3, 4]), np.array([1.0, 2, 3, 8])
    assert regression_metrics(a, b)["r2"] != regression_metrics(b, a)["r2"]


@check("conformal intervals hit nominal coverage on exchangeable data")
def _conformal():
    from src.models.metrics import conformal_intervals, interval_coverage
    rng = np.random.default_rng(0)
    calib = rng.normal(0, 1, 500)
    truth = rng.normal(0, 1, 2000)
    lo, hi, q = conformal_intervals(calib, np.zeros(2000), alpha=0.2)
    cov = interval_coverage(truth, lo, hi)["coverage"]
    assert 0.74 < cov < 0.86, f"coverage {cov} far from nominal 0.80"


@check("permutation and bootstrap p-values are never exactly zero")
def _pvalues():
    from src.models.metrics import permutation_pvalue, paired_bootstrap_difference
    rng = np.random.default_rng(0)
    y = np.linspace(0, 10, 80)
    pred = y + rng.normal(0, 0.5, 80)
    p = permutation_pvalue(1.0, y, pred, n_perm=200, seed=0)
    assert p["p_value"] > 0
    d = paired_bootstrap_difference(y, pred, rng.normal(0, 1, 80),
                                    n_boot=200, seed=0)
    assert d["p_two_sided"] > 0
    assert d["ci_low"] <= d["difference"] <= d["ci_high"]


@check("Benjamini-Hochberg is monotone and bounded")
def _bh():
    from src.models.metrics import benjamini_hochberg
    p = np.array([0.001, 0.01, 0.03, 0.2, 0.9])
    rej, padj = benjamini_hochberg(p)
    assert np.all(padj >= p - 1e-12), "adjusted p must never fall below raw p"
    assert np.all(np.diff(padj) >= -1e-12), "adjusted p must be monotone"
    assert np.all((padj >= 0) & (padj <= 1))


@check("report formatting renders booleans, blanks and tiny numbers correctly")
def _fmt():
    import pandas as pd
    from src.report.build_report import fmt, md_table
    assert fmt(True) == "yes" and fmt(False) == "no", "bools must not print as 1.000"
    assert fmt(np.bool_(True)) == "yes"
    assert fmt(float("nan")) == "n/a"
    assert "e-" in fmt(3e-9, 5), "tiny p-values must not collapse to 0.00000"
    assert fmt(1.23456, 2) == "1.23"
    df = pd.DataFrame({"a": [1.0], "flag": [True]})
    out = md_table(df, ["a", "flag", "absent"], ["a", "flag", "absent"])
    assert "yes" in out and "_n/a_" in out, "missing column must be marked"
    assert md_table(pd.DataFrame(), ["a"], ["a"]).startswith("_(no data")


@check("every model in the zoo fits at low and high dimensionality")
def _zoo():
    from sklearn.model_selection import GridSearchCV, GroupKFold
    from src.models.zoo import regression_zoo, classification_zoo
    rng = np.random.default_rng(0)
    for p in (4, 300):
        X = rng.normal(size=(60, p))
        y = X[:, :2].sum(1) + rng.normal(size=60) * 0.3
        g = np.arange(60) // 2
        for name, (est, grid) in regression_zoo().items():
            s = GridSearchCV(est, grid, scoring="neg_mean_absolute_error",
                             cv=GroupKFold(n_splits=3), n_jobs=1)
            s.fit(X, y, groups=g)
            pred = np.asarray(s.best_estimator_.predict(X[:4]))
            assert pred.ndim == 1, f"{name} must return a 1-D prediction"
        yc = (y > np.quantile(y, 0.8)).astype(int)
        for name, (est, grid) in classification_zoo().items():
            s = GridSearchCV(est, grid, scoring="average_precision",
                             cv=GroupKFold(n_splits=3), n_jobs=1)
            s.fit(X, yc, groups=g)


@check("CDR clustering keeps identical sequences together")
def _clustering():
    from src.data.splits import cluster_by_identity
    ident = np.array([
        [1.00, 0.95, 0.10],
        [0.95, 1.00, 0.12],
        [0.10, 0.12, 1.00],
    ])
    labels = cluster_by_identity(ident, threshold=0.8)
    assert labels[0] == labels[1], "95%-identical pair must share a cluster"
    assert labels[2] != labels[0], "10%-identical sequence must be separate"


@check("no function references an undefined name")
def _undefined_names():
    """
    Catch NameErrors that hide inside long functions.

    A missing variable on a rarely-reached branch -- the external-validation
    block runs only after an hour of model fitting -- will not surface until
    that branch executes. A static pass over every module finds it in
    milliseconds. This exists because exactly that happened here once.
    """
    import ast
    import builtins
    import pathlib

    def bindings(node) -> set[str]:
        """Every name bound anywhere in this subtree, by any mechanism."""
        out: set[str] = set()
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
                out.add(n.id)
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                out.add(n.name)
                a = getattr(n, "args", None)
                if a is not None:
                    out.update(x.arg for x in (list(a.args) + list(a.kwonlyargs)
                                               + list(getattr(a, "posonlyargs", []))))
                    if a.vararg:
                        out.add(a.vararg.arg)
                    if a.kwarg:
                        out.add(a.kwarg.arg)
            elif isinstance(n, ast.Lambda):
                a = n.args
                out.update(x.arg for x in (list(a.args) + list(a.kwonlyargs)
                                           + list(getattr(a, "posonlyargs", []))))
                if a.vararg:
                    out.add(a.vararg.arg)
                if a.kwarg:
                    out.add(a.kwarg.arg)
            elif isinstance(n, ast.ExceptHandler) and n.name:
                out.add(n.name)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                for alias in n.names:
                    out.add((alias.asname or alias.name).split(".")[0])
            elif isinstance(n, (ast.Global, ast.Nonlocal)):
                out.update(n.names)
        return out

    problems = []
    for path in sorted(pathlib.Path("src").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        module_scope = set(dir(builtins)) | bindings(tree)

        # Only top-level functions are analysed, and a nested function's body
        # is treated as part of its parent -- so a closure referencing an
        # enclosing local is correctly accepted. The check is deliberately
        # conservative: it flags a name only when it is bound NOWHERE in the
        # function, which is the case that is always a genuine bug.
        for fn in tree.body:
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            a = fn.args
            bound = set(module_scope) | bindings(fn)
            bound.update(x.arg for x in (list(a.args) + list(a.kwonlyargs)
                                         + list(getattr(a, "posonlyargs", []))))
            if a.vararg:
                bound.add(a.vararg.arg)
            if a.kwarg:
                bound.add(a.kwarg.arg)
            for node in ast.walk(fn):
                if (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                        and node.id not in bound):
                    problems.append(f"{path}:{node.lineno} '{node.id}' in {fn.name}()")

    assert not problems, "undefined names:\n    " + "\n    ".join(sorted(set(problems)))


@check("nested CV folds never leak a test antibody into training")
def _fold_integrity():
    from src.data.splits import make_cv_folds
    y = np.random.default_rng(0).normal(size=100)
    groups = np.arange(100) // 2
    repeats = make_cv_folds(y, groups, n_splits=5, n_repeats=2, seed=0)
    for folds in repeats:
        seen = []
        for tr, te in folds:
            assert not (set(tr) & set(te)), "train and test overlap"
            assert not (set(groups[tr]) & set(groups[te])), "a group straddles the split"
            seen.extend(te.tolist())
        assert sorted(seen) == list(range(100)), "folds must partition the data"


def main() -> int:
    print("=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED: {', '.join(FAILURES)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
