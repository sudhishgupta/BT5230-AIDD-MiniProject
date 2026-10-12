"""
Metrics, confidence intervals and significance tests.

Two decisions here matter more than the formulas:

**Pool, do not average.**  With 137 antibodies and 5 folds, each test fold has
~27 points.  R-squared and Spearman computed on 27 points are extremely noisy,
and averaging those noisy per-fold values is biased -- R-squared in particular,
because each fold's variance denominator is computed on its own small sample.
We therefore collect out-of-fold predictions across the whole repeat, then
compute one metric on all 137 predictions.  Repeats give the spread.

**Report intervals, not point estimates.**  Every headline number carries a
bootstrap confidence interval, and the primary rank metric carries an
empirical p-value from a label-permutation test.  On a dataset this small, a
Spearman of 0.4 with a CI spanning [0.1, 0.6] is a very different claim from
one with a CI of [0.35, 0.45], and the reader needs to see which they have.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import (
    average_precision_score, f1_score, matthews_corrcoef,
    mean_absolute_error, r2_score, roc_auc_score, balanced_accuracy_score,
)

from src import config


# --------------------------------------------------------------------------
# Point metrics
# --------------------------------------------------------------------------

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[ok], y_pred[ok]
    if len(y_true) < 3:
        return {k: float("nan") for k in
                ("mae", "rmse", "r2", "spearman", "pearson", "n")}

    rho, rho_p = stats.spearmanr(y_true, y_pred)
    r, r_p = stats.pearsonr(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": float(rho),
        "spearman_p": float(rho_p),
        "pearson": float(r),
        "pearson_p": float(r_p),
        "n": int(len(y_true)),
    }


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray,
                           threshold: float | None = None,
                           positive_rate: float | None = None) -> dict[str, float]:
    """
    Classification metrics from a continuous score.

    ``threshold`` fixes the operating point for F1 and MCC.  Supply it whenever
    one is available from training, because the fallback is not free of
    test-set information: without a threshold we take the score quantile
    matching the positive rate, and if ``positive_rate`` is also absent we fall
    back to the *observed* rate in ``y_true``.  That last case lets the test
    labels influence the operating point, which flatters F1 and MCC.

    The ranking metrics -- ROC-AUC and PR-AUC -- are threshold-free and are
    therefore the ones to trust when no training threshold was supplied.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    ok = np.isfinite(y_score)
    y_true, y_score = y_true[ok], y_score[ok]

    out: dict[str, float] = {"n": int(len(y_true)),
                             "n_positive": int(y_true.sum()),
                             "prevalence": float(y_true.mean()) if len(y_true) else float("nan")}
    if len(np.unique(y_true)) < 2:
        for k in ("roc_auc", "pr_auc", "pr_auc_lift", "f1", "mcc", "balanced_accuracy"):
            out[k] = float("nan")
        return out

    out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    out["pr_auc"] = float(average_precision_score(y_true, y_score))
    # PR-AUC is only interpretable against the prevalence baseline: a PR-AUC
    # of 0.3 is excellent at 10% prevalence and useless at 50%.
    out["pr_auc_lift"] = float(out["pr_auc"] / out["prevalence"]) if out["prevalence"] else float("nan")

    if threshold is None:
        rate = positive_rate if positive_rate is not None else float(y_true.mean())
        threshold = float(np.quantile(y_score, 1.0 - rate))
        out["threshold_source"] = ("supplied_positive_rate" if positive_rate is not None
                                   else "observed_test_prevalence")
    else:
        out["threshold_source"] = "fixed"
    y_hat = (y_score >= threshold).astype(int)
    out["threshold"] = float(threshold)
    out["f1"] = float(f1_score(y_true, y_hat, zero_division=0))
    out["mcc"] = float(matthews_corrcoef(y_true, y_hat))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_hat))
    return out


# --------------------------------------------------------------------------
# Bootstrap confidence intervals
# --------------------------------------------------------------------------

def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, metric: str,
                 kind: str = "regression", n_boot: int | None = None,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap interval for one metric."""
    n_boot = n_boot or config.N_BOOTSTRAP
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    fn = regression_metrics if kind == "regression" else classification_metrics

    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if kind == "classification" and len(np.unique(y_true[idx])) < 2:
            continue
        v = fn(y_true[idx], y_pred[idx]).get(metric, float("nan"))
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < 20:
        return float("nan"), float("nan")
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# --------------------------------------------------------------------------
# Permutation test
# --------------------------------------------------------------------------

def permutation_pvalue(observed: float, y_true: np.ndarray, y_pred: np.ndarray,
                       metric: str = "spearman", n_perm: int | None = None,
                       seed: int = 0) -> dict[str, float]:
    """
    Empirical p-value by permuting the labels against fixed predictions.

    This is the cheap permutation test: it asks whether the association
    between the model's ranking and the truth could have arisen by chance.
    It does NOT re-fit the model, so it does not test the whole pipeline --
    it tests the final correlation.  The full refit-based test is reported
    separately by the training code for the headline model only, because
    refitting 1000 times is expensive.
    """
    n_perm = n_perm or config.N_PERMUTATIONS
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rng = np.random.default_rng(seed)

    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        perm = rng.permutation(y_true)
        if metric == "spearman":
            null[i] = stats.spearmanr(perm, y_pred).statistic
        else:
            null[i] = regression_metrics(perm, y_pred).get(metric, np.nan)

    null = null[np.isfinite(null)]
    # One-sided (we only care about positive association), with the +1
    # correction that keeps the p-value from ever being exactly zero.
    p = float((np.sum(null >= observed) + 1) / (len(null) + 1))
    return {
        "observed": float(observed),
        "p_value": p,
        "null_mean": float(null.mean()),
        "null_std": float(null.std(ddof=1)),
        "n_permutations": int(len(null)),
    }


# --------------------------------------------------------------------------
# Paired comparison between two models
# --------------------------------------------------------------------------

def paired_bootstrap_difference(y_true: np.ndarray, pred_a: np.ndarray,
                                pred_b: np.ndarray, metric: str = "spearman",
                                n_boot: int | None = None, seed: int = 0) -> dict[str, float]:
    """
    Is model A better than model B on the SAME antibodies?

    Resampling antibodies jointly and taking the difference each time is the
    right way to compare two models on one small dataset -- it cancels the
    shared "which antibodies did we happen to draw" variance that dominates
    the two independent intervals, which is why two heavily overlapping CIs
    can still correspond to a reliable difference.
    """
    n_boot = n_boot or config.N_BOOTSTRAP
    y_true = np.asarray(y_true, dtype=float)
    pred_a = np.asarray(pred_a, dtype=float)
    pred_b = np.asarray(pred_b, dtype=float)
    n = len(y_true)
    rng = np.random.default_rng(seed)

    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        a = regression_metrics(y_true[idx], pred_a[idx]).get(metric, np.nan)
        b = regression_metrics(y_true[idx], pred_b[idx]).get(metric, np.nan)
        if np.isfinite(a) and np.isfinite(b):
            diffs.append(a - b)
    if len(diffs) < 20:
        return {"difference": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "p_two_sided": float("nan")}
    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # The +1 correction keeps the achieved significance level away from exactly
    # zero: with B resamples the smallest p we can honestly resolve is
    # 2/(B+1), and reporting 0.000 would claim more resolution than the
    # resampling provides.
    n = len(diffs)
    p = 2.0 * min((np.sum(diffs <= 0) + 1) / (n + 1),
                  (np.sum(diffs >= 0) + 1) / (n + 1))
    return {
        "difference": float(diffs.mean()),
        "ci_low": float(lo), "ci_high": float(hi),
        "p_two_sided": float(min(1.0, p)),
        "n_resamples": int(n),
    }


# --------------------------------------------------------------------------
# Multiple-comparison control
# --------------------------------------------------------------------------

def benjamini_hochberg(pvalues: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Return (rejected, adjusted_p) under Benjamini-Hochberg FDR control."""
    try:
        from statsmodels.stats.multitest import multipletests
        rej, padj, _, _ = multipletests(pvalues, alpha=alpha, method="fdr_bh")
        return rej, padj
    except Exception:
        p = np.asarray(pvalues, dtype=float)
        n = len(p)
        order = np.argsort(p)
        ranked = p[order] * n / (np.arange(n) + 1)
        padj_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
        padj = np.empty(n)
        padj[order] = np.clip(padj_sorted, 0, 1)
        return padj <= alpha, padj


# --------------------------------------------------------------------------
# Conformal prediction
# --------------------------------------------------------------------------

def conformal_intervals(residuals_calib: np.ndarray, predictions: np.ndarray,
                        alpha: float = None) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Split-conformal prediction intervals.

    Given absolute residuals on a calibration set never used for fitting, the
    (1-alpha) quantile q of those residuals gives intervals pred +/- q with a
    finite-sample guarantee: the interval covers the truth with probability at
    least 1-alpha, assuming only that calibration and test points are
    exchangeable.  No distributional assumption, no asymptotics.

    The exchangeability assumption is exactly what a cohort shift breaks, so
    we report measured coverage on the external sets rather than claiming the
    guarantee transfers.
    """
    alpha = config.CONFORMAL_ALPHA if alpha is None else alpha
    r = np.sort(np.abs(np.asarray(residuals_calib, dtype=float)))
    n = len(r)
    # The finite-sample-valid quantile level.
    k = int(np.ceil((n + 1) * (1 - alpha))) - 1
    k = int(np.clip(k, 0, n - 1))
    q = float(r[k])
    preds = np.asarray(predictions, dtype=float)
    return preds - q, preds + q, q


def interval_coverage(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    inside = (y_true >= lo) & (y_true <= hi)
    return {
        "coverage": float(inside.mean()),
        "mean_width": float(np.mean(hi - lo)),
        "n": int(len(y_true)),
    }
