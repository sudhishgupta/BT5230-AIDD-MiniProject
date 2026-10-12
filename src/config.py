"""
Central configuration for the antibody developability project.

Everything that another person would need to change in order to re-run a
variant of this study lives here: paths, the pinned dataset revision, the
endpoint definition, the split protocol and the random seeds.

Design decision (see docs/METHODOLOGY.md for the full argument):
  The primary endpoint is HIC retention time in minutes, measured on the
  Jain et al. 2017 clinical-stage antibody panel.  It is chosen because it
  is (a) a single, unambiguous assay readout, (b) mechanistically tied to a
  real developability liability (hydrophobicity-driven aggregation and poor
  purification behaviour), and (c) the only endpoint in FLAb that appears in
  three independent cohorts with the same nominal readout, which gives us
  genuine held-out antibodies rather than a re-shuffle of the training set.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

MODELS_DIR = PROJECT_ROOT / "models_store"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
LOGS_DIR = PROJECT_ROOT / "logs"
REPORT_DIR = PROJECT_ROOT / "report"
DOCS_DIR = PROJECT_ROOT / "docs"

FLAB_DIR = RAW_DIR / "FLAb"

ALL_DIRS = [
    DATA_DIR, RAW_DIR, INTERIM_DIR, PROCESSED_DIR, EXTERNAL_DIR,
    RESULTS_DIR, FIGURES_DIR, TABLES_DIR, PREDICTIONS_DIR,
    MODELS_DIR, CACHE_DIR, LOGS_DIR, REPORT_DIR, DOCS_DIR,
]


def ensure_dirs() -> None:
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Data provenance -- pinned so the study is reproducible
# --------------------------------------------------------------------------

FLAB_REPO_URL = "https://github.com/Graylab/FLAb.git"

# Pinned revision of the FLAb repository.  Recorded at acquisition time on
# 2026-10-04.  Re-running acquire.py checks out exactly this commit, so the
# numbers in the report are reproducible even if FLAb changes upstream.
FLAB_PINNED_COMMIT = "e05e3f0135783ffb7de412df6a1467e7a0031911"

# Only the data/ subtree is checked out.  The full tree contains paths that
# fail to check out on Windows, and we do not need the scoring outputs.
FLAB_SPARSE_PATHS = ["data"]


# --------------------------------------------------------------------------
# Endpoint definitions
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Cohort:
    """One assay x one antibody cohort = one source CSV."""

    key: str                  # short identifier used in filenames
    relpath: str              # path inside the FLAb repo
    value_column: str         # verbatim column name holding the measurement
    units: str
    cohort_name: str          # what population of antibodies this is
    publication: str
    year: int
    favourable_direction: str  # "lower" or "higher" is better, or "none"
    paired: bool = True        # has both heavy and light chains

    # How this cohort may legitimately be used in the evaluation.
    #   "train"               -- model development
    #   "external_validation" -- an independent panel on a comparable scale;
    #                            absolute-error metrics are meaningful
    #   "stress_test"         -- a cohort that violates the training
    #                            distribution so badly that only a weak,
    #                            heavily caveated claim can be made from it
    evaluation_role: str = "external_validation"

    # Right-censoring ceiling, if the assay has one.  Values at exactly this
    # number are not true measurements; they mean "did not elute before the
    # run ended".  Metrics must be computed with this made explicit.
    censoring_value: float | None = None
    censoring_direction: str = "right"


@dataclass(frozen=True)
class Endpoint:
    """A developability endpoint: one training cohort plus external cohorts."""

    key: str
    pretty_name: str
    units: str
    train: Cohort
    externals: tuple[Cohort, ...] = ()
    favourable_direction: str = "lower"
    # Fraction of the TRAINING distribution treated as "high risk" when the
    # continuous measurement is binarised.  Jain et al. 2017 flag an antibody
    # when a value falls in the worst 10% of the approved-antibody
    # distribution; we reproduce that convention on the training cohort and
    # additionally report a more permissive 20% cut as a sensitivity check.
    risk_quantile: float = 0.90
    risk_quantile_sensitivity: tuple[float, ...] = (0.80, 0.90)
    # Absolute threshold from the literature, if one is used instead of the
    # distributional cut.  Set by configure_thresholds() when known.
    risk_threshold_absolute: float | None = None
    # Whether external cohorts are on a comparable ABSOLUTE scale.  When
    # False, only rank-based metrics (Spearman) are treated as primary for
    # that cohort and absolute-error metrics are reported with a caveat.
    external_absolute_comparable: dict[str, bool] = field(default_factory=dict)


# -- Primary endpoint: HIC retention time ----------------------------------

HIC_TRAIN = Cohort(
    key="jain2017_hic",
    relpath="data/aggregation/jain2017biophyscial_HICRT.csv",
    value_column="HIC Retention Time (Min)a",
    units="min",
    cohort_name="Clinical-stage therapeutic antibodies",
    publication="Jain et al. 2017, PNAS, doi:10.1073/pnas.1616408114",
    year=2017,
    favourable_direction="lower",
    evaluation_role="train",
    # The training endpoint is itself mildly right-censored: two antibodies sit
    # at exactly 25.0 min because they exceeded the maximum measurement time
    # and were assigned that value.  They are kept (removing them would bias
    # the tail the model most needs to learn) but flagged, and the headline
    # result is repeated without them as a sensitivity check.
    censoring_value=25.0,
    censoring_direction="right",
)

HIC_EXT_JAIN2024 = Cohort(
    key="jain2024_hic",
    relpath="data/aggregation/jain2024assessment_HIC.csv",
    value_column="HIC RT (min)",
    units="min",
    cohort_name="Germline Fv antibodies",
    publication="Jain et al. 2024, mAbs, doi:10.1080/19420862.2024.2384104",
    year=2024,
    favourable_direction="lower",
)

# Demoted from "external validation" to "stress test" after the data audit.
# Three disqualifying properties, all verified on the raw file:
#   (1) It is not a diverse panel but a point-mutant library of a SINGLE Fv --
#       all 97 heavy chains are 120 aa with pairwise Hamming distance <= 3,
#       and 90 of 97 rows share one identical light chain.
#   (2) It is right-censored: 30 of 97 values sit at exactly 30.00 min, the
#       column run-time ceiling, so a third of the labels are inequalities.
#   (3) It sits on a different absolute scale (median 27.6 min vs 9.9 min for
#       the Jain cohorts), i.e. a different column/gradient.
# It is therefore useless for absolute-error metrics and is reported only as
# a hard out-of-distribution ranking check.
HIC_EXT_JETHA2019 = Cohort(
    key="jetha2019_hic",
    relpath="data/aggregation/jetha2019homology_RT.csv",
    value_column="HIC RT (min)",
    units="min",
    cohort_name="Integrin alpha-11 Fv point-mutant series (single parent)",
    publication="Jetha et al. 2019, mAbs, doi:10.1080/19420862.2018.1475871",
    year=2019,
    favourable_direction="lower",
    evaluation_role="stress_test",
    censoring_value=30.0,
    censoring_direction="right",
)

ENDPOINT_HIC = Endpoint(
    key="hic",
    pretty_name="HIC retention time",
    units="min",
    train=HIC_TRAIN,
    externals=(HIC_EXT_JAIN2024, HIC_EXT_JETHA2019),
    favourable_direction="lower",
    # Published threshold, not one we invented.  Jain et al. 2017 Table 1 sets
    # the HIC warning flag at > 11.7 +/- 0.6 min, derived as the worst 10% of
    # the 48 APPROVED antibodies (not of all 137).  Using the published number
    # rather than a quantile of our training set means the positive class has
    # a fixed clinical meaning and transfers unchanged to any new cohort run
    # on the same assay.
    risk_threshold_absolute=11.7,
    # Different labs run different columns, gradients and salt systems, so
    # absolute minutes are NOT assumed transferable.  Set per cohort after
    # inspecting the distributions in the EDA step.
    external_absolute_comparable={"jain2024_hic": True, "jetha2019_hic": False},
)


# -- Secondary endpoints (modelled independently, never pooled) ------------

ENDPOINT_ACSINS = Endpoint(
    key="acsins",
    pretty_name="AC-SINS self-interaction",
    units="nm",
    train=Cohort(
        key="jain2017_acsins",
        relpath="data/aggregation/jain2017biophysical_ACSINS.csv",
        value_column="Affinity-Capture Self-Interaction Nanoparticle Spectroscopy (AC-SINS) ∆λmax (nm) Average",
        units="nm",
        cohort_name="Clinical-stage therapeutic antibodies",
        publication="Jain et al. 2017, PNAS, doi:10.1073/pnas.1616408114",
        year=2017,
        favourable_direction="lower",
    ),
    externals=(
        Cohort(
            key="jain2024_acsins",
            relpath="data/aggregation/jain2024assessment_ACSINS.csv",
            value_column="AC-SINS ∆λmax (nm)",
            units="nm",
            cohort_name="Germline Fv antibodies",
            publication="Jain et al. 2024, mAbs, doi:10.1080/19420862.2024.2384104",
            year=2024,
            favourable_direction="lower",
        ),
    ),
    favourable_direction="lower",
    external_absolute_comparable={"jain2024_acsins": True},
)

ENDPOINT_TM = Endpoint(
    key="tm",
    pretty_name="Fab melting temperature (DSF)",
    units="degC",
    train=Cohort(
        key="jain2017_tm",
        relpath="data/thermostability/jain2017biophysical_Tm.csv",
        value_column="Fab Tm by DSF (°C)",
        units="degC",
        cohort_name="Clinical-stage therapeutic antibodies",
        publication="Jain et al. 2017, PNAS, doi:10.1073/pnas.1616408114",
        year=2017,
        favourable_direction="higher",
    ),
    externals=(
        Cohort(
            key="jain2024_tm",
            relpath="data/thermostability/jain2024assessment_Tm.csv",
            value_column="Fab Tm by DSF (°C)",
            units="degC",
            cohort_name="Germline Fv antibodies",
            publication="Jain et al. 2024, mAbs, doi:10.1080/19420862.2024.2384104",
            year=2024,
            favourable_direction="higher",
        ),
    ),
    favourable_direction="higher",
    # For a "higher is better" endpoint the risky tail is the LOWER tail;
    # the binarisation code reads favourable_direction and flips accordingly.
    external_absolute_comparable={"jain2024_tm": True},
)

ENDPOINT_PSR = Endpoint(
    key="psr",
    pretty_name="Poly-specificity reagent score",
    units="a.u. (0-1)",
    train=Cohort(
        key="jain2017_psr",
        relpath="data/polyreactivity/jain2017biophysical_PSR.csv",
        value_column="Poly-Specificity Reagent (PSR) SMP Score (0-1)",
        units="a.u.",
        cohort_name="Clinical-stage therapeutic antibodies",
        publication="Jain et al. 2017, PNAS, doi:10.1073/pnas.1616408114",
        year=2017,
        favourable_direction="lower",
    ),
    externals=(
        Cohort(
            key="jain2024_psr",
            relpath="data/polyreactivity/jain2024assessment_PSR.csv",
            value_column="PSR",
            units="a.u.",
            cohort_name="Germline Fv antibodies",
            publication="Jain et al. 2024, mAbs, doi:10.1080/19420862.2024.2384104",
            year=2024,
            favourable_direction="lower",
        ),
    ),
    favourable_direction="lower",
    external_absolute_comparable={"jain2024_psr": True},
)

ENDPOINTS: dict[str, Endpoint] = {
    "hic": ENDPOINT_HIC,
    "acsins": ENDPOINT_ACSINS,
    "tm": ENDPOINT_TM,
    "psr": ENDPOINT_PSR,
}

PRIMARY_ENDPOINT = "hic"
SECONDARY_ENDPOINTS = ["acsins", "tm", "psr"]


# --------------------------------------------------------------------------
# Evaluation protocol
# --------------------------------------------------------------------------

SEED = 20261004

# Clustered cross-validation.  Antibodies are grouped by CDR sequence
# identity so that near-duplicate antibodies cannot straddle the train/test
# boundary.  Whole-chain identity is useless for this because every antibody
# shares ~75-85% framework identity; the discriminating signal is in the CDRs.
CDR_IDENTITY_THRESHOLD = 0.80       # single-linkage cluster cut on CDR identity
CDR_IDENTITY_SENSITIVITY = (0.70, 0.80, 0.90)

CV_N_SPLITS = 5
# 5 repeats x 5 folds = 25 outer fits per (feature set, model).  Ten repeats
# would halve the Monte-Carlo noise on the repeat-to-repeat spread, but the
# dominant uncertainty here is the 137-antibody sample itself -- which the
# bootstrap CI captures and which no amount of re-splitting reduces -- so the
# extra compute buys very little.  The full sweep is 7 feature sets x 7 model
# families x 4 endpoints; at ten repeats it would not finish in a usable time.
CV_N_REPEATS = 5
INNER_CV_N_SPLITS = 3               # for hyper-parameter selection

N_BOOTSTRAP = 2000                  # bootstrap resamples for metric CIs
N_PERMUTATIONS = 1000               # label permutations for the empirical p-value

CONFORMAL_ALPHA = 0.2               # target 80% prediction intervals

# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------

ABLANG2_MODEL = "ablang2-paired"
ESM2_HF_ID = "facebook/esm2_t12_35M_UR50D"
ESM2_FAIR_NAME = "esm2_t12_35M_UR50D"
ESM2_REPR_LAYER = 12                # final layer of the 12-layer 35M model

PLM_BATCH_SIZE = 8
TORCH_DEVICE = os.environ.get("ABDEV_DEVICE", "cpu")

STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"

# Feature sets compared in the headline table.  Keys must match the builders
# registered in src/features/build.py.
FEATURE_SETS = [
    "baseline_trivial",     # length + gravy only; the model that must be beaten
    "desc_global",          # whole-chain physicochemical descriptors
    "desc_cdr",             # CDR-restricted descriptors + liability motifs
    "desc_all",             # desc_global + desc_cdr
    "esm2_35m",             # general protein language model baseline
    "ablang2",              # antibody-specific paired language model
    "desc_all+ablang2",     # fusion
]

# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

FIG_DPI = 300
FIG_FORMAT = "png"
