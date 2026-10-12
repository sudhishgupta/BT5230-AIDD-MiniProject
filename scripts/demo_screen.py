"""
End-to-end demonstration of the screening tool.

Builds an example input CSV from the held-out Jain 2024 cohort -- antibodies
the model has never seen, but for which we have ground-truth measurements --
runs the screening tool on it, and prints how the predictions and risk calls
compare with the truth.

Using a cohort with known answers makes this a demonstration that can be
checked, rather than a screenshot of plausible-looking numbers.

    python scripts/demo_screen.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config                       # noqa: E402
from src.tool.screen import screen           # noqa: E402


def main() -> int:
    config.ensure_dirs()
    endpoint = config.PRIMARY_ENDPOINT
    ep = config.ENDPOINTS[endpoint]

    ds = pd.read_csv(config.PROCESSED_DIR / f"dataset_{endpoint}.csv")
    ext_key = ep.externals[0].key
    held_out = ds[ds["cohort"] == ext_key].reset_index(drop=True)
    if held_out.empty:
        print(f"No rows for cohort {ext_key}; run the pipeline first.")
        return 1

    example = held_out[["antibody_id", "heavy_chain", "light_chain"]].copy()
    example_path = config.EXTERNAL_DIR / "example_candidates.csv"
    example.to_csv(example_path, index=False, encoding="utf-8")
    print(f"[demo] wrote {len(example)} example candidates -> {example_path}\n")

    out = screen(
        input_csv=example_path,
        endpoint=endpoint,
        output_csv=config.RESULTS_DIR / "demo_screening_results.csv",
        report_md=config.RESULTS_DIR / "demo_screening_report.md",
    )

    truth = held_out.set_index("antibody_id")[["value", "risk_label"]]
    merged = out.set_index("antibody_id").join(truth)

    from scipy import stats
    rho = stats.spearmanr(merged["predicted_value"], merged["value"]).statistic
    mae = float(np.mean(np.abs(merged["predicted_value"] - merged["value"])))
    covered = float(np.mean((merged["value"] >= merged["pi_low"]) &
                            (merged["value"] <= merged["pi_high"])))

    print("\n" + "=" * 72)
    print(f"DEMO RESULT on the held-out {ext_key} cohort (n={len(merged)})")
    print("=" * 72)
    print(f"  Spearman (predicted vs measured) : {rho:+.3f}")
    print(f"  MAE                              : {mae:.2f} {ep.units}")
    print(f"  Prediction-interval coverage     : {covered:.0%} "
          f"(nominal {int((1 - config.CONFORMAL_ALPHA) * 100)}%)")

    print("\n  Risk-call breakdown vs truth:")
    xt = pd.crosstab(merged["risk_call"], merged["risk_label"])
    print(xt.rename(columns={0: "truly low risk", 1: "truly high risk"})
            .to_string().replace("\n", "\n    "))

    print("\n  Applicability domain:")
    for band, n in merged["applicability"].value_counts().items():
        print(f"    {band:<16} {n}")

    print("\n  Top 5 safest candidates by prediction:")
    cols = ["predicted_value", "pi_low", "pi_high", "risk_call", "value"]
    head = merged.sort_values("predicted_value")[cols].head(5)
    for aid, r in head.iterrows():
        print(f"    {aid:<22} pred={r['predicted_value']:5.2f} "
              f"[{r['pi_low']:5.2f},{r['pi_high']:5.2f}] "
              f"{r['risk_call']:<10} measured={r['value']:5.2f}")

    print("\n  Top 5 riskiest candidates by prediction:")
    tail = merged.sort_values("predicted_value", ascending=False)[cols].head(5)
    for aid, r in tail.iterrows():
        print(f"    {aid:<22} pred={r['predicted_value']:5.2f} "
              f"[{r['pi_low']:5.2f},{r['pi_high']:5.2f}] "
              f"{r['risk_call']:<10} measured={r['value']:5.2f}")

    print("\n  Full results : results/demo_screening_results.csv")
    print("  Report       : results/demo_screening_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
