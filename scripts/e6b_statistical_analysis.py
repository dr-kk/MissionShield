from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(r"D:\MissionShield")
RESULTS = ROOT / "results"

INPUT = RESULTS / "e6a_multiseed_results.csv"

NEURAL_MODELS = ["MissionShield", "LSTM", "Transformer", "GraphTemporal"]
BASELINES = ["LSTM", "Transformer", "GraphTemporal"]
HORIZONS = [10, 30, 60]
METRICS = ["auroc", "auprc"]

# Fixed before looking at E6-B significance results.
BOOTSTRAP_REPS = 20000
BOOTSTRAP_SEED = 2606
ALPHA = 0.05


def bootstrap_mean_ci(values, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED):
    """
    Nonparametric percentile bootstrap CI for the mean across training seeds.
    Seeds are the replicate unit.
    """
    x = np.asarray(values, dtype=float)
    if len(x) < 2:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(reps, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def bootstrap_paired_diff_ci(ms, baseline, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED):
    """
    95% percentile bootstrap CI for paired seed-wise difference:
    MissionShield - baseline.
    """
    d = np.asarray(ms, dtype=float) - np.asarray(baseline, dtype=float)
    if len(d) < 2:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(reps, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def holm_adjust(pvalues):
    """
    Holm family-wise error correction.
    Returns adjusted p-values in original order.
    """
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)

    running = 0.0
    for rank, idx in enumerate(order):
        candidate = (m - rank) * p[idx]
        running = max(running, candidate)
        adjusted[idx] = min(running, 1.0)

    return adjusted


def paired_wilcoxon(ms, baseline):
    """
    Two-sided paired Wilcoxon signed-rank test across matched training seeds.
    If all differences are exactly zero, p=1.
    """
    d = np.asarray(ms, dtype=float) - np.asarray(baseline, dtype=float)

    if np.allclose(d, 0.0):
        return 0.0, 1.0

    stat, p = wilcoxon(
        d,
        alternative="two-sided",
        zero_method="wilcox",
        correction=False,
        method="auto",
    )
    return float(stat), float(p)


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    df = pd.read_csv(INPUT)

    required = {"model", "horizon", "seed", "auroc", "auprc"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")

    print("=" * 104)
    print("MissionShield E6-B | MULTI-SEED STATISTICAL ANALYSIS")
    print("10-seed neural comparison | bootstrap 95% CIs | paired Wilcoxon | Holm correction")
    print("=" * 104)

    # ------------------------------------------------------------------
    # Integrity checks
    # ------------------------------------------------------------------
    neural = df[df["model"].isin(NEURAL_MODELS)].copy()
    gbdt = df[df["model"] == "GBDT"].copy()

    expected_seeds = None
    for H in HORIZONS:
        for model in NEURAL_MODELS:
            g = neural[(neural["horizon"] == H) & (neural["model"] == model)]
            seeds = sorted(g["seed"].astype(int).tolist())

            if len(seeds) != 10 or len(set(seeds)) != 10:
                raise RuntimeError(
                    f"Expected 10 unique seeds for {model}, H={H}; got {seeds}"
                )

            if expected_seeds is None:
                expected_seeds = seeds
            elif seeds != expected_seeds:
                raise RuntimeError(
                    f"Paired-seed mismatch for {model}, H={H}: {seeds}"
                )

    print("Paired neural seeds:", expected_seeds)
    print("Integrity check: PASS")

    # ------------------------------------------------------------------
    # Table 1: descriptive statistics + bootstrap CI
    # ------------------------------------------------------------------
    summary_rows = []

    for H in HORIZONS:
        for model in NEURAL_MODELS:
            g = neural[
                (neural["horizon"] == H) &
                (neural["model"] == model)
            ].sort_values("seed")

            row = {
                "horizon": H,
                "model": model,
                "n_seeds": len(g),
            }

            for metric in METRICS:
                x = g[metric].to_numpy(float)
                lo, hi = bootstrap_mean_ci(
                    x,
                    seed=BOOTSTRAP_SEED + H + (0 if metric == "auroc" else 100)
                )
                row[f"{metric}_mean"] = float(x.mean())
                row[f"{metric}_sd"] = float(x.std(ddof=1))
                row[f"{metric}_ci95_low"] = lo
                row[f"{metric}_ci95_high"] = hi

            summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    # Add deterministic GBDT reference without fake CI.
    gbdt_rows = []
    for H in HORIZONS:
        g = gbdt[gbdt["horizon"] == H]
        if len(g) != 1:
            raise RuntimeError(
                f"Expected exactly one deterministic GBDT row for H={H}; found {len(g)}"
            )

        r = g.iloc[0]
        gbdt_rows.append({
            "horizon": H,
            "model": "GBDT",
            "n_seeds": 1,
            "auroc_mean": float(r["auroc"]),
            "auroc_sd": np.nan,
            "auroc_ci95_low": np.nan,
            "auroc_ci95_high": np.nan,
            "auprc_mean": float(r["auprc"]),
            "auprc_sd": np.nan,
            "auprc_ci95_low": np.nan,
            "auprc_ci95_high": np.nan,
        })

    summary_all = pd.concat(
        [summary, pd.DataFrame(gbdt_rows)],
        ignore_index=True
    ).sort_values(["horizon", "model"])

    # ------------------------------------------------------------------
    # Table 2: paired MissionShield comparisons
    # ------------------------------------------------------------------
    comparison_rows = []

    for H in HORIZONS:
        ms = (
            neural[
                (neural["horizon"] == H) &
                (neural["model"] == "MissionShield")
            ]
            .sort_values("seed")
            .set_index("seed")
        )

        for baseline in BASELINES:
            b = (
                neural[
                    (neural["horizon"] == H) &
                    (neural["model"] == baseline)
                ]
                .sort_values("seed")
                .set_index("seed")
            )

            common = ms.index.intersection(b.index)
            if len(common) != 10:
                raise RuntimeError(
                    f"Expected 10 paired seeds: H={H}, baseline={baseline}"
                )

            for metric in METRICS:
                a = ms.loc[common, metric].to_numpy(float)
                c = b.loc[common, metric].to_numpy(float)
                diff = a - c

                stat, p = paired_wilcoxon(a, c)
                lo, hi = bootstrap_paired_diff_ci(
                    a, c,
                    seed=BOOTSTRAP_SEED + H +
                         (0 if metric == "auroc" else 100) +
                         1000 * (BASELINES.index(baseline) + 1)
                )

                tol = 1e-12
                wins = int(np.sum(diff > tol))
                ties = int(np.sum(np.abs(diff) <= tol))
                losses = int(np.sum(diff < -tol))

                comparison_rows.append({
                    "horizon": H,
                    "metric": metric,
                    "comparison": f"MissionShield vs {baseline}",
                    "baseline": baseline,
                    "n_pairs": len(common),
                    "missionshield_mean": float(a.mean()),
                    "baseline_mean": float(c.mean()),
                    "mean_difference_ms_minus_baseline": float(diff.mean()),
                    "median_difference_ms_minus_baseline": float(np.median(diff)),
                    "diff_ci95_low": lo,
                    "diff_ci95_high": hi,
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "wilcoxon_statistic": stat,
                    "p_raw": p,
                })

    comparisons = pd.DataFrame(comparison_rows)

    # One predeclared Holm family across all 18 neural comparisons:
    # 3 horizons x 2 metrics x 3 baselines.
    comparisons["p_holm"] = holm_adjust(comparisons["p_raw"].to_numpy())
    comparisons["significant_holm_0.05"] = comparisons["p_holm"] < ALPHA

    def direction(row):
        d = row["mean_difference_ms_minus_baseline"]
        if d > 0:
            return "MissionShield higher"
        elif d < 0:
            return "MissionShield lower"
        return "tie"

    comparisons["direction"] = comparisons.apply(direction, axis=1)

    # ------------------------------------------------------------------
    # Compact manuscript table
    # ------------------------------------------------------------------
    manuscript_rows = []
    order = ["MissionShield", "LSTM", "Transformer", "GraphTemporal", "GBDT"]

    for H in HORIZONS:
        for model in order:
            r = summary_all[
                (summary_all["horizon"] == H) &
                (summary_all["model"] == model)
            ].iloc[0]

            if model == "GBDT":
                auroc_txt = f'{r["auroc_mean"]:.4f} (det.)'
                auprc_txt = f'{r["auprc_mean"]:.4f} (det.)'
            else:
                auroc_txt = (
                    f'{r["auroc_mean"]:.4f} '
                    f'[{r["auroc_ci95_low"]:.4f}, {r["auroc_ci95_high"]:.4f}]'
                )
                auprc_txt = (
                    f'{r["auprc_mean"]:.4f} '
                    f'[{r["auprc_ci95_low"]:.4f}, {r["auprc_ci95_high"]:.4f}]'
                )

            manuscript_rows.append({
                "horizon": H,
                "model": model,
                "AUROC mean [95% CI]": auroc_txt,
                "AUPRC mean [95% CI]": auprc_txt,
            })

    manuscript = pd.DataFrame(manuscript_rows)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    p1 = RESULTS / "e6b_multiseed_summary_ci.csv"
    p2 = RESULTS / "e6b_paired_wilcoxon_holm.csv"
    p3 = RESULTS / "e6b_manuscript_table.csv"

    summary_all.to_csv(p1, index=False)
    comparisons.to_csv(p2, index=False)
    manuscript.to_csv(p3, index=False)

    print("\nMULTI-SEED SUMMARY")
    print(manuscript.to_string(index=False))

    print("\nPAIRED MISSIONShield COMPARISONS")
    show = comparisons[
        [
            "horizon", "metric", "baseline",
            "mean_difference_ms_minus_baseline",
            "diff_ci95_low", "diff_ci95_high",
            "wins", "ties", "losses",
            "p_raw", "p_holm",
            "significant_holm_0.05", "direction",
        ]
    ]
    print(show.to_string(index=False))

    print("\nSaved:")
    print(" ", p1)
    print(" ", p2)
    print(" ", p3)

    print("\nE6-B COMPLETE")
    print(
        "Interpret Holm-significant results together with effect direction and "
        "paired-difference CI. GBDT is descriptive only."
    )


if __name__ == "__main__":
    main()
