from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(r"D:\MissionShield")
RESULTS = ROOT / "results"

MODELS = ["gbdt", "lstm", "transformer", "graphtemporal", "missionshield"]
DISPLAY = {
    "gbdt": "GBDT",
    "lstm": "LSTM",
    "transformer": "Transformer",
    "graphtemporal": "GraphTemporal",
    "missionshield": "MissionShield",
}
HORIZONS = [10, 30, 60]


def safe_auc(y, p):
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, p))


def safe_auprc(y, p):
    y = np.asarray(y, dtype=int)
    if y.sum() == 0:
        return np.nan
    return float(average_precision_score(y, p))


def event_stats(g):
    """
    Event-level quantities within one attack family.

    Positive pre-failure samples are grouped by run_id. A run is detected
    if at least one positive sample crosses the already-frozen operational
    alarm threshold. Lead time is the earliest available warning before
    failure, represented by maximum steps_to_failure among alarmed positive
    samples in that run.
    """
    pos = g[(g["label"] == 1) & g["steps_to_failure"].notna()].copy()

    if pos.empty:
        return 0, 0, np.nan, np.nan, np.nan

    detected = 0
    leads = []

    for _, run in pos.groupby("run_id"):
        alarmed = run[run["alarm_fixed_far"] == 1]
        if len(alarmed):
            detected += 1
            leads.append(float(alarmed["steps_to_failure"].max()))

    n_events = int(pos["run_id"].nunique())
    event_dr = detected / n_events if n_events else np.nan

    return (
        n_events,
        int(detected),
        float(event_dr),
        float(np.median(leads)) if leads else np.nan,
        float(np.mean(leads)) if leads else np.nan,
    )


def evaluate_group(model, H, family, g):
    y = g["label"].to_numpy(dtype=int)
    p = g["probability_calibrated"].to_numpy(dtype=float)
    alarm = g["alarm_fixed_far"].to_numpy(dtype=int)

    positives = int(y.sum())
    negatives = int(len(y) - positives)

    tp = int(((alarm == 1) & (y == 1)).sum())
    fp = int(((alarm == 1) & (y == 0)).sum())
    tn = int(((alarm == 0) & (y == 0)).sum())
    fn = int(((alarm == 0) & (y == 1)).sum())

    sensitivity = tp / positives if positives else np.nan
    far = fp / negatives if negatives else np.nan
    specificity = tn / negatives if negatives else np.nan
    precision = tp / (tp + fp) if (tp + fp) else np.nan

    events, detected_events, event_dr, median_lead, mean_lead = event_stats(g)

    return {
        "model": DISPLAY[model],
        "horizon": H,
        "attack_family": family,
        "samples": int(len(g)),
        "runs": int(g["run_id"].nunique()),
        "positives": positives,
        "negatives": negatives,
        "positive_rate": float(y.mean()) if len(y) else np.nan,
        "auroc": safe_auc(y, p),
        "auprc": safe_auprc(y, p),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "sensitivity": float(sensitivity),
        "far": float(far),
        "specificity": float(specificity),
        "precision": float(precision),
        "events": events,
        "detected_events": detected_events,
        "event_detection_rate": event_dr,
        "median_lead": median_lead,
        "mean_lead": mean_lead,
    }


def main():
    print("=" * 96)
    print("MissionShield E4-A | FROZEN TEST ATTACK-FAMILY ROBUSTNESS")
    print("NO RETRAINING | NO RETUNING | E3-B FIXED OPERATING DECISIONS")
    print("=" * 96)

    rows = []

    for H in HORIZONS:
        print("\n" + "=" * 96)
        print(f"HORIZON H={H}")
        print("=" * 96)

        for model in MODELS:
            path = RESULTS / f"e3b_{model}_H{H}_operational_predictions.csv"
            if not path.exists():
                raise FileNotFoundError(f"Missing E3-B file: {path}")

            df = pd.read_csv(path)

            required = {
                "run_id", "attack_family", "label", "steps_to_failure",
                "probability_calibrated", "alarm_fixed_far"
            }
            missing = required - set(df.columns)
            if missing:
                raise RuntimeError(
                    f"{DISPLAY[model]} H{H}: missing columns {sorted(missing)}"
                )

            df["label"] = df["label"].astype(int)
            df["alarm_fixed_far"] = df["alarm_fixed_far"].astype(int)

            print(f"\n{DISPLAY[model]}")

            for family, g in df.groupby("attack_family", sort=True):
                r = evaluate_group(model, H, str(family), g)
                rows.append(r)

                auc_text = "NA" if np.isnan(r["auroc"]) else f'{r["auroc"]:.3f}'
                ap_text = "NA" if np.isnan(r["auprc"]) else f'{r["auprc"]:.3f}'
                sens_text = "NA" if np.isnan(r["sensitivity"]) else f'{r["sensitivity"]:.3f}'
                far_text = "NA" if np.isnan(r["far"]) else f'{r["far"]:.3f}'
                edr_text = (
                    "NA" if np.isnan(r["event_detection_rate"])
                    else f'{r["event_detection_rate"]:.3f}'
                )

                print(
                    f"  {str(family):22s} "
                    f"n={r['samples']:4d} pos={r['positives']:3d} | "
                    f"AUC={auc_text:>5s} AP={ap_text:>5s} | "
                    f"sens={sens_text:>5s} FAR={far_text:>5s} | "
                    f"eventDR={edr_text:>5s}"
                )

    result = pd.DataFrame(rows)
    result_file = RESULTS / "e4a_attack_family_robustness.csv"
    result.to_csv(result_file, index=False)

    # Compact MissionShield-only table for quick inspection.
    ms = result[result["model"] == "MissionShield"].copy()
    ms_file = RESULTS / "e4a_missionshield_attack_family_summary.csv"
    ms.to_csv(ms_file, index=False)

    print("\n" + "=" * 96)
    print("MISSIONShield ATTACK-FAMILY SUMMARY")
    print("=" * 96)

    cols = [
        "horizon", "attack_family", "samples", "positives",
        "auroc", "auprc", "sensitivity", "far",
        "events", "event_detection_rate", "median_lead"
    ]
    print(ms[cols].to_string(index=False))

    print(f"\nSaved: {result_file}")
    print(f"Saved: {ms_file}")
    print("\nE4-A COMPLETE")
    print("Undefined AUROC/AUPRC values are left as NaN when a family has only one class.")


if __name__ == "__main__":
    main()
