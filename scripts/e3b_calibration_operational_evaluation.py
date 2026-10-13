from pathlib import Path
import numpy as np
import pandas as pd

from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)

ROOT = Path(r"D:\MissionShield")
RESULTS = ROOT / "results"

MODELS = [
    "gbdt",
    "lstm",
    "transformer",
    "graphtemporal",
    "missionshield",
]
DISPLAY = {
    "gbdt": "GBDT",
    "lstm": "LSTM",
    "transformer": "Transformer",
    "graphtemporal": "GraphTemporal",
    "missionshield": "MissionShield",
}
HORIZONS = [10, 30, 60]

# Pre-specified operating point for this experiment.
# Threshold is selected ONLY on validation negatives.
TARGET_FAR = 0.05
EPS = 1e-6


def ece(y, p, bins=10):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        if i == bins - 1:
            m = (p >= lo) & (p <= hi)
        else:
            m = (p >= lo) & (p < hi)
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def temperature_fit(y, p):
    """
    Validation-only scalar temperature scaling on logits.
    No labels from test are used.
    """
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    z = logit(p)

    def nll(log_temperature):
        T = np.exp(log_temperature)
        q = expit(z / T)
        q = np.clip(q, EPS, 1.0 - EPS)
        return -np.mean(y * np.log(q) + (1.0 - y) * np.log(1.0 - q))

    result = minimize_scalar(
        nll,
        bounds=(np.log(0.05), np.log(20.0)),
        method="bounded",
    )
    return float(np.exp(result.x))


def apply_temperature(p, temperature):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return expit(logit(p) / temperature)


def fixed_far_threshold(y_val, p_val, target_far):
    """
    Choose the lowest threshold whose empirical validation-negative FAR
    does not exceed target_far.

    This uses validation labels/probabilities only.
    """
    y_val = np.asarray(y_val, dtype=int)
    p_val = np.asarray(p_val, dtype=float)

    neg = np.sort(p_val[y_val == 0])[::-1]
    if len(neg) == 0:
        raise RuntimeError("No validation negatives available.")

    candidates = np.r_[np.inf, np.unique(neg)]
    candidates = np.sort(candidates)[::-1]

    best = np.inf
    best_far = 0.0

    # Walk from strict to permissive; retain most permissive threshold
    # that still satisfies the requested FAR.
    for threshold in candidates:
        far = float(np.mean(neg >= threshold))
        if far <= target_far + 1e-12:
            best = threshold
            best_far = far
        else:
            break

    if np.isinf(best):
        best = np.nextafter(neg.max(), np.inf)
        best_far = 0.0

    return float(best), float(best_far)


def classification_metrics(y, p, threshold):
    y = np.asarray(y, dtype=int)
    pred = (np.asarray(p) >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    far = fp / (fp + tn) if (fp + tn) else np.nan
    precision = tp / (tp + fp) if (tp + fp) else np.nan

    return {
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "far": float(far),
        "precision": float(precision),
    }


def event_metrics(df, probability_col, threshold):
    """
    Event-level warning analysis using the frozen manifest metadata.

    For each run represented by positive pre-failure samples:
      - event is considered detected if any positive pre-failure sample
        crosses threshold.
      - warning lead time is max(steps_to_failure) among crossing positive
        samples, i.e. earliest available threshold crossing before failure.

    This does not infer unobserved samples outside the frozen manifest.
    """
    x = df.copy()
    x["alarm"] = x[probability_col].to_numpy() >= threshold

    positive = x[
        (x["label"].astype(int) == 1)
        & x["steps_to_failure"].notna()
    ].copy()

    if positive.empty:
        return {
            "events": 0,
            "detected_events": 0,
            "event_detection_rate": np.nan,
            "median_lead": np.nan,
            "mean_lead": np.nan,
            "p25_lead": np.nan,
            "p75_lead": np.nan,
        }

    event_rows = []

    # Dataset contains at most the relevant next failure per run under the
    # frozen pre-failure protocol; evaluate per run.
    for run_id, g in positive.groupby("run_id"):
        alarms = g[g["alarm"]]
        if len(alarms):
            lead = float(alarms["steps_to_failure"].max())
            detected = 1
        else:
            lead = np.nan
            detected = 0

        event_rows.append((run_id, detected, lead))

    ev = pd.DataFrame(
        event_rows,
        columns=["run_id", "detected", "lead"],
    )

    leads = ev.loc[ev["detected"] == 1, "lead"].dropna()

    return {
        "events": int(len(ev)),
        "detected_events": int(ev["detected"].sum()),
        "event_detection_rate": float(ev["detected"].mean()),
        "median_lead": float(leads.median()) if len(leads) else np.nan,
        "mean_lead": float(leads.mean()) if len(leads) else np.nan,
        "p25_lead": float(leads.quantile(0.25)) if len(leads) else np.nan,
        "p75_lead": float(leads.quantile(0.75)) if len(leads) else np.nan,
    }


def load_pair(model, H):
    val = pd.read_csv(
        RESULTS / f"e3a_validation_{model}_H{H}.csv"
    )

    if model == "missionshield":
        test_path = RESULTS / f"e1_missionshield_H{H}_predictions.csv"
    else:
        test_path = RESULTS / f"e2_frozen_{model}_H{H}_predictions.csv"

    test = pd.read_csv(test_path)

    required = {
        "sample_id", "run_id", "time", "attack_family",
        "horizon", "steps_to_failure", "label", "probability"
    }

    for name, frame in [("validation", val), ("test", test)]:
        missing = required - set(frame.columns)
        if missing:
            raise RuntimeError(
                f"{model} H{H} {name} missing columns: {sorted(missing)}"
            )

    return val, test


def main():
    print("=" * 88)
    print("MissionShield E3-B | VALIDATION-ONLY CALIBRATION + FIXED-FAR EVALUATION")
    print(f"Pre-specified target validation FAR = {TARGET_FAR:.1%}")
    print("=" * 88)

    summary = []
    event_rows = []

    for H in HORIZONS:
        print("\n" + "=" * 88)
        print(f"HORIZON H={H}")
        print("=" * 88)

        for model in MODELS:
            name = DISPLAY[model]
            val, test = load_pair(model, H)

            yv = val["label"].to_numpy(dtype=int)
            pv_raw = val["probability"].to_numpy(dtype=float)
            yt = test["label"].to_numpy(dtype=int)
            pt_raw = test["probability"].to_numpy(dtype=float)

            # Fit calibration using validation only.
            temperature = temperature_fit(yv, pv_raw)
            pv = apply_temperature(pv_raw, temperature)
            pt = apply_temperature(pt_raw, temperature)

            # Choose operating threshold using calibrated validation scores only.
            threshold, val_far = fixed_far_threshold(
                yv, pv, TARGET_FAR
            )

            op = classification_metrics(yt, pt, threshold)

            raw_auroc = roc_auc_score(yt, pt_raw)
            raw_auprc = average_precision_score(yt, pt_raw)
            calibrated_auroc = roc_auc_score(yt, pt)
            calibrated_auprc = average_precision_score(yt, pt)

            # Ranking metrics should be invariant to temperature scaling,
            # apart from tiny floating-point effects.
            if abs(raw_auroc - calibrated_auroc) > 1e-10:
                raise RuntimeError("AUROC changed unexpectedly after temperature scaling.")
            if abs(raw_auprc - calibrated_auprc) > 1e-10:
                raise RuntimeError("AUPRC changed unexpectedly after temperature scaling.")

            raw_ece = ece(yt, pt_raw)
            cal_ece = ece(yt, pt)
            raw_brier = brier_score_loss(yt, pt_raw)
            cal_brier = brier_score_loss(yt, pt)

            test_eval = test.copy()
            test_eval["probability_calibrated"] = pt
            test_eval["alarm_fixed_far"] = (pt >= threshold).astype(int)

            ev = event_metrics(
                test_eval,
                "probability_calibrated",
                threshold,
            )

            out_pred = RESULTS / f"e3b_{model}_H{H}_operational_predictions.csv"
            test_eval.to_csv(out_pred, index=False)

            summary.append({
                "model": name,
                "horizon": H,
                "temperature": temperature,
                "target_validation_far": TARGET_FAR,
                "validation_far": val_far,
                "threshold_calibrated": threshold,
                "test_positive_rate": float(yt.mean()),
                "auroc": float(raw_auroc),
                "auprc": float(raw_auprc),
                "raw_ece": float(raw_ece),
                "calibrated_ece": float(cal_ece),
                "raw_brier": float(raw_brier),
                "calibrated_brier": float(cal_brier),
                **op,
                **ev,
            })

            event_rows.append({
                "model": name,
                "horizon": H,
                **ev,
            })

            print(
                f"{name:14s} | "
                f"T={temperature:6.3f} | "
                f"thr={threshold:7.4f} | "
                f"valFAR={val_far:6.3f} | "
                f"testFAR={op['far']:6.3f} | "
                f"sens={op['sensitivity']:6.3f} | "
                f"eventDR={ev['event_detection_rate']:6.3f} | "
                f"medianLead={ev['median_lead']}"
            )

    summary_df = pd.DataFrame(summary)
    event_df = pd.DataFrame(event_rows)

    summary_file = RESULTS / "e3b_operational_summary.csv"
    event_file = RESULTS / "e3b_event_leadtime_summary.csv"

    summary_df.to_csv(summary_file, index=False)
    event_df.to_csv(event_file, index=False)

    print("\n" + "=" * 88)
    print("E3-B SUMMARY")
    print("=" * 88)

    cols = [
        "model", "horizon",
        "auroc", "auprc",
        "raw_ece", "calibrated_ece",
        "raw_brier", "calibrated_brier",
        "validation_far", "far", "sensitivity",
        "event_detection_rate", "median_lead", "mean_lead",
    ]

    print(summary_df[cols].to_string(index=False))

    print(f"\nSaved: {summary_file}")
    print(f"Saved: {event_file}")
    print("\nE3-B COMPLETE")


if __name__ == "__main__":
    main()
