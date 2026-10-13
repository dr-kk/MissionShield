from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("results/e12")
OUT = Path("results/e17")
OUT.mkdir(parents=True, exist_ok=True)

files = (
    list((ROOT / "predictions").glob("*_predictions.csv"))
    + list((ROOT / "e12e_predictions").glob("*_predictions.csv"))
)

THRESHOLD = 0.50

rows = []
events_out = []

for path in sorted(files):
    df = pd.read_csv(path)

    required = [
        "run_id", "time", "attack_family",
        "horizon", "label",
        "steps_to_failure", "probability"
    ]
    assert all(c in df.columns for c in required), path

    name = path.stem.lower()

    if "missionshield" in name:
        model = "MissionShield"
    elif "graphwave" in name:
        model = "graphwave"
    elif "lstm" in name:
        model = "lstm"
    elif "patch" in name:
        model = "patch"
    else:
        raise ValueError(f"Unknown model: {path.name}")

    import re
    match = re.search(r"seed[_-]?(\d+)", name)
    if not match:
        raise ValueError(f"Seed not found: {path.name}")

    seed = int(match.group(1))
    family = str(df["attack_family"].iloc[0])
    horizon = int(df["horizon"].iloc[0])

    df["alert"] = (
        pd.to_numeric(df["probability"]) >= THRESHOLD
    )

    positive = df["label"].eq(1)
    negative = df["label"].eq(0)

    tp = int((df["alert"] & positive).sum())
    fp = int((df["alert"] & negative).sum())

    positive_n = int(positive.sum())
    negative_n = int(negative.sum())

    false_alert_runs = int(
        df.loc[df["alert"] & negative, "run_id"].nunique()
    )

    pos = df.loc[positive].copy()
    pos["failure_time"] = (
        pd.to_numeric(pos["time"])
        + pd.to_numeric(pos["steps_to_failure"])
    )

    event_groups = pos.groupby(
        ["run_id", "failure_time"]
    )

    detected_events = 0
    event_leads = []

    for (run_id, failure_time), event in event_groups:
        alerts = event[event["alert"]]

        detected = len(alerts) > 0
        lead = (
            float(alerts["steps_to_failure"].max())
            if detected else np.nan
        )

        detected_events += int(detected)

        if detected:
            event_leads.append(lead)

        events_out.append({
            "model": model,
            "family": family,
            "horizon": horizon,
            "seed": seed,
            "run_id": run_id,
            "failure_time": failure_time,
            "detected": int(detected),
            "first_alert_lead_steps": lead
        })

    rows.append({
        "model": model,
        "family": family,
        "horizon": horizon,
        "seed": seed,
        "threshold": THRESHOLD,
        "positive_windows": positive_n,
        "negative_windows": negative_n,
        "positive_window_recall": (
            tp / positive_n if positive_n else np.nan
        ),
        "false_alerts_per_100_negative_windows": (
            100 * fp / negative_n if negative_n else np.nan
        ),
        "runs_with_false_alerts": false_alert_runs,
        "represented_events": len(event_groups),
        "detected_events": detected_events,
        "event_detection_rate": (
            detected_events / len(event_groups)
            if len(event_groups) else np.nan
        ),
        "mean_first_alert_lead_steps": (
            np.mean(event_leads)
            if event_leads else np.nan
        )
    })

summary = pd.DataFrame(rows)
events = pd.DataFrame(events_out)

assert len(summary) == 360, len(summary)

summary.to_csv(
    OUT / "e17c1_operational_seed_metrics.csv",
    index=False
)
events.to_csv(
    OUT / "e17c1_event_detection.csv",
    index=False
)

print("\n=== E17-C1 OPERATIONAL SUMMARY ===")

print(
    summary.groupby("model")[
        [
            "positive_window_recall",
            "false_alerts_per_100_negative_windows",
            "event_detection_rate",
            "mean_first_alert_lead_steps"
        ]
    ].mean().round(4).to_string()
)

print("\n=== SERVICE DEGRADATION H60 ===")
print(
    summary[
        (summary["family"] == "service_degradation")
        & (summary["horizon"] == 60)
    ].groupby("model")[
        [
            "positive_window_recall",
            "false_alerts_per_100_negative_windows",
            "event_detection_rate",
            "mean_first_alert_lead_steps"
        ]
    ].mean().round(4).to_string()
)

print("\nSaved E17-C1 results. Frozen E1-E16 untouched.")
