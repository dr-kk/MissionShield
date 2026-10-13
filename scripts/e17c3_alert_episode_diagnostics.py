from pathlib import Path
import re
import pandas as pd

ROOT = Path("results/e12")
OUT = Path("results/e17")
OUT.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.50

files = (
    list((ROOT / "predictions").glob("*_predictions.csv"))
    + list((ROOT / "e12e_predictions").glob("*_predictions.csv"))
)

rows = []

for path in sorted(files):
    df = pd.read_csv(path)

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
        raise ValueError(path.name)

    match = re.search(r"seed[_-]?(\d+)", name)
    assert match, path.name

    seed = int(match.group(1))
    family = str(df["attack_family"].iloc[0])
    horizon = int(df["horizon"].iloc[0])

    df["time"] = pd.to_numeric(df["time"])
    df["label"] = pd.to_numeric(df["label"])
    df["probability"] = pd.to_numeric(df["probability"])

    total_false_episodes = 0
    total_alert_episodes = 0
    missions_with_false_episodes = 0

    for run_id, run in df.groupby("run_id"):
        run = run.sort_values("time").copy()

        alert = run["probability"].ge(THRESHOLD)
        negative = run["label"].eq(0)

        # A new alert episode begins when the preceding
        # time step is absent or was not alerting.
        consecutive = run["time"].diff().eq(1)

        alert_start = alert & (
            ~alert.shift(fill_value=False) | ~consecutive
        )

        total_alert_episodes += int(alert_start.sum())

        # False-alert episodes are contiguous alerting
        # negative-label windows. This is a diagnostic
        # definition, not a fully adjudicated alarm episode.
        false_window = alert & negative

        false_start = false_window & (
            ~false_window.shift(fill_value=False) | ~consecutive
        )

        count = int(false_start.sum())

        total_false_episodes += count
        missions_with_false_episodes += int(count > 0)

    n_missions = df["run_id"].nunique()

    rows.append({
        "model": model,
        "family": family,
        "horizon": horizon,
        "seed": seed,
        "threshold": THRESHOLD,
        "missions": n_missions,
        "alert_episodes": total_alert_episodes,
        "negative_window_alert_runs": total_false_episodes,
        "negative_window_alert_runs_per_mission":
            total_false_episodes / n_missions,
        "missions_with_negative_window_alert_runs":
            missions_with_false_episodes
    })

result = pd.DataFrame(rows)

assert len(result) == 360, len(result)
assert result["missions"].eq(5).all()

result.to_csv(
    OUT / "e17c3_alert_episode_diagnostics.csv",
    index=False
)

print("\n=== E17-C3 ALERT EPISODE DIAGNOSTICS ===")

print(
    result.groupby("model")[
        [
            "alert_episodes",
            "negative_window_alert_runs_per_mission",
            "missions_with_negative_window_alert_runs"
        ]
    ].mean().round(4).to_string()
)

print("\n=== SERVICE DEGRADATION H60 ===")

print(
    result[
        (result.family == "service_degradation")
        & (result.horizon == 60)
    ].groupby("model")[
        [
            "alert_episodes",
            "negative_window_alert_runs_per_mission",
            "missions_with_negative_window_alert_runs"
        ]
    ].mean().round(4).to_string()
)

print("\nSaved E17-C3 diagnostic output.")
