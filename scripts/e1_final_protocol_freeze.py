from pathlib import Path
import json
import numpy as np
import pandas as pd


# ============================================================
# MissionShield E1-FINAL
# Protocol Freeze
#
# Purpose:
#   1. Same frozen run split
#   2. HISTORY=10 available
#   3. Entire history pre-failure
#   4. Current mission operational
#   5. Complete future H-step observation available
#   6. Recompute future-failure labels from failure trajectory
#   7. Save immutable sample IDs for all final models
#
# NO MODEL TRAINING IN THIS SCRIPT.
# ============================================================

ROOT = Path(r"D:\MissionShield")

DATA = (
    ROOT / "data" / "processed" /
    "missionshield_e0_v2_dataset.csv"
)

SPLIT_FILE = (
    ROOT / "results" /
    "e1_v2_prefailure_run_split.json"
)

RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

HISTORY = 10
HORIZONS = [10, 30, 60]


# ============================================================
# Load one system-level row per run/time
# ============================================================

def build_timeline(df):

    required = [
        "run_id",
        "time",
        "attack_family",
        "failure",
        "mission_effectiveness",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing columns: {missing}"
        )

    timeline = (
        df[required]
        .drop_duplicates(
            ["run_id", "time"]
        )
        .copy()
    )

    # Verify node rows agree on ground truth
    check = (
        df.groupby(
            ["run_id", "time"]
        )["failure"]
        .nunique()
    )

    if check.max() != 1:
        raise RuntimeError(
            "Failure ground truth differs "
            "between nodes at same timestamp."
        )

    timeline = timeline.sort_values(
        ["run_id", "time"]
    ).reset_index(drop=True)

    return timeline


# ============================================================
# Validate each run timeline
# ============================================================

def validate_runs(timeline):

    print("\nValidating run timelines...")

    lengths = []

    for run_id, g in timeline.groupby(
        "run_id"
    ):

        g = g.sort_values("time")

        times = g["time"].to_numpy()

        if len(np.unique(times)) != len(times):
            raise RuntimeError(
                f"Duplicate time in run {run_id}"
            )

        # Require contiguous unit-step timeline.
        if len(times) > 1:

            delta = np.diff(times)

            if not np.all(delta == 1):
                raise RuntimeError(
                    f"Non-contiguous timeline "
                    f"in run {run_id}"
                )

        lengths.append(len(g))

    print(
        f"Runs validated : "
        f"{timeline.run_id.nunique()}"
    )

    print(
        f"Min run length : "
        f"{min(lengths)}"
    )

    print(
        f"Max run length : "
        f"{max(lengths)}"
    )


# ============================================================
# Build eligible samples for a horizon
#
# IMPORTANT:
#
# At current index i:
#
# History:
#   i-HISTORY+1 ... i
#
# Future forecast window:
#   i+1 ... i+H
#
# Requirements:
#   - full history exists
#   - all history failure == 0
#   - full H future steps exist
#
# Target:
#   1 if ANY failure occurs in i+1 ... i+H
# ============================================================

def build_horizon_samples(
    timeline,
    horizon,
):

    records = []

    excluded_no_history = 0
    excluded_failed_history = 0
    excluded_censored = 0

    for run_id, g in timeline.groupby(
        "run_id"
    ):

        g = (
            g.sort_values("time")
            .reset_index(drop=True)
        )

        failures = (
            g["failure"]
            .to_numpy(dtype=int)
        )

        times = (
            g["time"]
            .to_numpy(dtype=int)
        )

        mission = (
            g["mission_effectiveness"]
            .to_numpy(dtype=float)
        )

        family = (
            g["attack_family"]
            .iloc[0]
        )

        n = len(g)

        for i in range(n):

            # --------------------------------------------
            # Full temporal history required
            # --------------------------------------------

            history_start = (
                i - HISTORY + 1
            )

            if history_start < 0:

                excluded_no_history += 1
                continue

            # --------------------------------------------
            # Strict pre-failure input history
            # --------------------------------------------

            history_failure = (
                failures[
                    history_start:i + 1
                ]
            )

            if np.any(
                history_failure != 0
            ):

                excluded_failed_history += 1
                continue

            # --------------------------------------------
            # Full future H-step observation required
            #
            # This removes right censoring.
            # --------------------------------------------

            future_end = (
                i + horizon
            )

            if future_end >= n:

                excluded_censored += 1
                continue

            # Exactly H future observations:
            # i+1 through i+H inclusive.
            future_failure = (
                failures[
                    i + 1:
                    i + horizon + 1
                ]
            )

            if len(
                future_failure
            ) != horizon:

                raise RuntimeError(
                    "Future-window length error: "
                    f"run={run_id}, "
                    f"time={times[i]}, "
                    f"H={horizon}"
                )

            label = int(
                np.any(
                    future_failure == 1
                )
            )

            # Distance to first future failure
            # useful later for lead-time evaluation.
            future_failure_idx = (
                np.where(
                    future_failure == 1
                )[0]
            )

            if len(
                future_failure_idx
            ) > 0:

                steps_to_failure = int(
                    future_failure_idx[0] + 1
                )

            else:

                steps_to_failure = np.nan

            sample_id = (
                f"{run_id}__t{int(times[i])}"
                f"__H{horizon}"
            )

            records.append({
                "sample_id":
                    sample_id,

                "run_id":
                    run_id,

                "time":
                    int(times[i]),

                "attack_family":
                    family,

                "horizon":
                    int(horizon),

                "history_start_time":
                    int(
                        times[history_start]
                    ),

                "history_end_time":
                    int(times[i]),

                "future_start_time":
                    int(times[i] + 1),

                "future_end_time":
                    int(
                        times[i] + horizon
                    ),

                "failure":
                    int(failures[i]),

                "mission_effectiveness":
                    float(mission[i]),

                "label":
                    label,

                "steps_to_failure":
                    steps_to_failure,
            })

    samples = pd.DataFrame(
        records
    )

    diagnostics = {
        "excluded_no_history":
            excluded_no_history,

        "excluded_failed_history":
            excluded_failed_history,

        "excluded_right_censored":
            excluded_censored,

        "eligible_samples":
            len(samples),
    }

    return (
        samples,
        diagnostics,
    )


# ============================================================
# Assign frozen train/validation/test split
# ============================================================

def assign_split(
    samples,
    train_runs,
    val_runs,
    test_runs,
):

    def get_split(run_id):

        if run_id in train_runs:
            return "train"

        if run_id in val_runs:
            return "validation"

        if run_id in test_runs:
            return "test"

        raise RuntimeError(
            f"Run {run_id} not found "
            f"in frozen split."
        )

    samples = samples.copy()

    samples["split"] = (
        samples["run_id"]
        .map(get_split)
    )

    return samples


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 78)
    print(
        "MissionShield E1-FINAL | "
        "Censoring-Safe Protocol Freeze"
    )
    print("=" * 78)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = pd.read_csv(
        DATA
    )

    print(
        f"Node rows : {len(df):,}"
    )

    print(
        f"Runs      : "
        f"{df.run_id.nunique()}"
    )

    # --------------------------------------------------------
    # Frozen split
    # --------------------------------------------------------

    with open(
        SPLIT_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        split = json.load(f)

    train_runs = set(
        split["train"]
    )

    val_runs = set(
        split["validation"]
    )

    test_runs = set(
        split["test"]
    )

    assert train_runs.isdisjoint(
        val_runs
    )

    assert train_runs.isdisjoint(
        test_runs
    )

    assert val_runs.isdisjoint(
        test_runs
    )

    all_split_runs = (
        train_runs |
        val_runs |
        test_runs
    )

    dataset_runs = set(
        df["run_id"].unique()
    )

    if (
        all_split_runs !=
        dataset_runs
    ):

        missing_from_split = (
            dataset_runs -
            all_split_runs
        )

        extra_in_split = (
            all_split_runs -
            dataset_runs
        )

        raise RuntimeError(
            "Frozen split mismatch.\n"
            f"Missing from split: "
            f"{missing_from_split}\n"
            f"Extra in split: "
            f"{extra_in_split}"
        )

    print("\nFrozen run split")

    print(
        f"Train      : "
        f"{len(train_runs)}"
    )

    print(
        f"Validation : "
        f"{len(val_runs)}"
    )

    print(
        f"Test       : "
        f"{len(test_runs)}"
    )

    # --------------------------------------------------------
    # System timeline
    # --------------------------------------------------------

    timeline = build_timeline(
        df
    )

    print(
        f"\nSystem timestamps : "
        f"{len(timeline):,}"
    )

    validate_runs(
        timeline
    )

    all_horizon_samples = []

    protocol_summary = []

    # ========================================================
    # Horizon-specific immutable sample sets
    # ========================================================

    for H in HORIZONS:

        print(
            "\n" +
            "=" * 78
        )

        print(
            f"FREEZING H={H}"
        )

        print(
            "=" * 78
        )

        samples, diagnostics = (
            build_horizon_samples(
                timeline,
                H,
            )
        )

        samples = assign_split(
            samples,
            train_runs,
            val_runs,
            test_runs,
        )

        # --------------------------------------------
        # Safety checks
        # --------------------------------------------

        if not (
            samples["failure"] == 0
        ).all():

            raise RuntimeError(
                f"H={H}: failed current "
                "states entered final set."
            )

        if (
            samples["sample_id"]
            .duplicated()
            .any()
        ):

            raise RuntimeError(
                f"H={H}: duplicate "
                "sample IDs."
            )

        # --------------------------------------------
        # Summary by split
        # --------------------------------------------

        print(
            "Excluded - no full history : "
            f"{diagnostics['excluded_no_history']:,}"
        )

        print(
            "Excluded - failed history  : "
            f"{diagnostics['excluded_failed_history']:,}"
        )

        print(
            "Excluded - right censored  : "
            f"{diagnostics['excluded_right_censored']:,}"
        )

        print(
            f"Eligible total             : "
            f"{len(samples):,}"
        )

        for split_name in [
            "train",
            "validation",
            "test",
        ]:

            s = samples[
                samples["split"]
                == split_name
            ]

            positives = int(
                s["label"].sum()
            )

            prevalence = (
                float(
                    s["label"].mean()
                )
                if len(s) > 0
                else np.nan
            )

            print(
                f"{split_name:10s}: "
                f"{len(s):6,d} | "
                f"positive="
                f"{positives:5,d} | "
                f"rate="
                f"{prevalence:.4f}"
            )

            protocol_summary.append({
                "horizon":
                    H,

                "split":
                    split_name,

                "samples":
                    len(s),

                "positives":
                    positives,

                "positive_rate":
                    prevalence,

                "excluded_no_history":
                    diagnostics[
                        "excluded_no_history"
                    ],

                "excluded_failed_history":
                    diagnostics[
                        "excluded_failed_history"
                    ],

                "excluded_right_censored":
                    diagnostics[
                        "excluded_right_censored"
                    ],
            })

        # --------------------------------------------
        # Save horizon-specific immutable file
        # --------------------------------------------

        output = (
            RESULTS /
            f"e1_final_samples_H{H}.csv"
        )

        samples.to_csv(
            output,
            index=False,
        )

        print(
            f"\nSaved: {output}"
        )

        all_horizon_samples.append(
            samples
        )

    # ========================================================
    # Combined immutable manifest
    # ========================================================

    combined = pd.concat(
        all_horizon_samples,
        ignore_index=True,
    )

    combined_file = (
        RESULTS /
        "e1_final_sample_manifest.csv"
    )

    combined.to_csv(
        combined_file,
        index=False,
    )

    # ========================================================
    # Protocol summary
    # ========================================================

    summary_df = pd.DataFrame(
        protocol_summary
    )

    summary_file = (
        RESULTS /
        "e1_final_protocol_summary.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False,
    )

    # ========================================================
    # Machine-readable frozen protocol
    # ========================================================

    config = {
        "dataset":
            str(DATA),

        "split_file":
            str(SPLIT_FILE),

        "history":
            HISTORY,

        "horizons":
            HORIZONS,

        "protocol": {
            "current_state":
                "operational_only",

            "history":
                "all_history_steps_pre_failure",

            "forecast_window":
                "t+1_through_t+H",

            "right_censoring":
                "excluded",

            "label":
                "any_failure_within_complete_future_window",

            "split_unit":
                "run",

            "test_set":
                "held_out_frozen_runs",
        },
    }

    config_file = (
        RESULTS /
        "e1_final_protocol.json"
    )

    with open(
        config_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            config,
            f,
            indent=2,
        )

    # ========================================================
    # Final display
    # ========================================================

    print(
        "\n" +
        "=" * 78
    )

    print(
        "FINAL FROZEN PROTOCOL SUMMARY"
    )

    print(
        "=" * 78
    )

    print(
        summary_df[
            [
                "horizon",
                "split",
                "samples",
                "positives",
                "positive_rate",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nFiles created:"
    )

    print(
        RESULTS /
        "e1_final_samples_H10.csv"
    )

    print(
        RESULTS /
        "e1_final_samples_H30.csv"
    )

    print(
        RESULTS /
        "e1_final_samples_H60.csv"
    )

    print(
        combined_file
    )

    print(
        summary_file
    )

    print(
        config_file
    )

    print(
        "\nE1 FINAL PROTOCOL FREEZE COMPLETE"
    )


if __name__ == "__main__":
    main()