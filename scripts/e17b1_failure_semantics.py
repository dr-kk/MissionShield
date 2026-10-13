from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("results/e12")
files = (
    list((ROOT / "predictions").glob("*_predictions.csv"))
    + list((ROOT / "e12e_predictions").glob("*_predictions.csv"))
)

# One frozen prediction file per attack-family/horizon condition.
selected = {}

for path in sorted(files):
    df = pd.read_csv(path)
    family = str(df["attack_family"].iloc[0])
    horizon = int(df["horizon"].iloc[0])
    selected.setdefault((family, horizon), path)

assert len(selected) == 9

print("\n=== E17-B1 FAILURE-TIME SEMANTICS ===")

for (family, horizon), path in sorted(selected.items()):
    df = pd.read_csv(path)

    labels = pd.to_numeric(df["label"], errors="coerce")
    steps = pd.to_numeric(
        df["steps_to_failure"], errors="coerce"
    )

    positives = labels.eq(1)
    negatives = labels.eq(0)
    finite = np.isfinite(steps)

    positive_missing = int((positives & ~finite).sum())
    negative_with_steps = int((negatives & finite).sum())

    positive_outside = int(
        (positives & finite &
         ((steps <= 0) | (steps > horizon))).sum()
    )

    negative_inside = int(
        (negatives & finite &
         (steps > 0) & (steps <= horizon)).sum()
    )

    print(f"\n{family} H{horizon}")
    print("  Samples:", len(df))
    print("  Positives:", int(positives.sum()))
    print("  Negatives:", int(negatives.sum()))
    print("  Positive labels missing failure steps:",
          positive_missing)
    print("  Negative labels with failure steps:",
          negative_with_steps)
    print("  Positive steps outside (0,H]:",
          positive_outside)
    print("  Negative steps inside (0,H]:",
          negative_inside)

    print("  Positive failure-step distribution:")
    print(
        steps[positives].describe().to_string()
    )

    print("  Example rows:")
    print(
        df[[
            "run_id", "time", "label",
            "steps_to_failure", "probability"
        ]].head(3).to_string(index=False)
    )

print("\nAudit only. No files modified.")
