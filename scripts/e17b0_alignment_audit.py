from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("results/e12")
files = (
    list((ROOT / "predictions").glob("*_predictions.csv"))
    + list((ROOT / "e12e_predictions").glob("*_predictions.csv"))
)

assert len(files) == 360, f"Expected 360 files, found {len(files)}"

required = [
    "sample_id", "run_id", "time",
    "attack_family", "horizon",
    "steps_to_failure", "label", "probability"
]

records = []
references = {}
issues = []

for path in sorted(files):
    df = pd.read_csv(path)

    missing = set(required) - set(df.columns)
    if missing:
        issues.append(f"{path}: missing {sorted(missing)}")
        continue

    key = (
        str(df["attack_family"].iloc[0]),
        int(df["horizon"].iloc[0])
    )

    identity = df[
        ["sample_id", "run_id", "time",
         "steps_to_failure", "label"]
    ].sort_values("sample_id").reset_index(drop=True)

    if key not in references:
        references[key] = identity
    elif not identity.equals(references[key]):
        issues.append(f"Ground-truth mismatch: {path}")

    if df["sample_id"].duplicated().any():
        issues.append(f"Duplicate sample IDs: {path}")

    if not np.isfinite(df["probability"]).all():
        issues.append(f"Nonfinite probabilities: {path}")

    if not df["probability"].between(0, 1).all():
        issues.append(f"Probabilities outside [0,1]: {path}")

    records.append({
        "file": str(path),
        "samples": len(df),
        "runs": df["run_id"].nunique(),
        "positives": int(df["label"].sum()),
        "missing_failure_steps": int(
            df["steps_to_failure"].isna().sum()
        )
    })

audit = pd.DataFrame(records)

print("\n=== E17-B0 ALIGNMENT AUDIT ===")
print("Files checked:", len(audit))
print("Conditions:", len(references))
print("Total issues:", len(issues))

print("\n=== SAMPLE COVERAGE ===")
print(
    audit[["samples", "runs", "positives"]]
    .describe()
    .round(2)
    .to_string()
)

print("\n=== FIRST 20 ISSUES ===")
for issue in issues[:20]:
    print(issue)

print("\n=== FAILURE-TIME FIELD ===")
print(
    "Missing steps_to_failure values:",
    int(audit["missing_failure_steps"].sum())
)

print("\nNo files modified.")

