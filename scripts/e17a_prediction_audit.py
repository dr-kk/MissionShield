from pathlib import Path
import pandas as pd

ROOT = Path("results")
OUT = Path("results/e17")
OUT.mkdir(parents=True, exist_ok=True)

records = []

keywords = [
    "prediction", "predictions", "forecast",
    "ground_truth", "manifest", "event"
]

for path in ROOT.rglob("*.csv"):
    if "e17" in path.parts:
        continue

    if not any(k in path.name.lower() for k in keywords):
        continue

    try:
        df = pd.read_csv(path, nrows=5)

        records.append({
            "file": str(path),
            "columns": " | ".join(map(str, df.columns)),
            "sample_rows": len(df),
            "has_run_id": any(
                "run" in str(c).lower()
                for c in df.columns
            ),
            "has_timestamp": any(
                x in str(c).lower()
                for c in df.columns
                for x in ["timestamp", "time", "step"]
            ),
            "has_label": any(
                x in str(c).lower()
                for c in df.columns
                for x in ["label", "target", "failure"]
            ),
            "has_score": any(
                x in str(c).lower()
                for c in df.columns
                for x in ["prob", "score", "prediction"]
            )
        })

    except Exception as exc:
        records.append({
            "file": str(path),
            "error": str(exc)
        })

audit = pd.DataFrame(records)

print("\n=== E17-A PREDICTION DATA AUDIT ===")
print("Candidate files:", len(audit))

if len(audit):
    print(
        audit.to_string(
            index=False,
            max_colwidth=100
        )
    )

    print("\n=== POTENTIAL OPERATIONAL EVALUATION FILES ===")

    required = [
        "has_run_id",
        "has_timestamp",
        "has_label",
        "has_score"
    ]

    for col in required:
        if col not in audit:
            audit[col] = False

    eligible = audit[
        audit[required].fillna(False).all(axis=1)
    ]

    print(
        eligible[
            ["file", "columns"]
        ].to_string(index=False)
        if len(eligible)
        else "No fully matched prediction files identified."
    )

    output = OUT / "e17a_prediction_file_audit.csv"

    if output.exists():
        raise RuntimeError(
            f"Refusing to overwrite: {output}"
        )

    audit.to_csv(output, index=False)
    print("\nSaved:", output)

else:
    print("No candidate CSV files found.")
