from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(r"D:\MissionShield")
RESULTS = ROOT / "results"

HORIZONS = [10, 30, 60]
HELD_OUT_FAMILIES = ["ddos", "multi_stage", "service_degradation"]


def stats(df):
    n = len(df)
    pos = int(df["label"].sum()) if n else 0
    neg = n - pos
    rate = pos / n if n else np.nan
    runs = int(df["run_id"].nunique()) if n else 0
    return n, pos, neg, rate, runs


def main():
    print("=" * 100)
    print("MissionShield E4-B0 | LOAO PROTOCOL AUDIT")
    print("NO TRAINING | NO MODEL CHANGES | FROZEN MANIFESTS ONLY")
    print("=" * 100)

    rows = []

    for H in HORIZONS:
        path = RESULTS / f"e1_final_samples_H{H}.csv"
        if not path.exists():
            raise FileNotFoundError(path)

        m = pd.read_csv(path)
        m["attack_family"] = m["attack_family"].astype(str)
        m["split"] = m["split"].astype(str)
        m["label"] = m["label"].astype(int)

        print("\n" + "=" * 100)
        print(f"HORIZON H={H} | frozen samples={len(m):,}")
        print("=" * 100)

        for held in HELD_OUT_FAMILIES:
            train = m[
                (m["split"] == "train") &
                (m["attack_family"] != held)
            ].copy()

            val = m[
                (m["split"] == "validation") &
                (m["attack_family"] != held)
            ].copy()

            test = m[
                (m["split"] == "test") &
                (m["attack_family"] == held)
            ].copy()

            tn, tp, tneg, trate, truns = stats(train)
            vn, vp, vneg, vrate, vruns = stats(val)
            en, ep, eneg, erate, eruns = stats(test)

            train_ok = tp > 0 and tneg > 0
            val_ok = vp > 0 and vneg > 0
            test_two_class = ep > 0 and eneg > 0
            event_runs = int(
                test.loc[
                    (test["label"] == 1) &
                    test["steps_to_failure"].notna(),
                    "run_id"
                ].nunique()
            )

            feasible = train_ok and val_ok and ep > 0

            print(f"\nHeld out: {held}")
            print(
                f"  TRAIN      n={tn:6d} pos={tp:5d} neg={tneg:6d} "
                f"rate={trate:.4f} runs={truns:3d}"
            )
            print(
                f"  VALIDATION n={vn:6d} pos={vp:5d} neg={vneg:6d} "
                f"rate={vrate:.4f} runs={vruns:3d}"
            )
            print(
                f"  TEST-LOAO  n={en:6d} pos={ep:5d} neg={eneg:6d} "
                f"rate={erate:.4f} runs={eruns:3d} failure_events={event_runs}"
            )
            print(
                f"  Feasible for training/checkpointing: {'YES' if feasible else 'NO'}"
            )
            if not test_two_class:
                print(
                    "  Note: held-out test is single-class; AUROC is undefined "
                    "although event sensitivity may still be measurable."
                )

            rows.append({
                "horizon": H,
                "held_out_family": held,
                "train_samples": tn,
                "train_positive": tp,
                "train_negative": tneg,
                "train_positive_rate": trate,
                "train_runs": truns,
                "validation_samples": vn,
                "validation_positive": vp,
                "validation_negative": vneg,
                "validation_positive_rate": vrate,
                "validation_runs": vruns,
                "test_samples": en,
                "test_positive": ep,
                "test_negative": eneg,
                "test_positive_rate": erate,
                "test_runs": eruns,
                "test_failure_events": event_runs,
                "train_two_class": train_ok,
                "validation_two_class": val_ok,
                "test_two_class": test_two_class,
                "loao_feasible": feasible,
            })

    result = pd.DataFrame(rows)
    outfile = RESULTS / "e4b0_loao_protocol_audit.csv"
    result.to_csv(outfile, index=False)

    print("\n" + "=" * 100)
    print("LOAO AUDIT SUMMARY")
    print("=" * 100)
    print(
        result[
            [
                "horizon", "held_out_family",
                "train_positive", "validation_positive",
                "test_positive", "test_failure_events",
                "test_two_class", "loao_feasible"
            ]
        ].to_string(index=False)
    )

    print(f"\nSaved: {outfile}")
    print("\nE4-B0 COMPLETE")


if __name__ == "__main__":
    main()
