"""
check_label_outliers.py

Flags patients with extreme AGE/HEIGHT/WEIGHT/DLP values relative to the
rest of the dataset. With only ~27 patients per CV fold, a single extreme
or mis-entered value can single-handedly drag a fold's R^2 deeply negative
-- this tells you whether that's what happened, and gives you specific
UIDs to go check against the original records.

Usage:
    python check_label_outliers.py --labels "C:\\...\\1 - 144  excel (2).xlsx"
"""
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--z-threshold", type=float, default=2.5)
    args = parser.parse_args()

    df = pd.read_excel(args.labels)
    df.columns = [c.strip() for c in df.columns]
    if df["AGE"].dtype == object:
        df = df.rename(columns={"GENDER": "AGE_tmp", "AGE": "GENDER_tmp"})
        df = df.rename(columns={"AGE_tmp": "AGE", "GENDER_tmp": "GENDER"})
        print("[note] detected swapped AGE/GENDER columns -- corrected automatically.\n")

    cols = ["AGE", "HEIGHT", "WEIGHT", "DLP -PLAIN", "DLP-VENOUS"]
    cols = [c for c in cols if c in df.columns]

    print(f"{len(df)} patients. Column ranges:")
    for c in cols:
        print(f"  {c:<12} min={df[c].min():.1f}  max={df[c].max():.1f}  "
              f"mean={df[c].mean():.1f}  std={df[c].std():.1f}")

    print(f"\nFlagging any patient with |z-score| > {args.z_threshold} on any column:\n")
    flagged_any = False
    for c in cols:
        z = (df[c] - df[c].mean()) / df[c].std()
        flagged = df[z.abs() > args.z_threshold]
        for _, row in flagged.iterrows():
            flagged_any = True
            print(f"  UID {row['UID']}: {c} = {row[c]:.1f}  (z={z.loc[row.name]:.2f})   "
                  f"[AGE={row.get('AGE')}, GENDER={row.get('GENDER')}, "
                  f"HEIGHT={row.get('HEIGHT')}, WEIGHT={row.get('WEIGHT')}, "
                  f"DLP-PLAIN={row.get('DLP -PLAIN')}, DLP-VENOUS={row.get('DLP-VENOUS')}]")

    if not flagged_any:
        print("  (none found at this threshold)")


if __name__ == "__main__":
    main()