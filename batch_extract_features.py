"""
batch_extract_features.py

Runs compute_size_features() over EVERY patient/phase folder under --root,
and writes one row per folder to a CSV. Folders are named things like
"1 Plain", "3 plain_", "61 p", "62 v" etc (patient number + phase, with
wildly inconsistent capitalization/spacing/abbreviation) -- this script
parses the UID and phase (plain/venous) out of each folder name with a
regex instead of assuming an exact match, so it's robust to that mess.

This is the fast, lightweight step you run across all 144 patients --
unlike batch_main.py (full organ annotation), which you should only run
on a handful of patients.

Usage:
    python batch_extract_features.py --root "C:\\Users\\omkar\\OneDrive\\Desktop\\major project\\1-144" --out size_features.csv
"""
import os
import re
import argparse
import time
import csv

from dose.extract_size_features import compute_size_features

# Handles every naming style seen in the actual data: "1 Plain", "3 plain_",
# "61 p", "62 v" -- leading digits (UID), separator, then either the full
# word or just the first letter (p/v), any case, optional trailing junk.
FOLDER_PATTERN = re.compile(r"^\s*(\d+)\s*[_\-\s]*\s*(plain|venous|p|v)", re.IGNORECASE)


def parse_folder_name(name: str):
    """Returns (uid, phase) or (None, None) if the folder name doesn't match."""
    m = FOLDER_PATTERN.match(name)
    if not m:
        return None, None
    uid = m.group(1)
    raw_phase = m.group(2).lower()
    phase = "plain" if raw_phase.startswith("p") else "venous"
    return uid, phase


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Folder containing one subfolder per patient PER PHASE (e.g. '1 Plain', '1 Venous', '61 p', '61 v', ...)")
    parser.add_argument("--out", default="size_features.csv", help="Output CSV path")
    args = parser.parse_args()

    all_dirs = sorted(
        d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d))
    )
    print(f"Found {len(all_dirs)} folders under --root.")

    rows = []
    skipped = []
    for i, folder in enumerate(all_dirs, 1):
        uid, phase = parse_folder_name(folder)
        if uid is None:
            skipped.append(folder)
            print(f"[{i}/{len(all_dirs)}] {folder} -- SKIPPED (doesn't match 'NNN plain/venous/p/v' pattern)")
            continue

        dicom_dir = os.path.join(args.root, folder)
        t0 = time.time()
        try:
            feats = compute_size_features(dicom_dir)
            feats["UID"] = uid
            feats["PHASE"] = phase
            rows.append(feats)
            print(f"[{i}/{len(all_dirs)}] {folder} -> UID={uid} PHASE={phase} "
                  f"done in {time.time()-t0:.1f}s (WED={feats['mean_water_equiv_diameter_cm']:.1f} cm)")
        except Exception as e:
            print(f"[{i}/{len(all_dirs)}] {folder} FAILED: {e}")

    if not rows:
        print("No folders processed successfully -- nothing written.")
        return

    fieldnames = ["UID", "PHASE", "n_slices", "mean_effective_diameter_cm", "mean_water_equiv_diameter_cm"]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} row(s) to {args.out}")
    if skipped:
        print(f"\n{len(skipped)} folder(s) skipped (didn't match naming pattern):")
        for s in skipped[:20]:
            print(f"  - {s}")
        if len(skipped) > 20:
            print(f"  ... and {len(skipped)-20} more")


if __name__ == "__main__":
    main()