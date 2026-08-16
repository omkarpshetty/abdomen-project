"""
batch_extract_dicom_params.py

Same folder-parsing pattern as batch_extract_features.py, but pulls
acquisition parameters (kVp, tube current, CTDIvol, estimated DLP) instead
of body-size metrics. Run this across all patient/phase folders.

Usage:
    python batch_extract_dicom_params.py --root "C:\\Users\\omkar\\OneDrive\\Desktop\\major project\\1-144" --out dicom_params.csv
"""
import os
import re
import argparse
import time
import csv

from dose.extract_dicom_params import extract_acquisition_params

FOLDER_PATTERN = re.compile(r"^\s*(\d+)\s*[_\-\s]*\s*(plain|venous|p|v)", re.IGNORECASE)


def parse_folder_name(name: str):
    m = FOLDER_PATTERN.match(name)
    if not m:
        return None, None
    uid = m.group(1)
    phase = "plain" if m.group(2).lower().startswith("p") else "venous"
    return uid, phase


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", default="dicom_params.csv")
    args = parser.parse_args()

    all_dirs = sorted(d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d)))
    print(f"Found {len(all_dirs)} folders under --root.")

    rows = []
    skipped = []
    for i, folder in enumerate(all_dirs, 1):
        uid, phase = parse_folder_name(folder)
        if uid is None:
            skipped.append(folder)
            print(f"[{i}/{len(all_dirs)}] {folder} -- SKIPPED")
            continue

        dicom_dir = os.path.join(args.root, folder)
        t0 = time.time()
        try:
            params = extract_acquisition_params(dicom_dir)
            params["UID"] = uid
            params["PHASE"] = phase
            rows.append(params)
            ctdi = params["mean_ctdivol_mGy"]
            ctdi_str = f"{ctdi:.1f}" if ctdi is not None else "N/A"
            print(f"[{i}/{len(all_dirs)}] {folder} -> UID={uid} PHASE={phase} "
                  f"done in {time.time()-t0:.1f}s (CTDIvol={ctdi_str} mGy)")
        except Exception as e:
            print(f"[{i}/{len(all_dirs)}] {folder} FAILED: {e}")

    if not rows:
        print("No folders processed successfully.")
        return

    fieldnames = ["UID", "PHASE", "mean_kvp", "mean_tube_current_mA", "mean_ctdivol_mGy",
                  "mean_exposure_time_ms", "mean_exposure_mAs", "pitch_factor",
                  "mean_slice_thickness_mm", "scan_length_cm", "estimated_dlp_mgycm"]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} row(s) to {args.out}")
    if skipped:
        print(f"{len(skipped)} folder(s) skipped.")


if __name__ == "__main__":
    main()