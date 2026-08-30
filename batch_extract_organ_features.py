"""
batch_extract_organ_features.py

Runs TotalSegmentator (organ segmentation only, NO colored PNG rendering --
that's the expensive part we don't need for training) across patient/phase
folders, and writes one row per (patient, phase, organ) with volume and
mean HU -- the real training features for a learned organ-dose model.

This is heavier than batch_extract_features.py (that one skips
TotalSegmentator entirely) -- budget real time for this to run.

RESUMABLE: writes each patient's result to --out immediately after it
finishes (not saved up until the end), and on a fresh run automatically
skips any (UID, PHASE) already present in an existing --out file. This
means if Colab disconnects or the runtime resets partway through, you can
just rerun the exact same command and it picks up where it left off
instead of losing all prior progress.

Usage:
    python batch_extract_organ_features.py --root "C:\\Users\\omkar\\OneDrive\\Desktop\\major project\\1-144" --out organ_features.csv --fast --device cpu --limit 40
"""
import os
import re
import argparse
import time
import csv
import shutil
import numpy as np
import nibabel as nib
import pandas as pd

from src.dicom_io import load_dicom_series, dicom_folder_to_nifti
from src.segment import (
    run_totalsegmentator, load_masks, merge_masks,
    get_bone_classes, get_muscle_classes,
)
from src.label_mapping import AUTO_LABEL_MAP

FOLDER_PATTERN = re.compile(r"^\s*(\d+)\s*[_\-\s]*\s*(plain|venous|p|v)", re.IGNORECASE)


def parse_folder_name(name: str):
    m = FOLDER_PATTERN.match(name)
    if not m:
        return None, None
    uid = m.group(1)
    phase = "plain" if m.group(2).lower().startswith("p") else "venous"
    return uid, phase


def process_one(dicom_dir: str, tmp_dir: str, fast: bool, device: str):
    """Runs segmentation for one patient and returns per-organ volume/HU rows."""
    nifti_path = os.path.join(tmp_dir, "volume.nii.gz")
    seg_dir = os.path.join(tmp_dir, "segmentation_masks")

    volume_hu, _ = load_dicom_series(dicom_dir)
    dicom_folder_to_nifti(dicom_dir, nifti_path)

    AUTO_LABEL_MAP["BONES"] = get_bone_classes(task="total")
    AUTO_LABEL_MAP["MUSCLE"] = get_muscle_classes(task="total")
    all_ts_classes = sorted({c for classes in AUTO_LABEL_MAP.values() if classes for c in classes})

    run_totalsegmentator(nifti_path, seg_dir, fast=fast, device=device,
                          body_seg=True, roi_subset=all_ts_classes, task="total")

    raw_masks = load_masks(seg_dir, all_ts_classes)
    img = nib.load(nifti_path)
    voxel_vol_cm3 = float(np.prod(img.header.get_zooms())) / 1000.0

    results = []
    for organ, ts_classes in AUTO_LABEL_MAP.items():
        if ts_classes is None:
            continue
        merged = merge_masks(raw_masks, ts_classes)
        if merged is None:
            continue
        volume_cm3 = float(merged.sum()) * voxel_vol_cm3
        mean_hu = float(volume_hu[merged].mean()) if merged.any() else None
        results.append({"organ": organ, "volume_cm3": volume_cm3, "mean_hu": mean_hu})

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", default="organ_features.csv")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--device", default="cpu", choices=["gpu", "cpu", "mps"])
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N matched folders (for a quick subset run)")
    parser.add_argument("--tmp-dir", default="./_organ_features_tmp",
                         help="Scratch folder for intermediate nifti/masks -- deleted after each patient")
    args = parser.parse_args()

    all_dirs = sorted(d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d)))
    matched = [(d, *parse_folder_name(d)) for d in all_dirs]
    matched = [(d, uid, phase) for d, uid, phase in matched if uid is not None]
    if args.limit:
        matched = matched[:args.limit]
    print(f"Processing {len(matched)} folder(s).")

    fieldnames = ["UID", "PHASE", "organ", "volume_cm3", "mean_hu"]
    file_exists = os.path.exists(args.out)
    already_done = set()
    if file_exists:
        existing = pd.read_csv(args.out, dtype={"UID": str})
        already_done = set(zip(existing["UID"], existing["PHASE"]))
        print(f"Resuming: {len(already_done)} (UID, PHASE) pairs already in {args.out}, will skip those.")

    csv_file = open(args.out, "a" if file_exists else "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()

    total_rows = 0
    for i, (folder, uid, phase) in enumerate(matched, 1):
        if (uid, phase) in already_done:
            print(f"[{i}/{len(matched)}] {folder} -- already done, skipping")
            continue
        dicom_dir = os.path.join(args.root, folder)
        t0 = time.time()
        try:
            organ_results = process_one(dicom_dir, args.tmp_dir, args.fast, args.device)
            for r in organ_results:
                r["UID"] = uid
                r["PHASE"] = phase
                writer.writerow(r)
                total_rows += 1
            csv_file.flush()  # write to disk immediately, don't wait for buffer
            print(f"[{i}/{len(matched)}] {folder} done in {time.time()-t0:.1f}s "
                  f"({len(organ_results)} organs) -- saved to disk")
        except Exception as e:
            print(f"[{i}/{len(matched)}] {folder} FAILED: {e}")
        finally:
            shutil.rmtree(args.tmp_dir, ignore_errors=True)  # free disk space between patients

    csv_file.close()
    print(f"\nDone. Wrote {total_rows} new organ-row(s) to {args.out} this run.")


if __name__ == "__main__":
    main()