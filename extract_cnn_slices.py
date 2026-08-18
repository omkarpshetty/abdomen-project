"""
extract_cnn_slices.py

Extracts ONE representative whole-abdomen axial slice per (patient, phase)
folder, windowed and saved as a PNG the CNN can read. This does NOT need
TotalSegmentator or organ masks -- it's the raw whole-scan image, per your
choice to have the CNN look at the whole abdomen rather than per-organ crops.

Slice choice: the middle slice by z-position (index len(volume)//2). For a
correctly-exported full abdominal series this lands roughly mid-abdomen.
For the ~17% of folders you already found have implausibly short
scan_length_cm (partial/incomplete exports), the "middle slice" is still
whatever's there -- those patients will just give the CNN a less
representative image, same limitation as everywhere else in this project.

Usage:
    python extract_cnn_slices.py --root "C:\\...\\1-144" --out-dir cnn_slices --manifest cnn_slice_manifest.csv
"""
import argparse
import csv
import os
import re

import numpy as np
from PIL import Image

from src.dicom_io import load_dicom_series

FOLDER_PATTERN = re.compile(r"^\s*(\d+)\s*[_\-\s]*\s*(plain|venous|p|v)", re.IGNORECASE)

# Standard abdomen soft-tissue window, matching the rest of this project
# (see README's --level/--width defaults).
WINDOW_LEVEL = 40
WINDOW_WIDTH = 400
IMG_SIZE = 224  # standard ImageNet input size, so we can use a pretrained backbone


def parse_folder_name(name: str):
    m = FOLDER_PATTERN.match(name)
    if not m:
        return None, None
    uid = m.group(1)
    phase = "plain" if m.group(2).lower().startswith("p") else "venous"
    return uid, phase


def window_and_resize(slice_hu: np.ndarray) -> Image.Image:
    lo = WINDOW_LEVEL - WINDOW_WIDTH / 2
    hi = WINDOW_LEVEL + WINDOW_WIDTH / 2
    clipped = np.clip(slice_hu, lo, hi)
    normalized = ((clipped - lo) / (hi - lo) * 255).astype(np.uint8)
    img = Image.fromarray(normalized, mode="L")
    return img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out-dir", default="cnn_slices")
    parser.add_argument("--manifest", default="cnn_slice_manifest.csv")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    all_dirs = sorted(d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d)))
    print(f"Found {len(all_dirs)} folders under --root.")

    rows = []
    for i, folder in enumerate(all_dirs, 1):
        uid, phase = parse_folder_name(folder)
        if uid is None:
            print(f"[{i}/{len(all_dirs)}] {folder} -- SKIPPED (name doesn't match pattern)")
            continue

        dicom_dir = os.path.join(args.root, folder)
        try:
            volume, _ = load_dicom_series(dicom_dir)
            mid = volume.shape[0] // 2
            img = window_and_resize(volume[mid])
            fname = f"{uid}_{phase}.png"
            img.save(os.path.join(args.out_dir, fname))
            rows.append({"UID": uid, "PHASE": phase, "slice_path": fname})
            print(f"[{i}/{len(all_dirs)}] {folder} -> saved {fname} (slice {mid}/{volume.shape[0]})")
        except Exception as e:
            print(f"[{i}/{len(all_dirs)}] {folder} FAILED: {e}")

    if not rows:
        print("No folders processed successfully.")
        return

    with open(args.manifest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["UID", "PHASE", "slice_path"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} slice(s) to {args.out_dir}/, manifest at {args.manifest}")


if __name__ == "__main__":
    main()