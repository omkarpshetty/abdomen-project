"""
batch_diagnose_ctdivol.py

Scans EVERY patient/phase folder (not just one) to give a definitive
answer on whether a real, independently-measured CTDIvol exists anywhere
in your dataset -- as a direct image-header tag, or hiding inside a
separate Dose Report SR object your other scripts never look at.

This only reads headers (stop_before_pixels=True) so it's much faster
than a full DICOM load -- safe to run across all 144 patients.

Usage:
    python batch_diagnose_ctdivol.py --root "C:\\...\\1-144"
"""
import argparse
import glob
import os
import pydicom

DOSE_SR_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.88.67"  # X-Ray Radiation Dose SR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()

    patient_folders = sorted(
        d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d))
    )
    print(f"Scanning {len(patient_folders)} folder(s) under {args.root} ...\n")

    modality_counts = {}
    sop_class_counts = {}
    dose_related_keywords = set()
    direct_ctdivol_files = []
    dose_sr_folders = []
    unreadable = 0
    total_files = 0

    for fi, folder in enumerate(patient_folders, 1):
        folder_path = os.path.join(args.root, folder)
        files = [f for f in glob.glob(os.path.join(folder_path, "*")) if os.path.isfile(f)]

        for f in files:
            total_files += 1
            try:
                ds = pydicom.dcmread(f, force=True, stop_before_pixels=True)
            except Exception:
                unreadable += 1
                continue

            modality = getattr(ds, "Modality", "UNKNOWN")
            modality_counts[modality] = modality_counts.get(modality, 0) + 1

            sop_class = getattr(ds, "SOPClassUID", "UNKNOWN")
            sop_class_counts[str(sop_class)] = sop_class_counts.get(str(sop_class), 0) + 1

            if hasattr(ds, "CTDIvol"):
                direct_ctdivol_files.append((folder, os.path.basename(f)))

            if str(sop_class) == DOSE_SR_SOP_CLASS or modality == "SR":
                dose_sr_folders.append((folder, os.path.basename(f)))

            try:
                found = ds.dir("dose")
            except Exception:
                found = []
            for kw in found:
                dose_related_keywords.add(kw)

        if fi % 20 == 0 or fi == len(patient_folders):
            print(f"  ...scanned {fi}/{len(patient_folders)} folders "
                  f"({total_files} files so far)")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nTotal files scanned: {total_files}  (unreadable: {unreadable})")

    print("\nModality breakdown:")
    for m, c in sorted(modality_counts.items(), key=lambda x: -x[1]):
        print(f"  {m:<10} {c}")

    print("\nSOP Class breakdown:")
    for s, c in sorted(sop_class_counts.items(), key=lambda x: -x[1]):
        print(f"  {s:<45} {c}")

    print(f"\nFiles with a direct CTDIvol tag (0018,9345): {len(direct_ctdivol_files)}")
    if direct_ctdivol_files:
        for folder, fname in direct_ctdivol_files[:10]:
            print(f"    {folder}/{fname}")
        if len(direct_ctdivol_files) > 10:
            print(f"    ... and {len(direct_ctdivol_files) - 10} more")

    print(f"\nDose Report SR objects found (SOPClass or Modality=SR): {len(dose_sr_folders)}")
    if dose_sr_folders:
        for folder, fname in dose_sr_folders[:10]:
            print(f"    {folder}/{fname}")

    print(f"\nAny tag pydicom recognizes with 'dose' in its name, seen anywhere: {sorted(dose_related_keywords)}")

    print("\n" + "-" * 60)
    if direct_ctdivol_files:
        print("CONCLUSION: some files DO carry a direct CTDIvol tag -- "
              "worth checking why extract_dicom_params.py isn't picking it up "
              "for those specific patients (folder naming, phase mismatch, etc).")
    elif dose_sr_folders:
        print("CONCLUSION: no direct CTDIvol tag anywhere, but Dose Report SR "
              "object(s) exist. CTDIvol is very likely recoverable from inside "
              "those -- worth writing a parser for their ContentSequence.")
    else:
        print("CONCLUSION: no direct CTDIvol tag and no Dose Report SR object "
              "anywhere in the dataset. CTDIvol genuinely was not exported/recorded "
              "for this study. The derived-from-real-DLP workaround already in "
              "place is the correct call, and the conventional-formula comparison "
              "should be documented as not possible with this data, not left as "
              "an open question.")


if __name__ == "__main__":
    main()