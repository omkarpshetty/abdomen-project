"""
osirix_export.py

OsiriX (free edition) stores every imported study as raw DICOM files on
disk under:

    ~/Documents/OsiriX Data/Database.noindex/

It also keeps a proprietary SQLite index of that folder, but that schema
is undocumented and changes between versions, so instead of parsing it
directly, this script does the reliable thing: it walks the actual DICOM
files on disk and groups them by SeriesInstanceUID (a standard DICOM tag),
which works regardless of OsiriX version.

This must be run ON the Mac where OsiriX is installed (it reads the local
filesystem). If your DICOM data lives elsewhere, just point --source at
that folder instead -- everything downstream works the same either way.
"""
import os
import argparse
import shutil
from collections import defaultdict
import pydicom

DEFAULT_OSIRIX_PATH = os.path.expanduser(
    "~/Documents/OsiriX Data/Database.noindex"
)


def find_series(source_dir: str):
    """
    Walk `source_dir` recursively, read DICOM headers only (fast), and
    group file paths by (PatientName, StudyDescription, SeriesDescription,
    SeriesInstanceUID).
    Returns: dict[series_key] -> list of file paths
    """
    series = defaultdict(list)
    for root, _dirs, files in os.walk(source_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            try:
                ds = pydicom.dcmread(fpath, stop_before_pixels=True, force=True)
            except Exception:
                continue
            if not hasattr(ds, "SeriesInstanceUID"):
                continue
            key = (
                str(getattr(ds, "PatientName", "Unknown")),
                str(getattr(ds, "StudyDescription", "Unknown")),
                str(getattr(ds, "SeriesDescription", "Unknown")),
                str(ds.SeriesInstanceUID),
            )
            series[key].append(fpath)
    return series


def export_series(files, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for i, f in enumerate(files):
        shutil.copy2(f, os.path.join(out_dir, f"slice_{i:04d}.dcm"))
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Find & export a DICOM series from OsiriX's local storage")
    parser.add_argument("--source", default=DEFAULT_OSIRIX_PATH,
                         help="OsiriX database folder (default: standard macOS location)")
    parser.add_argument("--out", default="./data/exported_series",
                         help="Where to copy the chosen series")
    args = parser.parse_args()

    if not os.path.isdir(args.source):
        raise SystemExit(
            f"Could not find OsiriX database at:\n  {args.source}\n"
            "Open OsiriX, right-click the study in the database window and choose "
            "'Export -> DICOM Files', then pass that export folder with --source instead."
        )

    print(f"Scanning {args.source} for DICOM series (this can take a minute)...")
    series = find_series(args.source)
    if not series:
        raise SystemExit("No DICOM series found. Double check the --source path.")

    keys = list(series.keys())
    print(f"\nFound {len(keys)} series:\n")
    for i, k in enumerate(keys):
        patient, study, series_desc, uid = k
        print(f"  [{i}] Patient={patient} | Study={study} | Series={series_desc} | "
              f"{len(series[k])} slices")

    choice = int(input("\nEnter the number of the series to export: "))
    chosen_files = series[keys[choice]]
    out_path = export_series(chosen_files, args.out)
    print(f"\nExported {len(chosen_files)} DICOM files to: {out_path}")
    print("Next: python main.py --dicom-dir " + out_path)


if __name__ == "__main__":
    main()
