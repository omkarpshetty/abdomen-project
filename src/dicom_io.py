"""
dicom_io.py

Loads a DICOM series (a folder of .dcm files, however exported from OsiriX)
into a sorted 3D numpy volume in Hounsfield Units, and converts it to a
NIfTI (.nii.gz) file, which is the format TotalSegmentator requires.
"""
import os
import glob
import numpy as np
import pydicom
import SimpleITK as sitk


def load_dicom_series(folder: str):
    """
    Reads every DICOM file in `folder` (non-recursive) and returns:
        volume: numpy array [z, y, x] in Hounsfield Units
        slices: the sorted list of pydicom Datasets (for spacing / metadata)
    """
    files = [f for f in glob.glob(os.path.join(folder, "*"))
             if os.path.isfile(f)]
    datasets = []
    for f in files:
        try:
            ds = pydicom.dcmread(f, force=True)
            if hasattr(ds, "pixel_array"):
                datasets.append(ds)
        except Exception:
            continue

    if not datasets:
        raise FileNotFoundError(
            f"No readable DICOM files found in {folder}. "
            "Export the series from OsiriX first (see README)."
        )

    # Sort slices into correct anatomical order
    datasets.sort(key=lambda d: float(getattr(d, "ImagePositionPatient", [0, 0, 0])[2]))

    # RescaleSlope/RescaleIntercept can differ per-slice (mixed reconstructions,
    # or an accidental second series in the same folder). Using only
    # datasets[0]'s values for the whole stack silently corrupts HU for any
    # slice where they differ -- which then breaks every HU-based threshold
    # downstream (bone, air, fat) with no error thrown. Apply per-slice.
    volume = np.stack([
        d.pixel_array.astype(np.float32) * float(getattr(d, "RescaleSlope", 1))
        + float(getattr(d, "RescaleIntercept", 0))
        for d in datasets
    ], axis=0)

    return volume, datasets


def dicom_folder_to_nifti(dicom_folder: str, out_nifti_path: str):
    """
    Uses SimpleITK's DICOM series reader (handles spacing/orientation more
    robustly than a manual pydicom stack) to write a .nii.gz TotalSegmentator
    can consume directly.
    """
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_folder)
    if not series_ids:
        raise FileNotFoundError(f"No DICOM series found in {dicom_folder}")

    # If multiple series live in the same folder, use the one with the most slices
    best_files, best_len = None, 0
    for sid in series_ids:
        files = reader.GetGDCMSeriesFileNames(dicom_folder, sid)
        if len(files) > best_len:
            best_files, best_len = files, len(files)

    reader.SetFileNames(best_files)
    image = reader.Execute()
    os.makedirs(os.path.dirname(out_nifti_path), exist_ok=True)
    sitk.WriteImage(image, out_nifti_path)
    return out_nifti_path