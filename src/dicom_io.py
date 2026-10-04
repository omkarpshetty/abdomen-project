"""Compatibility adapters around the canonical physical-space image loader."""
from pathlib import Path
import SimpleITK as sitk
from ct_dose.imaging import load_scan


def load_dicom_series(folder, series_uid=None):
    scan = load_scan(folder, series_uid)
    return scan.array, scan.datasets


def dicom_folder_to_nifti(dicom_folder, out_nifti_path, series_uid=None):
    scan = load_scan(dicom_folder, series_uid)
    Path(out_nifti_path).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(scan.image, str(out_nifti_path))
    return out_nifti_path
