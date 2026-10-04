"""One physical image grid for CT, masks, features, and rendered slices."""
from dataclasses import dataclass
from pathlib import Path
import hashlib

import numpy as np
import pydicom
import SimpleITK as sitk


@dataclass
class Scan:
    image: sitk.Image
    datasets: list
    fingerprint: str
    series_uid: str | None

    @property
    def array(self):
        return sitk.GetArrayFromImage(self.image)


def fingerprint(paths):
    digest = hashlib.sha256()
    for path in sorted(paths, key=str):
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def discover(folder):
    """Header-only discovery: exclude SR, scouts and non-CT objects."""
    groups = {}
    for path in sorted(Path(folder).rglob("*")):
        if not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
        except (pydicom.errors.InvalidDicomError, OSError):
            continue
        if getattr(ds, "Modality", None) != "CT":
            continue
        if "LOCALIZER" in getattr(ds, "ImageType", []):
            continue
        uid = str(getattr(ds, "SeriesInstanceUID", ""))
        if not uid:
            raise ValueError("CT object missing SeriesInstanceUID")
        groups.setdefault(uid, []).append((path, ds))
    return groups


def load_scan(source, series_uid=None):
    source = Path(source)
    if source.is_file() and str(source).endswith((".nii", ".nii.gz")):
        if series_uid:
            raise ValueError("Series selection applies only to DICOM")
        image = sitk.ReadImage(str(source), sitk.sitkFloat32)
        if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
            raise ValueError("Expected a scalar 3D CT image in Hounsfield units")
        if not np.isfinite(sitk.GetArrayFromImage(image)).all():
            raise ValueError("CT contains non-finite intensities")
        return Scan(image, [], fingerprint([source]), None)
    groups = discover(source)
    if not groups:
        raise ValueError("No supported CT series found")
    if series_uid is None:
        if len(groups) != 1:
            raise ValueError("Multiple CT series found; select --series-uid using inspect")
        series_uid = next(iter(groups))
    if series_uid not in groups:
        raise ValueError("Selected CT series does not exist")
    entries = groups[series_uid]
    first = entries[0][1]
    required = ("ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing",
                "RescaleSlope", "RescaleIntercept", "Rows", "Columns", "SOPInstanceUID",
                "FrameOfReferenceUID", "StudyInstanceUID")
    for _, ds in entries:
        if int(getattr(ds, "NumberOfFrames", 1)) != 1:
            raise ValueError("Enhanced/multiframe CT requires conversion to a supported single-frame series")
        if any(not hasattr(ds, key) for key in required):
            raise ValueError("Missing CT geometry, identity, or HU calibration")
        if getattr(ds, "RescaleType", "HU") != "HU":
            raise ValueError("CT intensity calibration is not HU")
        if str(ds.StudyInstanceUID) != str(first.StudyInstanceUID):
            raise ValueError("Inconsistent CT study identity")
        if str(ds.FrameOfReferenceUID) != str(first.FrameOfReferenceUID):
            raise ValueError("Inconsistent CT frame of reference")
        if (ds.Rows, ds.Columns) != (first.Rows, first.Columns):
            raise ValueError("Inconsistent CT dimensions")
        if not np.allclose(ds.ImageOrientationPatient, first.ImageOrientationPatient, atol=1e-5):
            raise ValueError("Inconsistent CT orientation")
        if not np.allclose(ds.PixelSpacing, first.PixelSpacing, atol=1e-5):
            raise ValueError("Inconsistent pixel spacing")
    orientation = np.asarray(first.ImageOrientationPatient, dtype=float)
    row, col = orientation[:3], orientation[3:]
    if not (np.isclose(np.linalg.norm(row), 1) and np.isclose(np.linalg.norm(col), 1)
            and np.isclose(np.dot(row, col), 0, atol=1e-5)):
        raise ValueError("Invalid direction cosines")
    normal = np.cross(row, col)
    entries.sort(key=lambda item: np.dot(np.asarray(item[1].ImagePositionPatient, float), normal))
    positions = np.array([ds.ImagePositionPatient for _, ds in entries], float)
    if len(entries) < 2:
        raise ValueError("At least two CT slices required to establish spacing")
    offsets = positions @ normal
    gaps = np.diff(offsets)
    if np.any(gaps <= 1e-4) or not np.allclose(gaps, gaps[0], atol=1e-3, rtol=1e-3):
        raise ValueError("Duplicate slices or irregular slice spacing")
    if not np.allclose(positions - positions[0], np.outer(offsets - offsets[0], normal), atol=1e-3):
        raise ValueError("Gantry tilt/sheared stacks require an explicitly validated resampling step")
    ids = [str(ds.SOPInstanceUID) for _, ds in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate SOP instance")
    pixels = []
    datasets = []
    for path, _ in entries:
        ds = pydicom.dcmread(path)
        values = ds.pixel_array.astype(np.float32) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
        if not np.isfinite(values).all() or float(ds.RescaleSlope) <= 0:
            raise ValueError("Invalid HU calibration or pixel data")
        pixels.append(values)
        datasets.append(ds)
    image = sitk.GetImageFromArray(np.stack(pixels))
    spacing = (float(first.PixelSpacing[1]), float(first.PixelSpacing[0]), float(gaps[0]))
    if min(spacing) <= 0 or not np.isfinite(spacing).all():
        raise ValueError("Invalid voxel spacing")
    image.SetSpacing(spacing)
    image.SetOrigin(tuple(positions[0]))
    image.SetDirection(tuple(np.column_stack((row, col, normal)).ravel()))
    return Scan(image, datasets, fingerprint([p for p, _ in entries]), series_uid)


def aligned_mask(path, reference):
    """Resample categorical masks in physical space, never array-index space."""
    mask = sitk.ReadImage(str(path))
    if mask.GetDimension() != 3 or mask.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError(f"Not a scalar 3D mask: {path}")
    values = sitk.GetArrayFromImage(mask)
    if not np.isfinite(values).all() or not np.isin(values, [0, 1]).all():
        raise ValueError(f"Expected a binary organ mask: {path}")
    aligned = sitk.Resample(mask, reference, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    result = sitk.GetArrayFromImage(aligned).astype(bool)
    if values.any() and not result.any():
        raise ValueError(f"Mask does not overlap the CT grid: {path}")
    return result


def write_mask(array, reference, path):
    mask = sitk.GetImageFromArray(array.astype(np.uint8))
    mask.CopyInformation(reference)
    sitk.WriteImage(mask, str(path))
