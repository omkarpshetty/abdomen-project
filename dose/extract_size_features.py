"""
dose/extract_size_features.py

Computes patient body-size metrics directly from a DICOM series --
NO TotalSegmentator call needed (this is intentionally lightweight so it
can run across all 144 patients quickly, unlike the full organ annotation
pipeline in main.py, which you should only run on a handful of patients
for figures, not the whole dataset).

Metrics computed, per AAPM Report 204/220 (standard patient-size dose
correction methodology):
  - effective_diameter_cm: size based purely on body outline area
  - water_equiv_diameter_cm: size corrected for actual tissue density
    (mean HU inside the body) -- this is the stronger predictor for dose
    modeling since it accounts for fat vs muscle vs bone content, not
    just outline area.
"""
import numpy as np

from src.dicom_io import load_dicom_series
from src.segment import derive_body_mask


def compute_size_features(dicom_dir: str):
    """
    Returns a dict of per-patient size metrics averaged across all slices
    in the series. Raises the same exceptions as load_dicom_series if the
    folder has no readable DICOM files.
    """
    volume_hu, dicom_slices = load_dicom_series(dicom_dir)
    body_mask = derive_body_mask(volume_hu, hu_threshold=-500)

    # Pixel spacing (mm) -- same for every slice in a series in practice.
    row_spacing, col_spacing = [float(x) for x in dicom_slices[0].PixelSpacing]
    pixel_area_cm2 = (row_spacing * col_spacing) / 100.0  # mm^2 -> cm^2

    eff_diams = []
    wed_diams = []
    for z in range(volume_hu.shape[0]):
        slice_mask = body_mask[z]
        n_pixels = int(slice_mask.sum())
        if n_pixels == 0:
            continue  # empty slice (e.g. above/below the patient), skip

        area_cm2 = n_pixels * pixel_area_cm2

        # AAPM 204: effective diameter from cross-sectional area alone.
        effective_diameter = 2.0 * np.sqrt(area_cm2 / np.pi)

        # AAPM 220: water-equivalent diameter corrects for actual tissue
        # density inside the body (mean HU), not just outline size --
        # this is what makes two same-sized patients with different fat/
        # muscle content get different, more accurate size estimates.
        mean_hu = float(volume_hu[z][slice_mask].mean())
        water_equiv_diameter = 2.0 * np.sqrt(((mean_hu / 1000.0) + 1.0) * area_cm2 / np.pi)

        eff_diams.append(effective_diameter)
        wed_diams.append(water_equiv_diameter)

    if not eff_diams:
        raise ValueError(f"No non-empty body slices found in {dicom_dir}")

    return {
        "n_slices": len(eff_diams),
        "mean_effective_diameter_cm": float(np.mean(eff_diams)),
        "mean_water_equiv_diameter_cm": float(np.mean(wed_diams)),
    }