"""
dose/extract_dicom_params.py

Extracts real CT acquisition parameters directly from DICOM headers --
the piece missing from your dose model so far. These are the scanner
settings that actually drive dose (kVp, tube current, CTDIvol), not just
patient demographics.

Also computes an "estimated DLP" using the standard clinical formula:
    DLP = CTDIvol x scan length (cm)
This is a genuine CONVENTIONAL (non-AI) dose estimate -- comparing it
against the real DLP in your spreadsheet is exactly the "AI predictions
vs conventional metrics" comparison your project objectives ask for.
"""
import numpy as np
from src.dicom_io import load_dicom_series


def extract_acquisition_params(dicom_dir: str):
    volume_hu, datasets = load_dicom_series(dicom_dir)

    def _vals(tag_names):
        """Try multiple possible tag names (vendors differ) and return a
        list of floats, skipping slices where none of the names are present."""
        out = []
        for d in datasets:
            v = None
            for name in tag_names:
                v = getattr(d, name, None)
                if v is not None:
                    break
            if v is not None:
                try:
                    out.append(float(v))
                except (TypeError, ValueError):
                    pass
        return out

    kvps = _vals(["KVP"])
    currents = _vals(["XRayTubeCurrent"])
    ctdivols = _vals(["CTDIvol"])
    exposure_time = _vals(["ExposureTime"])
    exposure_mas = _vals(["Exposure"])
    pitch = _vals(["SpiralPitchFactor", "PitchFactor"])
    slice_thickness = _vals(["SliceThickness"])

    # Scan length from the actual z-range of ImagePositionPatient, which is
    # more reliable than slice_count x slice_thickness (accounts for gaps).
    z_positions = [float(getattr(d, "ImagePositionPatient", [0, 0, 0])[2]) for d in datasets]
    scan_length_cm = (max(z_positions) - min(z_positions)) / 10.0 if z_positions else None

    mean_ctdivol = float(np.mean(ctdivols)) if ctdivols else None
    estimated_dlp = (mean_ctdivol * scan_length_cm) if (mean_ctdivol and scan_length_cm) else None

    return {
        "mean_kvp": float(np.mean(kvps)) if kvps else None,
        "mean_tube_current_mA": float(np.mean(currents)) if currents else None,
        "mean_ctdivol_mGy": mean_ctdivol,
        "mean_exposure_time_ms": float(np.mean(exposure_time)) if exposure_time else None,
        "mean_exposure_mAs": float(np.mean(exposure_mas)) if exposure_mas else None,
        "pitch_factor": float(np.mean(pitch)) if pitch else None,
        "mean_slice_thickness_mm": float(np.mean(slice_thickness)) if slice_thickness else None,
        "scan_length_cm": scan_length_cm,
        "estimated_dlp_mgycm": estimated_dlp,  # conventional formula, no ML
    }