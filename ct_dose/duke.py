"""Reproducible adapter for Duke's CC-BY-4.0 Zenodo 3579490 benchmark.

Raw encoding/orientation is an explicitly recorded interpretation of headerless
images, not recovered DICOM geometry. Use only for research.
"""
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import SimpleITK as sitk
from openpyxl import load_workbook

from .imaging import Scan, fingerprint
from .segmentation import annotate, SEGMENTATOR_VERSION

URL = "https://zenodo.org/api/records/3579490/files/CT_organ_dose_MC_database.zip/content"
MD5 = "f813e3186ae7eb22eb83ff2c33a095d7"
ARCHIVE = "CT_organ_dose_MC_database.zip"
PROFILE = "duke-3579490-abdo-fixed-v1"
LABELS = {"liver": "Liver (mGy)", "spleen": "Spleen (mGy)",
          "pancreas": "Pancreas (mGy)", "stomach": "Stomach (mGy)",
          "gallbladder": "Gall bladder (mGy)", "urinary_bladder": "Bladder (mGy)"}
FEATURES = ["organ", "volume_cm3", "mean_hu", "std_hu", "ctdivol_mGy", "kvp"]


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def acquire(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / ARCHIVE
    if not archive.exists():
        temporary = archive.with_suffix(".partial")
        with urllib.request.urlopen(URL, timeout=120) as response, temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream, 8 * 1024 * 1024)
        if digest(temporary, "md5") != MD5:
            raise ValueError("Duke archive checksum mismatch; partial download retained")
        temporary.replace(archive)
    if digest(archive, "md5") != MD5:
        raise ValueError("Duke archive checksum mismatch")
    extracted = directory / "extracted"
    extracted.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for item in bundle.infolist():
            if item.is_dir() or not (item.filename.startswith("Patient_Images/abdo/") or
                    item.filename == "Patient_Images/Patient_information.xlsx" or
                    item.filename.startswith("Organ_Dose/Abdo_Dose/") or
                    item.filename.startswith("CT_Acquisition_Condition/")):
                continue
            destination = (extracted / item.filename).resolve()
            if not destination.is_relative_to(extracted.resolve()):
                raise ValueError("Unsafe archive path")
            # Always replace extracted inputs from the verified archive, including on resume.
            destination.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(item) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
    return extracted


def patients(root):
    workbook = load_workbook(Path(root) / "Patient_Images/Patient_information.xlsx", data_only=True, read_only=True)
    rows = list(workbook.active.values)
    workbook.close()
    headers = rows[0]
    result = [dict(zip(headers, row)) for row in rows[1:]
              if isinstance(row[0], str) and row[0].startswith("pt")]
    ids = [r["Patient number"] for r in result]
    if len(ids) != len(set(ids)) or not result:
        raise ValueError("Missing or duplicate patient identities")
    if any(float(r["Age(Yr)"]) < 18 for r in result):
        raise ValueError("Dataset contains pediatric patients")
    return result


def raw_image(path, row):
    size = tuple(int(row[f"Abdo_{axis}_Pixel"]) for axis in "XYZ")
    spacing = (float(row["XY Pixel Size (mm/pixel)"]),) * 2 + (float(row["Z Pixel Size (mm/pixel)"]),)
    if min(size) <= 0 or not np.isfinite(spacing).all() or min(spacing) <= 0:
        raise ValueError("Invalid source geometry")
    if Path(path).stat().st_size != int(np.prod(size)) * 2:
        raise ValueError("Raw byte count does not match published dimensions")
    pixels = np.fromfile(path, dtype=">i2").reshape(size[::-1]).astype(np.int16)
    image = sitk.GetImageFromArray(pixels)
    image.SetSpacing(spacing)
    image.SetOrigin((0., 0., (size[2] - 1) * spacing[2]))
    image.SetDirection((1., 0., 0., 0., 1., 0., 0., 0., -1.))
    return image


def prepare(directory, output, limit_patients=None):
    root = acquire(directory)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    all_records = patients(root)
    if limit_patients is not None and (limit_patients < 1 or limit_patients > len(all_records)):
        raise ValueError("Patient limit must be between 1 and the available cohort size")
    records = all_records[:limit_patients]
    dose_path = root / "Organ_Dose/Abdo_Dose/Abdo_Fixed.xlsx"
    doses = pd.read_excel(dose_path).set_index("Patient")
    doses.columns = [" ".join(column.split()) for column in doses.columns]
    if not doses.columns.is_unique or not set(LABELS.values()) <= set(doses.columns):
        raise ValueError("Missing, duplicated, or changed organ-dose columns/units")
    if not doses.index.is_unique or set(doses.index) != {r["Patient number"] for r in all_records}:
        raise ValueError("Dose table and image patient identities do not match")
    rows, audit = [], []
    for index, row in enumerate(records, 1):
        patient = row["Patient number"]
        print(f"[{index}/{len(records)}] {patient}", flush=True)
        raw = root / "Patient_Images/abdo" / row["Abdo_image_name"]
        identity = {"raw_sha256": digest(raw), "metadata": row,
                    "adapter_version": 1, "segmentation_version": SEGMENTATOR_VERSION,
                    "resolution": "fast", "organs": list(LABELS)}
        cache = output / "patients" / patient
        cache.mkdir(parents=True, exist_ok=True)
        report = cache / "features.json"
        if report.exists():
            result = json.loads(report.read_text())
            if result["identity"] != identity:
                raise ValueError(f"Stale cache for {patient}; use a fresh output directory")
            for name, info in result["annotation"]["organs"].items():
                if info["status"] == "present" and fingerprint([cache / "annotation/raw_masks" / f"{name}.nii.gz"]) != info["input_sha256"]:
                    raise ValueError(f"Changed cached mask for {patient}/{name}")
        else:
            image = raw_image(raw, row)
            scan = Scan(image, [], identity["raw_sha256"], None)
            # A failed incomplete run can be restarted without trusting stale masks.
            if (cache / "annotation").exists():
                shutil.rmtree(cache / "annotation")
            annotation = annotate(scan, cache / "annotation", list(LABELS), fast=True)
            from .dosimetry import dose_report
            size = dose_report(scan)["quantities"]["water_equivalent_diameter"]
            result = {"identity": identity, "annotation": annotation, "body_size": size}
            report.write_text(json.dumps(result, indent=2, allow_nan=False))
        included = []
        for measure in result["annotation"]["measurements"]:
            organ = measure["organ"]
            if measure["coverage"] == "potentially_partial":
                continue
            target = float(doses.loc[patient, LABELS[organ]])
            if not np.isfinite(target) or target < 0:
                continue
            rows.append({"patient_id": patient, "acquisition_id": patient + "-abdo-fixed",
                         "protocol": PROFILE, **{k: measure[k] for k in FEATURES if k in measure},
                         "ctdivol_mGy": float(row["Abdominopelvic CTDIvol (mGy)"]), "kvp": 120.,
                         "water_equivalent_diameter_cm": result["body_size"]["value"], "dose_mGy": target})
            included.append(organ)
        audit.append({"patient_id": patient, "age": row["Age(Yr)"], "raw_sha256": identity["raw_sha256"],
                      "included_organs": included, "excluded_organs": sorted(set(LABELS) - set(included)),
                      "organ_status": result["annotation"]["organs"], "body_size": result["body_size"]})
    csv = output / "training.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    manifest = {"dataset_id": "Duke-Zenodo-3579490-abdo-fixed", "source_url": "https://zenodo.org/records/3579490",
                "license": "CC-BY-4.0", "attribution": "Ehsan Samei, Francesco Ria, Xiaoyu Tian, Paul W Segars (2019)",
                "reference_method": "validated_monte_carlo",
                "reference_evidence": "Publisher describes verified Monte Carlo organ doses; Abdo_Fixed.xlsx supplies mGy targets. Publisher validation is not independent validation of this surrogate.",
                "population": "adult", "region": "abdomen_pelvis", "data_sha256": digest(csv),
                "available_patients": len(all_records), "selected_patients": len(records),
                "selection_policy": "Published worksheet order; pilot convenience subset" if limit_patients else "All published abdominal patients",
                "archive_md5": MD5, "dose_workbook_sha256": digest(dose_path),
                "metadata_workbook_sha256": digest(root / "Patient_Images/Patient_information.xlsx"),
                "features": FEATURES, "protocol_profile": PROFILE,
                "experimental_only": True, "input_geometry_validation": "pending publisher confirmation",
                "segmentation_resolution": "fast", "segmentation_version": SEGMENTATOR_VERSION,
                "acquisition_scope": {"kvp": 120, "kvp_source": "Archive acquisition spectrum at 120 kVp", "tube_current": "fixed",
                                      "pitch": 0.8, "collimation_mm": 38.4, "source_isocentre_cm": 59.5,
                                      "ctdi_phantom_cm": 32, "scanner_model": "Published benchmark geometry/spectrum/bowtie only"},
                "image_decoding": {"dtype": ">i2", "layout": "zyx C order", "intensity": "signed HU without offset",
                                   "spacing": "Patient_information.xlsx unchanged", "orientation": "relative LPS, superior-to-inferior slices inferred visually; no source affine"},
                "limitations": [f"{len(records)} selected patients of {len(all_records)} available; internal holdout only", "No expert organ masks supplied",
                                "Raw-image encoding/orientation interpretation requires publisher confirmation",
                                "Published spacing used verbatim; unusually large derived volumes require investigation",
                                "Body diameter omitted from model features when unreliable; unavailable sizes retained in audit",
                                "No validation outside the published fixed-current acquisition or on real clinical deployments"],
                "patient_audit": audit}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    return manifest
