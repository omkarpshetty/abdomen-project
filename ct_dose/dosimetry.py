"""Dose quantities with provenance. Missing inputs stay missing."""
from pathlib import Path
import math

import numpy as np
import pydicom
from scipy import ndimage


def quantity(value, unit, source, reason=None):
    return {"value": value, "unit": unit, "source": source,
            "status": "available" if value is not None else "unavailable", "reason": reason}


def header_metric(datasets, keyword, unit):
    values = []
    for ds in datasets:
        raw = getattr(ds, keyword, None)
        if raw is not None:
            try:
                val = float(raw)
            except (ValueError, TypeError):
                raise ValueError(f"Invalid {keyword}")
            if not math.isfinite(val) or val <= 0:
                raise ValueError(f"Invalid {keyword}")
            values.append(val)
    result = quantity(float(np.mean(values)) if values else None, unit, f"DICOM.{keyword}",
                      None if values else "Not present in selected series")
    result.update(samples=len(values), total_slices=len(datasets), complete=bool(datasets) and len(values) == len(datasets))
    if values:
        result.update(minimum=min(values), maximum=max(values))
    return result


def code(node):
    seq = getattr(node, "ConceptNameCodeSequence", [])
    return (str(seq[0].CodingSchemeDesignator), str(seq[0].CodeValue)) if seq else None


def walk(node):
    yield node
    for child in getattr(node, "ContentSequence", []):
        yield from walk(child)


def read_rdsr(paths):
    """Read coded CT irradiation-event containers (DICOM TID 10013).

    No free-text matching or assignment of study totals to individual series.
    """
    events = {}
    for path in paths:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        if str(getattr(ds, "SOPClassUID", "")) != "1.2.840.10008.5.1.4.1.1.88.67":
            raise ValueError("Expected an X-Ray Radiation Dose SR")
        for container in walk(ds):
            if code(container) != ("DCM", "113819"):
                continue
            event = {"study_uid": str(getattr(ds, "StudyInstanceUID", "")), "event_uid": None}
            for node in walk(container):
                concept = code(node)
                if concept == ("DCM", "113769"):
                    event["event_uid"] = str(getattr(node, "UID", ""))
                if concept == ("DCM", "113835"):
                    seq = getattr(node, "ConceptCodeSequence", [])
                    if seq and str(seq[0].CodingSchemeDesignator) == "DCM":
                        event["phantom_cm"] = {"113690": 16, "113691": 32}.get(str(seq[0].CodeValue))
                if concept in (("DCM", "113830"), ("DCM", "113838")):
                    name, expected_unit = ("ctdivol_mGy", "mGy") if concept[1] == "113830" else ("dlp_mGy_cm", "mGy.cm")
                    seq = getattr(node, "MeasuredValueSequence", [])
                    if not seq:
                        continue
                    units = getattr(seq[0], "MeasurementUnitsCodeSequence", [])
                    if not units or str(units[0].CodingSchemeDesignator) != "UCUM" or str(units[0].CodeValue) != expected_unit:
                        raise ValueError(f"Unsupported RDSR unit for {name}")
                    val = float(seq[0].NumericValue)
                    if not math.isfinite(val) or val < 0:
                        raise ValueError("Invalid RDSR dose value")
                    if name in event and event[name] != val:
                        raise ValueError("Conflicting dose quantities within an irradiation event")
                    event[name] = val
            if not event["event_uid"] or not event["study_uid"]:
                raise ValueError("RDSR irradiation event missing identity")
            key = (event["study_uid"], event["event_uid"])
            if key in events and events[key] != event:
                raise ValueError("Conflicting duplicate irradiation event")
            events[key] = event
    return list(events.values())


def size_profile(scan):
    """Largest body component per slice, excluding disconnected table components."""
    area = np.prod(scan.image.GetSpacing()[:2]) / 100.0
    profile, truncated = [], False
    for pixels in scan.array:
        labels, count = ndimage.label(pixels > -500)
        if not count:
            continue
        counts = np.bincount(labels.ravel()); counts[0] = 0
        mask = ndimage.binary_fill_holes(labels == counts.argmax())
        if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
            truncated = True
        water_area = float(np.sum(np.maximum(0, 1 + pixels[mask] / 1000)) * area)
        profile.append(2 * math.sqrt(water_area / math.pi))
    return profile, truncated


def dose_report(scan, rdsr_paths=(), overrides=None):
    overrides = overrides or {}
    metrics = {name: header_metric(scan.datasets, key, unit) for name, key, unit in [
        ("ctdivol", "CTDIvol", "mGy"), ("kvp", "KVP", "kV"), ("exposure", "Exposure", "mAs")
    ]}
    events = read_rdsr(rdsr_paths)
    studies = {str(getattr(ds, "StudyInstanceUID", "")) for ds in scan.datasets}
    event_ids = {str(ds.IrradiationEventUID) for ds in scan.datasets if hasattr(ds, "IrradiationEventUID")}
    matched = [e for e in events if e["study_uid"] in studies and e["event_uid"] in event_ids]
    complete_event_identity = bool(scan.datasets) and all(hasattr(ds, "IrradiationEventUID") for ds in scan.datasets)
    single = matched[0] if len(matched) == 1 and len(event_ids) == 1 and complete_event_identity else None
    phantoms = set()
    for ds in scan.datasets:
        for item in getattr(ds, "CTDIPhantomTypeCodeSequence", []):
            if str(getattr(item, "CodingSchemeDesignator", "")) == "DCM":
                value = {"113690": 16, "113691": 32}.get(str(item.CodeValue))
                if value is not None:
                    phantoms.add(value)
    if len(phantoms) > 1:
        raise ValueError("Conflicting CTDI phantom types in selected series")
    phantom = next(iter(phantoms)) if phantoms else None
    if single and "ctdivol_mGy" in single:
        metrics["ctdivol"] = quantity(single["ctdivol_mGy"], "mGy", "RDSR matched irradiation event")
        metrics["ctdivol"]["complete"] = True
        if single.get("phantom_cm") is not None:
            if phantom is not None and phantom != single["phantom_cm"]:
                raise ValueError("RDSR/header CTDI phantom conflict")
            phantom = single["phantom_cm"]
    metrics["dlp"] = quantity(single.get("dlp_mGy_cm") if single else None, "mGy.cm", "RDSR matched irradiation event",
                              None if single and "dlp_mGy_cm" in single else "No uniquely matched event DLP")
    for name, unit in [("ctdivol", "mGy"), ("dlp", "mGy.cm")]:
        if name in overrides:
            val = float(overrides[name])
            if not math.isfinite(val) or val <= 0 or not overrides.get("source"):
                raise ValueError("Dose overrides require positive finite values and --dose-source")
            metrics[name] = quantity(val, unit, "user: " + overrides["source"])
            metrics[name]["complete"] = True
    if "phantom_cm" in overrides:
        if overrides["phantom_cm"] not in (16, 32) or not overrides.get("source"):
            raise ValueError("Phantom override requires 16/32 cm and a source")
        phantom = overrides["phantom_cm"]
    profile, truncated = size_profile(scan)
    metrics["water_equivalent_diameter"] = quantity(
        float(np.mean(profile)) if profile and not truncated else None, "cm", "CT pixels: AAPM 220 water-equivalent area",
        "Truncated body outline" if truncated else (None if profile else "Body not measurable"))
    diameter = metrics["water_equivalent_diameter"]["value"]
    ctdi = metrics["ctdivol"]["value"]
    # AAPM 204 exponential fits (16/32 cm CTDI phantom); AAPM 220 Dw substitution.
    # Restrict to the published fit domain; no extrapolated result.
    ssde = None
    if diameter is not None and 8 <= diameter <= 45 and phantom in (16, 32) and ctdi is not None and metrics["ctdivol"].get("complete"):
        a, b = (1.874799, 0.03871313) if phantom == 16 else (3.704369, 0.03671937)
        ssde = ctdi * a * math.exp(-b * diameter)
    metrics["ssde"] = quantity(ssde, "mGy", "AAPM 204 fit / AAPM 220 Dw; mean-size approximation",
                               None if ssde is not None else "Requires complete CTDIvol, known phantom, untruncated Dw within 8–45 cm")
    metrics["ssde"]["phantom_cm"] = phantom
    metrics["effective_dose"] = quantity(None, "mSv", None, "No validated effective-dose method with sufficient tissue coverage configured")
    all_matched = [e for e in events if e["study_uid"] in studies]
    study_dlp = sum(e["dlp_mGy_cm"] for e in all_matched) if all_matched and all("dlp_mGy_cm" in e for e in all_matched) else None
    return {"selected_series_event_count": len(event_ids), "quantities": metrics, "irradiation_events": all_matched,
            "provided_events_dlp_sum": quantity(study_dlp, "mGy.cm", "Unique provided RDSR events in selected study",
                                                 "Completeness of study exposure history is not established"),
            "unmatched_rdsr_events": len(events) - len(all_matched),
            "clinical_validation": "pending", "warnings": ["Scan coverage is not proof of complete organ coverage",
            "SSDE is a size-specific estimate, not an individual organ dose"]}
