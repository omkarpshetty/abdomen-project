"""Supported CLI: python -m ct_dose {inspect,annotate,predict,train,evaluate}."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from .imaging import discover, load_scan
from .dosimetry import dose_report
from .segmentation import annotate, ORGANS
from . import __version__


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def parser():
    root = argparse.ArgumentParser(description="Adult abdomen/pelvis CT pipeline; clinical validation pending")
    commands = root.add_subparsers(dest="command", required=True)
    inspection = commands.add_parser("inspect", help="List DICOM series without loading pixels")
    inspection.add_argument("input")
    for command in ("annotate", "predict"):
        p = commands.add_parser(command)
        p.add_argument("input", help="DICOM directory or HU-calibrated NIfTI")
        p.add_argument("--series-uid")
        p.add_argument("--output", required=True, help="New, empty output directory")
        p.add_argument("--organs", nargs="+", choices=list(ORGANS), default=None)
        p.add_argument("--fast", action="store_true", help="Explicit lower-resolution segmentation")
        p.add_argument("--masks", help="Import individual binary NIfTI masks instead of running a model")
        p.add_argument("--corrections", help="Expert binary NIfTI masks named by organ")
        p.add_argument("--rdsr", nargs="*", default=[])
        p.add_argument("--ctdivol", type=float)
        p.add_argument("--kvp", type=float, help="Documented acquisition kVp override; requires --dose-source")
        p.add_argument("--dlp", type=float)
        p.add_argument("--phantom-cm", type=int, choices=[16, 32])
        p.add_argument("--dose-source", help="Provenance for explicit dose/phantom overrides")
        p.add_argument("--adult-confirmed", action="store_true", help="Confirm adult population when age cannot be established")
        if command == "predict":
            p.add_argument("--experimental-model", action="store_true", help="Allow provisional trained models in a separate experimental report")
            p.add_argument("--protocol-profile", help="Confirm acquisition matches the model manifest profile")
            p.add_argument("--model", help="Trusted local artifact from this pipeline's train command")
            p.add_argument("--exploratory", action="store_true", help="Write separate, unvalidated legacy formula estimates")
    dataset = commands.add_parser("prepare-duke", help="Download/check Duke public CT dose references and extract AI features on CPU")
    dataset.add_argument("--limit-patients", type=int, help="Explicit convenience subset for a quick experimental pilot; omitted means all patients")
    dataset.add_argument("--data-dir", required=True, help="Download/cache directory (several GB)")
    dataset.add_argument("--output", required=True, help="Resumable feature extraction directory")
    training = commands.add_parser("train")
    training.add_argument("--data", required=True)
    training.add_argument("--manifest", required=True)
    training.add_argument("--output", required=True)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--kind", choices=["segmentation", "dose"], required=True)
    evaluation.add_argument("--input")
    evaluation.add_argument("--series-uid")
    evaluation.add_argument("--predicted")
    evaluation.add_argument("--reference")
    evaluation.add_argument("--organs", nargs="+")
    evaluation.add_argument("--data", help="Paired CSV: dose_mGy,predicted_dose_mGy")
    evaluation.add_argument("--output", required=True)
    return root


def adult_check(scan, confirmed):
    ages = set()
    for ds in scan.datasets:
        value = str(getattr(ds, "PatientAge", ""))
        if value and len(value) == 4 and value[:3].isdigit() and value[-1] in "DWMY":
            years = int(value[:3]) / {"D": 365.25, "W": 52.18, "M": 12, "Y": 1}[value[-1]]
            ages.add(years)
    if ages and min(ages) < 18:
        raise ValueError("Pediatric scans are outside this pipeline's supported population")
    if not ages and not confirmed:
        raise ValueError("Adult age unavailable: confirm eligibility with --adult-confirmed")


def run_scan(args):
    destination = Path(args.output)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Output directory must be empty; prevents stale masks and cross-patient results")
    scan = load_scan(args.input, args.series_uid)
    adult_check(scan, args.adult_confirmed)
    overrides = {name: getattr(args, name) for name in ("ctdivol", "dlp", "phantom_cm", "kvp") if getattr(args, name) is not None}
    overrides["source"] = args.dose_source
    doses = dose_report(scan, args.rdsr, overrides)
    result = {"schema_version": 1, "pipeline_version": __version__, "scan_sha256": scan.fingerprint,
              "series_uid": scan.series_uid, "population": "adult", "adult_confirmed_by_operator": args.adult_confirmed, "clinical_validation": "pending",
              "geometry": {"size_xyz": list(scan.image.GetSize()), "spacing_mm_xyz": list(scan.image.GetSpacing()),
                           "origin_lps_mm": list(scan.image.GetOrigin()), "direction_lps": list(scan.image.GetDirection())},
              "dependencies": {name: version(name) for name in ["numpy", "pydicom", "SimpleITK"]},
              "dose": doses}
    destination.mkdir(parents=True, exist_ok=True)
    try:
        result.update(annotate(scan, destination, args.organs, args.fast, args.masks, args.corrections))
    except Exception as exc:
        result["status"] = "annotation_failed"
        result["error"] = str(exc)
        dump(destination / "report.json", result)
        raise
    pd.DataFrame(result["measurements"]).to_csv(destination / "organ_measurements.csv", index=False)
    result["organ_dose"] = {"status": "unavailable", "reason": "No qualified reference-trained model supplied"}
    if args.command == "predict":
        q = doses["quantities"]
        features = pd.DataFrame(result["measurements"])
        for name, metric in [("ctdivol_mGy", "ctdivol"), ("water_equivalent_diameter_cm", "water_equivalent_diameter"), ("kvp", "kvp")]:
            features[name] = q[metric]["value"]
        features.to_csv(destination / "prediction_features.csv", index=False)
        if args.model:
            from .modeling import predict
            try:
                if doses["selected_series_event_count"] > 1:
                    raise ValueError("Multiple irradiation events in selected series; organ prediction withheld")
                if any(row["coverage"] == "potentially_partial" for row in result["measurements"]):
                    raise ValueError("Organ touches image boundary; complete-organ prediction withheld")
                if not q["ctdivol"].get("complete") or not q["kvp"].get("complete"):
                    raise ValueError("Incomplete CTDIvol or kVp metadata")
                values, meta = predict(features, args.model, context={
                    "protocol_profile": args.protocol_profile, "allow_experimental": args.experimental_model,
                    "ctdi_phantom_cm": q["ssde"].get("phantom_cm"),
                    "segmentation_resolution": result["segmentation"]["resolution"],
                    "segmentation_version": result["segmentation"]["version"]})
                predictions = [{"organ": organ, "dose_mGy": float(value)} for organ, value in zip(features.organ, values)]
                result["organ_dose"] = {"status": "research_prediction", "clinical_validation": "pending", "values": predictions,
                                        "model_sha256": meta["model_sha256"], "reference_dataset": meta["dataset"]["dataset_id"],
                                        "uncertainty": "Not calibrated; no confidence intervals claimed"}
                if meta["dataset"].get("experimental_only"):
                    result["organ_dose"]["status"] = "experimental_model_prediction"
                    result["organ_dose"]["limitations"] = meta["dataset"].get("limitations", [])
                    dump(destination / "experimental_model.json", result["organ_dose"])
                    pd.DataFrame(predictions).to_csv(destination / "experimental_organ_doses.csv", index=False)
                    result["organ_dose"] = {"status": "unavailable", "reason": "Dataset geometry unresolved; provisional model estimates exported separately"}
                else:
                    pd.DataFrame(predictions).to_csv(destination / "organ_doses.csv", index=False)
            except (ValueError, OSError, KeyError) as exc:
                result["organ_dose"] = {"status": "unavailable", "reason": str(exc)}
        if args.exploratory:
            # Preserve the old formula only as an explicitly unvalidated comparison.
            factors = {"liver": 1.2, "spleen": 1.1, "kidney_left": 1.0, "kidney_right": 1.0,
                       "pancreas": 1.0, "stomach": .9, "gallbladder": 1.0, "heart": 1.1,
                       "aorta": .7, "urinary_bladder": .9, "spinal_cord": 1.3}
            kvp = q["kvp"]["value"]
            values = []
            if kvp is not None:
                for row in result["measurements"]:
                    if row["organ"] in factors:
                        estimate = kvp * scan.image.GetSize()[2] * .001 * factors[row["organ"]] * (1 + row["mean_hu"] / 1000)
                        values.append({"organ": row["organ"], "estimate_mGy": max(0, estimate)})
            dump(destination / "exploratory.json", {"schema_version": 1, "status": "unvalidated_exploratory",
                 "method": "legacy kVp × slice count × organ coefficient × HU correction",
                 "limitations": ["Not calibrated to measured organ doses", "Depends on reconstruction slice count",
                                 "Coefficients have no verified reference provenance", "Excluded from clinical and primary dose reports"],
                 "values": values, "reason": None if values else "Required measured kVp or supported organ absent"})
    result["status"] = "completed_with_limitations"
    dump(destination / "report.json", result)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "inspect":
            groups = discover(args.input)
            print(json.dumps({"series": [{"series_uid": uid, "slices": len(items)} for uid, items in groups.items()]}, indent=2))
            if not groups:
                return 2
        elif args.command in ("annotate", "predict"):
            result = run_scan(args)
            print(f"Report: {Path(args.output) / 'report.json'}")
            print(f"Organ dose: {result['organ_dose']['status']}; clinical validation pending")
        elif args.command == "prepare-duke":
            from .duke import prepare
            result = prepare(args.data_dir, args.output, args.limit_patients)
            print(f"Prepared dataset: {result['dataset_id']}")
        elif args.command == "train":
            from .modeling import train
            result = train(args.data, args.manifest, args.output)
            print(json.dumps({"selected_model": result["selected_model"], "test_metrics": result["test_metrics"]}, indent=2))
        else:
            from .evaluation import evaluate_masks, regression_metrics
            if args.kind == "segmentation":
                if not all([args.input, args.predicted, args.reference, args.organs]):
                    raise ValueError("Segmentation evaluation requires --input, --predicted, --reference, --organs")
                result = evaluate_masks(load_scan(args.input, args.series_uid), args.predicted, args.reference, args.organs)
            else:
                if not args.data:
                    raise ValueError("Dose evaluation requires --data")
                frame = pd.read_csv(args.data)
                result = {"overall": regression_metrics(frame.dose_mGy, frame.predicted_dose_mGy)}
                if "water_equivalent_diameter_cm" in frame:
                    frame["size_group"] = pd.cut(frame.water_equivalent_diameter_cm, [0, 25, 35, np.inf], labels=["under_25_cm", "25_to_35_cm", "over_35_cm"])
                for column in ("organ", "protocol", "size_group"):
                    if column in frame:
                        result[column] = {str(key): regression_metrics(group.dose_mGy, group.predicted_dose_mGy)
                                          for key, group in frame.groupby(column, observed=True)}
            dump(args.output, {"schema_version": 1, "metrics": result, "independent_validation": "not_established"})
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
