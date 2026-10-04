"""Patient-disjoint dose training and a strict, versioned inference contract."""
from importlib.metadata import version
from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .evaluation import regression_metrics

FEATURES = ["organ", "volume_cm3", "mean_hu", "std_hu", "ctdivol_mGy", "water_equivalent_diameter_cm", "kvp"]
NUMERIC = FEATURES[1:]
ORGAN_FEATURES = [name for name in FEATURES if name != "water_equivalent_diameter_cm"]

def feature_schema(features):
    if features not in (FEATURES, ORGAN_FEATURES):
        raise ValueError("Incompatible model feature schema")
    return features


DEPENDENCIES = ["numpy", "scikit-learn", "xgboost", "joblib"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def within_bounds(values, low, high):
    """Allow only float serialization noise (e.g. NIfTI float32 spacing)."""
    return ((values >= low) | np.isclose(values, low, rtol=1e-6, atol=1e-8)) & ((values <= high) | np.isclose(values, high, rtol=1e-6, atol=1e-8))


def validate_features(frame, features=None):
    features = feature_schema(features or FEATURES)
    if not set(features) <= set(frame.columns) or frame.empty:
        raise ValueError(f"Required feature columns: {features}")
    if frame["organ"].isna().any() or (frame["organ"].astype(str).str.len() == 0).any():
        raise ValueError("Organ identity missing")
    numeric = frame[features[1:]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Missing/nonfinite dose features; no fabricated defaults are allowed")
    for name in ["volume_cm3", "ctdivol_mGy", "water_equivalent_diameter_cm", "kvp"]:
        if name in features and (frame[name] <= 0).any():
            raise ValueError(f"{name} must be positive")
    if (frame.std_hu < 0).any():
        raise ValueError("HU standard deviation cannot be negative")


def train(csv_path, manifest_path, output):
    from xgboost import XGBRegressor
    manifest = json.loads(Path(manifest_path).read_text())
    required = ["dataset_id", "source_url", "license", "reference_method", "reference_evidence", "population", "region", "data_sha256"]
    if any(not manifest.get(key) for key in required):
        raise ValueError(f"Reference manifest requires {required}")
    if manifest["reference_method"] not in ("measured", "validated_monte_carlo"):
        raise ValueError("Formula-generated labels cannot qualify organ-dose training")
    if manifest["population"] != "adult" or manifest["region"] != "abdomen_pelvis":
        raise ValueError("Only adult abdomen/pelvis reference data is supported")
    if manifest["data_sha256"] != sha256(csv_path):
        raise ValueError("Reference dataset checksum mismatch")
    frame = pd.read_csv(csv_path, dtype={"patient_id": str, "acquisition_id": str})
    features = feature_schema(manifest.get("features", FEATURES))
    validate_features(frame, features)
    required_rows = ["patient_id", "acquisition_id", "dose_mGy", "protocol"]
    if not set(required_rows) <= set(frame.columns) or frame[required_rows].isna().any().any():
        raise ValueError(f"Reference rows require {required_rows}")
    if frame.duplicated(["patient_id", "acquisition_id", "organ"]).any():
        raise ValueError("Duplicate patient/acquisition/organ reference rows")
    if not np.isfinite(frame.dose_mGy).all() or (frame.dose_mGy < 0).any():
        raise ValueError("Invalid reference doses")
    if frame.protocol.nunique() != 1:
        raise ValueError("This feature schema cannot distinguish multiple acquisition protocols; train one protocol per model")
    if manifest.get("protocol_profile") and set(frame.protocol) != {manifest["protocol_profile"]}:
        raise ValueError("Protocol rows disagree with dataset manifest")
    if frame.patient_id.nunique() < 10:
        raise ValueError("At least 10 independent patients required for three-way benchmarking; this is not a clinical sample-size criterion")
    dev, test = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(frame, groups=frame.patient_id))
    train_idx, val_idx = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=43).split(frame.iloc[dev], groups=frame.iloc[dev].patient_id))
    train_idx, val_idx = dev[train_idx], dev[val_idx]
    organs = set(frame.iloc[train_idx].organ)
    if not set(frame.organ) <= organs:
        raise ValueError("Some organs absent from training split; increase reference coverage")
    candidates = {
        "random_forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=1),
        "xgboost": XGBRegressor(n_estimators=200, max_depth=4, learning_rate=.05, objective="reg:squarederror", random_state=42, n_jobs=1),
    }
    pipelines, validation = {}, {}
    for name, estimator in candidates.items():
        transform = ColumnTransformer([("organ", OneHotEncoder(handle_unknown="error", sparse_output=False), ["organ"]),
                                       ("numeric", "passthrough", features[1:])])
        pipeline = Pipeline([("features", transform), ("regressor", estimator)])
        pipeline.fit(frame.iloc[train_idx][features], frame.iloc[train_idx].dose_mGy)
        validation[name] = regression_metrics(frame.iloc[val_idx].dose_mGy, pipeline.predict(frame.iloc[val_idx][features]))
        pipelines[name] = pipeline
    baseline_means = frame.iloc[train_idx].groupby("organ").dose_mGy.mean()
    baseline_validation = regression_metrics(frame.iloc[val_idx].dose_mGy, frame.iloc[val_idx].organ.map(baseline_means))
    selected = min(validation, key=lambda name: validation[name]["mae_mGy"])
    model = pipelines[selected]
    test_frame = frame.iloc[test].copy()
    test_frame["predicted_dose_mGy"] = model.predict(test_frame[features])
    baseline_test = regression_metrics(test_frame.dose_mGy, test_frame.organ.map(baseline_means))
    test_metrics = regression_metrics(test_frame.dose_mGy, test_frame.predicted_dose_mGy)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    joblib.dump(model, output / "model.joblib")
    # Save numeric training bounds to detect unsupported extrapolation at inference.
    bounds = {name: [float(frame.iloc[train_idx][name].min()), float(frame.iloc[train_idx][name].max())] for name in features[1:]}
    organ_bounds = {organ: {name: [float(group[name].min()), float(group[name].max())] for name in features[1:]}
                    for organ, group in frame.iloc[train_idx].groupby("organ")}
    diameter = test_frame.get("water_equivalent_diameter_cm", pd.Series(np.nan, index=test_frame.index))
    test_frame["size_group"] = pd.cut(diameter, [0, 25, 35, np.inf], labels=["under_25_cm", "25_to_35_cm", "over_35_cm"]).astype(object).fillna("unavailable")
    test_frame["within_training_bounds"] = np.logical_and.reduce([
        within_bounds(test_frame[name], low, high).to_numpy() for name, (low, high) in bounds.items()])
    for organ, limits in organ_bounds.items():
        for name, (low, high) in limits.items():
            test_frame.loc[test_frame.organ == organ, "within_training_bounds"] &= within_bounds(test_frame.loc[test_frame.organ == organ, name], low, high)
    eligible = test_frame[test_frame.within_training_bounds]
    deployment_metrics = regression_metrics(eligible.dose_mGy, eligible.predicted_dose_mGy) if len(eligible) else None
    stratified = {}
    for column in ["organ", "protocol", "size_group"]:
        stratified[column] = {str(key): regression_metrics(group.dose_mGy, group.predicted_dose_mGy)
                              for key, group in test_frame.groupby(column, observed=True)}
    metadata = {"schema_version": 1, "features": features, "feature_units": {name: {"organ": "categorical", "volume_cm3": "cm3", "mean_hu": "HU", "std_hu": "HU", "ctdivol_mGy": "mGy", "water_equivalent_diameter_cm": "cm", "kvp": "kV"}[name] for name in features}, "organs": sorted(organs), "bounds": bounds, "organ_bounds": organ_bounds, "bound_tolerance": {"relative": 1e-6, "absolute": 1e-8, "purpose": "Floating-point serialization noise only"},
                "dependencies": {name: version(name) for name in DEPENDENCIES}, "model_sha256": sha256(output / "model.joblib"),
                "dataset": manifest, "selected_model": selected,
                "training_configuration": {"seed": 42, "validation_split_seed": 43, "preprocessing": "Organ one-hot; numeric passthrough; fit on training patients only",
                                           "candidate_parameters": {"random_forest": {"n_estimators": 200, "min_samples_leaf": 2, "n_jobs": 1}, "xgboost": {"n_estimators": 200, "max_depth": 4, "learning_rate": .05, "objective": "reg:squarederror", "n_jobs": 1}}}, "validation_metrics": validation,
                "test_metrics": test_metrics,
                "test_metric_scope": "All held-out rows, including extrapolation; inference rejects out-of-range batches",
                "in_range_test_rows": len(eligible), "in_range_test_metrics": deployment_metrics,
                "baseline": {"method": "Training-only per-organ mean", "per_organ_mean_mGy": baseline_means.to_dict(), "validation": baseline_validation, "test": baseline_test,
                             "selected_model_beats_validation_baseline": validation[selected]["mae_mGy"] < baseline_validation["mae_mGy"]}, "test_stratified": stratified,
                "splits": {name: sorted(frame.iloc[idx].patient_id.unique().tolist()) for name, idx in [("train", train_idx), ("validation", val_idx), ("test", test)]},
                "clinical_validation": "pending", "status": "experimental_geometry_unverified" if manifest.get("experimental_only") else "reference_benchmarked_research_model"}
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
    test_frame.to_csv(output / "test_predictions.csv", index=False)
    return metadata


def predict(frame, model_dir, context=None):
    path = Path(model_dir)
    meta = json.loads((path / "manifest.json").read_text())
    features = feature_schema(meta.get("features"))
    validate_features(frame, features)
    if meta.get("schema_version") != 1:
        raise ValueError("Incompatible model feature schema")
    if meta.get("dataset", {}).get("reference_method") not in ("measured", "validated_monte_carlo"):
        raise ValueError("Model does not have qualified reference-data provenance")
    requirements = meta.get("dataset", {})
    context = context or {}
    for key in ("protocol_profile", "segmentation_resolution", "segmentation_version"):
        if requirements.get(key) and context.get(key) != requirements[key]:
            raise ValueError(f"Model requires matching {key}: {requirements[key]}")
    phantom = requirements.get("acquisition_scope", {}).get("ctdi_phantom_cm")
    if phantom is not None and context.get("ctdi_phantom_cm") != phantom:
        raise ValueError(f"Model requires CTDI phantom {phantom} cm")
    if requirements.get("segmentation_version") and "source" in frame:
        if not frame.source.eq("TotalSegmentator " + requirements["segmentation_version"]).all():
            raise ValueError("Mask provenance differs from model training preprocessing")
    if requirements.get("experimental_only") and not context.get("allow_experimental"):
        raise ValueError("Experimental model: unresolved dataset geometry; explicitly allow experimental use")
    if meta.get("dependencies") != {name: version(name) for name in DEPENDENCIES}:
        raise ValueError("Model runtime differs from its training manifest")
    if meta.get("model_sha256") != sha256(path / "model.joblib"):
        raise ValueError("Model checksum mismatch")
    if not set(frame.organ) <= set(meta["organs"]):
        raise ValueError("Model does not support all requested organs")
    outside = [name for name, (low, high) in meta["bounds"].items() if not within_bounds(frame[name], low, high).all()]
    for organ, group in frame.groupby("organ"):
        for name, (low, high) in meta.get("organ_bounds", {}).get(organ, {}).items():
            if not within_bounds(group[name], low, high).all():
                outside.append(f"{organ}.{name}")
    if outside:
        raise ValueError(f"Features outside training range: {outside}")
    # Only load locally trusted artifacts. Hashes provide integrity, not authentication.
    model = joblib.load(path / "model.joblib")
    values = np.asarray(model.predict(frame[features]), dtype=float)
    if values.shape != (len(frame),) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid model predictions")
    return values, meta
