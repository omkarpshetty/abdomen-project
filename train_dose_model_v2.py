"""
train_dose_model_v2.py

Merges dose labels + size features (WED) + DICOM acquisition params, then
trains THREE models to compare:
  1. Baseline:  age, gender, height, weight
  2. Imaging:   + water-equivalent diameter (from segmentation)
  3. Full:      + real acquisition params (kVp, tube current, CTDIvol)

Uses K-FOLD CROSS-VALIDATION (default 5 folds) rather than one train/test
split. With ~140 patients, a single 80/20 split has a lot of luck in it --
which ~28 patients happen to land in the test fold can swing R² noticeably
on its own. Reporting mean +/- std across folds is the honest number for a
dataset this size; report BOTH the mean and the spread, not just the mean.

Also prints a CONVENTIONAL (non-AI) comparison: how close does the plain
physics formula (CTDIvol x scan length = estimated DLP) get to the real
DLP, with zero machine learning involved. That's your true "conventional
metric" baseline for objective #4 -- the ML models above should beat it.
NOTE: mean_ctdivol_mGy is currently unavailable for this dataset (see
project limitations), so this conventional check has no valid data to run
on. It's left in so it activates automatically if that ever changes.

Usage:
    python train_dose_model_v2.py --labels "C:\\...\\1 - 144  excel (2).xlsx" --features size_features.csv --params dicom_params.csv --folds 5
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder


def load_labels(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    if df["AGE"].dtype == object:
        df = df.rename(columns={"GENDER": "AGE_tmp", "AGE": "GENDER_tmp"})
        df = df.rename(columns={"AGE_tmp": "AGE", "GENDER_tmp": "GENDER"})
        print("[note] detected swapped AGE/GENDER columns -- corrected automatically.")
    df["UID"] = df["UID"].astype(int).astype(str)
    return df


def run_model_cv(df: pd.DataFrame, feature_cols: list, target_col: str, label: str, n_splits: int):
    X = df[feature_cols].copy()
    if "GENDER" in X.columns:
        X["GENDER"] = LabelEncoder().fit_transform(X["GENDER"])
    X = X.fillna(X.mean(numeric_only=True))
    y = df[target_col]

    n_splits = min(n_splits, len(df))  # can't have more folds than samples
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    model = RandomForestRegressor(n_estimators=300, random_state=42)

    results = cross_validate(
        model, X, y, cv=cv,
        scoring={"MAE": "neg_mean_absolute_error", "R2": "r2"},
    )
    mae_scores = -results["test_MAE"]
    r2_scores = results["test_R2"]

    print(f"  {label:10s} -> MAE: {mae_scores.mean():6.2f} +/- {mae_scores.std():5.2f} mGy·cm   "
          f"R²: {r2_scores.mean():6.3f} +/- {r2_scores.std():.3f}   "
          f"(n={len(df)}, {n_splits}-fold CV)")
    print(f"               per-fold R²: [{', '.join(f'{v:.2f}' for v in r2_scores)}]")
    return mae_scores, r2_scores


def conventional_formula_check(df: pd.DataFrame, target_col: str):
    """Zero-ML comparison: how good is CTDIvol x scan_length alone? Deterministic
    formula -- no train/test split or CV needed, it's not a fitted model."""
    valid = df.dropna(subset=["estimated_dlp_mgycm", target_col])
    if len(valid) < 5:
        print("  [conventional] not enough data with estimated_dlp_mgycm to compare.")
        return
    mae = mean_absolute_error(valid[target_col], valid["estimated_dlp_mgycm"])
    r2 = r2_score(valid[target_col], valid["estimated_dlp_mgycm"])
    print(f"  {'Conventional':10s} -> MAE: {mae:6.2f} mGy·cm   R²: {r2:.3f}   "
          f"(n={len(valid)}, formula: CTDIvol x scan length, NO ML)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    labels = load_labels(args.labels)
    features = pd.read_csv(args.features, dtype={"UID": str})
    params = pd.read_csv(args.params, dtype={"UID": str})

    for target, phase in [("DLP -PLAIN", "plain"), ("DLP-VENOUS", "venous")]:
        print(f"\n=== Predicting {target} ({phase}-phase) ===")
        pf = features[features["PHASE"] == phase]
        pp = params[params["PHASE"] == phase]
        merged = labels.merge(pf, on="UID", how="inner").merge(pp, on="UID", how="inner", suffixes=("", "_p"))
        print(f"  Matched {len(merged)} patients.")
        if len(merged) < 10:
            print("  [warn] too few matched patients -- skipping.")
            continue

        conventional_formula_check(merged, target)

        baseline_cols = ["AGE", "GENDER", "HEIGHT", "WEIGHT"]
        imaging_cols = baseline_cols + ["mean_water_equiv_diameter_cm"]
        full_cols = imaging_cols + ["mean_kvp", "mean_tube_current_mA", "mean_ctdivol_mGy",
                                     "pitch_factor", "scan_length_cm"]
        full_cols = [c for c in full_cols if c in merged.columns]  # drop any missing columns safely

        run_model_cv(merged, baseline_cols, target, "Baseline", args.folds)
        run_model_cv(merged, imaging_cols, target, "Imaging", args.folds)
        run_model_cv(merged, full_cols, target, "Full", args.folds)


if __name__ == "__main__":
    main()