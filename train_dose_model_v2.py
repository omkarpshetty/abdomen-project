"""
train_dose_model_v2.py

Merges dose labels + size features (WED) + DICOM acquisition params, then
trains THREE models to compare:
  1. Baseline:  age, gender, height, weight
  2. Imaging:   + water-equivalent diameter (from segmentation)
  3. Full:      + real acquisition params (kVp, tube current, CTDIvol)

Also prints a CONVENTIONAL (non-AI) comparison: how close does the plain
physics formula (CTDIvol x scan length = estimated DLP) get to the real
DLP, with zero machine learning involved. That's your true "conventional
metric" baseline for objective #4 -- the ML models above should beat it.

Usage:
    python train_dose_model_v2.py --labels "C:\\...\\1 - 144  excel (2).xlsx" --features size_features.csv --params dicom_params.csv
"""
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
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


def run_model(df: pd.DataFrame, feature_cols: list, target_col: str, label: str):
    X = df[feature_cols].copy()
    if "GENDER" in X.columns:
        X["GENDER"] = LabelEncoder().fit_transform(X["GENDER"])
    X = X.fillna(X.mean(numeric_only=True))
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"  {label:10s} -> MAE: {mae:6.2f} mGy·cm   R²: {r2:.3f}   (n={len(df)})")
    return mae, r2


def conventional_formula_check(df: pd.DataFrame, target_col: str):
    """Zero-ML comparison: how good is CTDIvol x scan_length alone?"""
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

        run_model(merged, baseline_cols, target, "Baseline")
        run_model(merged, imaging_cols, target, "Imaging")
        run_model(merged, full_cols, target, "Full")


if __name__ == "__main__":
    main()