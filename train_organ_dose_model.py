"""
train_organ_dose_model.py

Trains a RandomForest to predict organ dose from ORGAN-LEVEL features
(volume, mean HU, which organ it is, CTDIvol) using the pseudo-labels from
estimate_organ_dose.py as training targets.

This is a genuine trained model, not a lookup formula: at inference time
it does NOT use the ratio table directly -- it has learned a function of
volume/HU/organ-identity/CTDIvol that approximates (and can deviate from)
the flat-ratio pseudo-labels, picking up patterns the formula can't
express (e.g. how dose scales with THIS patient's actual organ size).

Usage:
    python train_organ_dose_model.py --labels organ_dose_labels.csv
"""
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, help="Path to organ_dose_labels.csv from estimate_organ_dose.py")
    args = parser.parse_args()

    df = pd.read_csv(args.labels)
    df = df.dropna(subset=["volume_cm3", "mean_hu", "mean_ctdivol_mGy", "pseudo_label_dose_mGy"])
    print(f"Training on {len(df)} organ-rows across {df['UID'].nunique()} patients.")

    organ_encoder = LabelEncoder()
    df["organ_encoded"] = organ_encoder.fit_transform(df["organ"])

    feature_cols = ["organ_encoded", "volume_cm3", "mean_hu", "mean_ctdivol_mGy"]
    X = df[feature_cols]
    y = df["pseudo_label_dose_mGy"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"\nOverall -> MAE: {mae:.2f} mGy   R²: {r2:.3f}   (n_test={len(X_test)})")

    # Per-organ breakdown -- shows which organs the model predicts well vs poorly
    test_df = df.loc[X_test.index].copy()
    test_df["predicted_dose_mGy"] = preds
    print("\nPer-organ performance on test set:")
    for organ, group in test_df.groupby("organ"):
        if len(group) < 2:
            continue
        organ_mae = mean_absolute_error(group["pseudo_label_dose_mGy"], group["predicted_dose_mGy"])
        print(f"  {organ:<18} MAE: {organ_mae:6.2f} mGy   (n={len(group)})")

    # Feature importance -- what's actually driving the model's predictions
    print("\nFeature importance:")
    for name, imp in sorted(zip(feature_cols, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {name:<20} {imp:.3f}")


if __name__ == "__main__":
    main()
    