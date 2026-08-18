"""
train_organ_dose_model.py

Trains a RandomForest to predict organ dose from ORGAN-LEVEL features
(volume, mean HU, which organ it is, CTDIvol) using the pseudo-labels from
estimate_organ_dose.py as training targets.

If --cnn-embeddings is given, this also merges in the CNN's learned
whole-abdomen features (from train_cnn_dose_features.py) as extra columns.
The CNN saw the whole scan, not this specific organ -- so those columns
are the SAME for every organ row belonging to a given patient/phase; they
add whole-scan visual context (patient build, overall attenuation pattern,
etc.) that the tabular columns alone don't capture, on top of the
per-organ numbers.

This is a genuine trained model, not a lookup formula: at inference time
it does NOT use the ratio table directly -- it has learned a function of
volume/HU/organ-identity/CTDIvol(/CNN features) that approximates (and can
deviate from) the flat-ratio pseudo-labels.

Usage:
    python train_organ_dose_model.py --labels organ_dose_labels.csv
    python train_organ_dose_model.py --labels organ_dose_labels.csv --cnn-embeddings cnn_embeddings.csv
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
    parser.add_argument("--cnn-embeddings", default=None,
                         help="Optional path to cnn_embeddings.csv from train_cnn_dose_features.py")
    args = parser.parse_args()

    df = pd.read_csv(args.labels, dtype={"UID": str})
    df = df.dropna(subset=["volume_cm3", "mean_hu", "mean_ctdivol_mGy", "pseudo_label_dose_mGy"])

    feature_cols = ["organ_encoded", "volume_cm3", "mean_hu", "mean_ctdivol_mGy"]

    if args.cnn_embeddings:
        cnn = pd.read_csv(args.cnn_embeddings, dtype={"UID": str})
        cnn_cols = [c for c in cnn.columns if c.startswith("cnn_emb_")]
        before = len(df)
        df = df.merge(cnn, on=["UID", "PHASE"], how="inner")
        print(f"Merged CNN features: {len(df)}/{before} organ-rows matched "
              f"a patient/phase with a CNN embedding.")
        feature_cols = feature_cols + cnn_cols

    print(f"Training on {len(df)} organ-rows across {df['UID'].nunique()} patients.")

    organ_encoder = LabelEncoder()
    df["organ_encoded"] = organ_encoder.fit_transform(df["organ"])

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

    # Feature importance -- what's actually driving the model's predictions.
    # With 64 individual CNN columns this gets noisy to read one-by-one, so
    # report the CNN block's importance summed together, alongside the
    # tabular features individually.
    print("\nFeature importance:")
    importances = dict(zip(feature_cols, model.feature_importances_))
    cnn_total = sum(v for k, v in importances.items() if k.startswith("cnn_emb_"))
    tabular = {k: v for k, v in importances.items() if not k.startswith("cnn_emb_")}
    for name, imp in sorted(tabular.items(), key=lambda x: -x[1]):
        print(f"  {name:<20} {imp:.3f}")
    if args.cnn_embeddings:
        print(f"  {'cnn_features (all 64, summed)':<20} {cnn_total:.3f}")


if __name__ == "__main__":
    main()