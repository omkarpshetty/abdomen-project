"""
train_ensemble_model.py

Trains the RF + XGBoost stacked ensemble for organ-specific CT dose estimation
and saves the model to disk for use by predict_organ_dose.py.

Usage:
    python train_ensemble_model.py \
        --labels organ_dose_labels.csv \
        --out saved_model/
"""
import argparse
import pandas as pd
from models.ensemble import OrganDoseEnsemble


def main():
    parser = argparse.ArgumentParser(description="Train RF+XGBoost ensemble for organ dose estimation.")
    parser.add_argument("--labels", required=True,
                        help="Path to organ_dose_labels.csv (from estimate_organ_dose.py)")
    parser.add_argument("--out", default="saved_model",
                        help="Directory to save the trained model artifacts")
    parser.add_argument("--rf-trees", type=int, default=300,
                        help="Number of trees in the Random Forest (default: 300)")
    parser.add_argument("--xgb-trees", type=int, default=400,
                        help="Number of boosting rounds for XGBoost (default: 400)")
    args = parser.parse_args()

    print(f"Loading labels from: {args.labels}")
    df = pd.read_csv(args.labels, dtype={"UID": str})
    print(f"  {len(df)} rows, {df['UID'].nunique()} patients, "
          f"{df['organ'].nunique()} unique organs")
    print(f"  Organs: {sorted(df['organ'].unique())}")

    model = OrganDoseEnsemble(
        n_estimators=args.rf_trees,
        xgb_n_estimators=args.xgb_trees,
    )
    metrics = model.fit(df)
    model.save(args.out)

    print("\n=== Training complete ===")
    print(f"Ensemble OOF  MAE: {metrics['ensemble_mae']:.2f} mGy   R²: {metrics['ensemble_r2']:.3f}")
    print(f"Model saved to: {args.out}/")
    print("\nNext step:")
    print("  python predict_organ_dose.py --dicom <path/to/dicom_folder> --model saved_model/")


if __name__ == "__main__":
    main()
