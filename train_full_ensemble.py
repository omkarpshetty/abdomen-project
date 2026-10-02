"""
train_full_ensemble.py

Trains the complete 4-model comparison: RF + XGBoost + LightGBM + SVR,
then stacks the best performers into a final ensemble.

This is your main training script for the project report's model comparison table.

Usage:
    python train_full_ensemble.py --labels organ_dose_labels.csv --out saved_model_full/
"""
import argparse
import pandas as pd
from models.ensemble_full import FullEnsemble


def main():
    parser = argparse.ArgumentParser(description="Train full 4-model ensemble")
    parser.add_argument("--labels", required=True,
                        help="Path to organ_dose_labels.csv")
    parser.add_argument("--out", default="saved_model_full",
                        help="Directory to save the trained models")
    args = parser.parse_args()

    print(f"Loading labels from: {args.labels}")
    df = pd.read_csv(args.labels, dtype={"UID": str})
    print(f"  {len(df)} rows, {df['UID'].nunique()} patients, "
          f"{df['organ'].nunique()} unique organs\n")

    model = FullEnsemble()
    metrics = model.fit(df)
    model.save(args.out)

    print("\n" + "="*70)
    print("TRAINING COMPLETE - Model Comparison Summary:")
    print("="*70)
    print(f"  Random Forest    MAE: {metrics['rf_mae']:6.2f} mGy   R²: {metrics['rf_r2']:.3f}")
    if metrics['xgb_mae']:
        print(f"  XGBoost          MAE: {metrics['xgb_mae']:6.2f} mGy   R²: {metrics['xgb_r2']:.3f}")
    if metrics['lgb_mae']:
        print(f"  LightGBM         MAE: {metrics['lgb_mae']:6.2f} mGy   R²: {metrics['lgb_r2']:.3f}")
    print(f"  SVR              MAE: {metrics['svr_mae']:6.2f} mGy   R²: {metrics['svr_r2']:.3f}")
    print(f"  ENSEMBLE         MAE: {metrics['ensemble_mae']:6.2f} mGy   R²: {metrics['ensemble_r2']:.3f}")
    print("="*70)
    print(f"\nModel saved to: {args.out}/")
    print("\nNext step:")
    print("  python predict_organ_dose.py --dicom <path/to/dicom_folder> --model saved_model_full/")


if __name__ == "__main__":
    main()
