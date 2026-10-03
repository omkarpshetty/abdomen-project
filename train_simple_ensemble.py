"""
Simplified Multi-Model Training

Trains Neural Network and XGBoost, then uses simple averaging
for ensemble predictions.

Usage:
    python train_simple_ensemble.py
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from models.patient_specific_ai import PatientSpecificAI
from models.xgboost_dose_model import XGBoostDosePredictor
from sklearn.metrics import mean_absolute_error, r2_score


def main():
    print("="*70)
    print("MULTI-MODEL ENSEMBLE TRAINING (SIMPLIFIED)")
    print("="*70)

    organ_features = "organ_features.csv"
    dicom_params = "dicom_params.csv"
    output_base = "trained_models_ensemble"

    # Check files
    if not Path(organ_features).exists() or not Path(dicom_params).exists():
        print("\n[ERROR] Data files not found!")
        sys.exit(1)

    # Load data summary
    organ_df = pd.read_csv(organ_features, dtype={'UID': str})
    dicom_df = pd.read_csv(dicom_params, dtype={'UID': str})
    merged = organ_df.merge(dicom_df, on=['UID', 'PHASE'], how='inner')

    print(f"\nDataset: {merged['UID'].nunique()} patients, {len(merged)} samples")

    Path(output_base).mkdir(parents=True, exist_ok=True)

    # Train Neural Network
    print("\n" + "="*70)
    print("MODEL 1/2: NEURAL NETWORK")
    print("="*70)

    nn_model = PatientSpecificAI(device='cpu')
    nn_metrics = nn_model.train(
        organ_features_path=organ_features,
        dicom_params_path=dicom_params,
        epochs=150,
        batch_size=32,
        lr=0.001
    )

    nn_save_path = Path(output_base) / 'patient_specific_nn'
    nn_model.save(str(nn_save_path))

    # Train XGBoost
    print("\n" + "="*70)
    print("MODEL 2/2: XGBOOST")
    print("="*70)

    xgb_model = XGBoostDosePredictor(n_estimators=500, max_depth=8, learning_rate=0.05)
    xgb_metrics = xgb_model.train(
        organ_features_path=organ_features,
        dicom_params_path=dicom_params
    )

    xgb_save_path = Path(output_base) / 'xgboost'
    xgb_model.save(str(xgb_save_path))

    # Evaluate Ensemble (Simple Average)
    print("\n" + "="*70)
    print("ENSEMBLE EVALUATION (Equal Weighting)")
    print("="*70)

    # Get predictions from both models
    nn_preds = nn_model.predict(organ_df, dicom_df)
    xgb_preds = xgb_model.predict(organ_df, dicom_df)

    # Simple average ensemble
    ensemble_preds = (nn_preds + xgb_preds) / 2.0

    # Generate targets for evaluation
    targets = nn_model._create_synthetic_labels(merged)

    # Calculate metrics
    ensemble_mae = mean_absolute_error(targets, ensemble_preds)
    ensemble_r2 = r2_score(targets, ensemble_preds)
    ensemble_rmse = np.sqrt(np.mean((targets - ensemble_preds) ** 2))

    # Summary
    print("\n" + "="*70)
    print("[SUCCESS] TRAINING COMPLETE!")
    print("="*70)

    print("\nModel Performance:")
    print("-" * 70)
    print(f"{'Model':<30} {'MAE':>10} {'R²':>10}")
    print("-" * 70)
    print(f"{'Neural Network':<30} {nn_metrics['mae']:>10.3f} {nn_metrics['r2']:>10.3f}")
    print(f"{'XGBoost':<30} {xgb_metrics['mae']:>10.3f} {xgb_metrics['r2']:>10.3f}")
    print("-" * 70)
    print(f"{'ENSEMBLE (Average)':<30} {ensemble_mae:>10.3f} {ensemble_r2:>10.3f}")
    print("=" * 70)

    print(f"\nModels saved to: {output_base}/")
    print("\nNext: Predict with ensemble")
    print(f"  python predict_simple_ensemble.py --dicom data/patient_138p")


if __name__ == "__main__":
    main()
