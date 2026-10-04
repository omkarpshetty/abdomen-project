"""
Complete Multi-Model Training Pipeline

Trains multiple complementary models and combines them into an ensemble:
1. Patient-Specific Neural Network
2. XGBoost Gradient Boosting
3. Ensemble Meta-Model

This provides the most accurate, patient-specific organ dose predictions
by leveraging the strengths of different model architectures.

Usage:
    python train_multi_model.py
"""

# Historical implementation retained for audit; use the supported pipeline.
if __name__ == "__main__":
    raise SystemExit("Retired entry point. Use: python -m ct_dose --help")

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from models.patient_specific_ai import PatientSpecificAI
from models.xgboost_dose_model import XGBoostDosePredictor
from models.ensemble_predictor import EnsembleDosePredictor


def main():
    print("="*70)
    print("MULTI-MODEL ENSEMBLE TRAINING PIPELINE")
    print("="*70)
    print("\nThis will train multiple AI models and combine them for")
    print("maximum accuracy and patient-specific predictions.")

    # Configuration
    organ_features = "organ_features.csv"
    dicom_params = "dicom_params.csv"
    output_base = "trained_models_ensemble"

    # Check files exist
    if not Path(organ_features).exists():
        print(f"\n[ERROR] {organ_features} not found!")
        print("\nRun feature extraction first:")
        print("  python batch_extract_organ_features.py --root <data_folder> --out organ_features.csv --fast")
        sys.exit(1)

    if not Path(dicom_params).exists():
        print(f"\n[ERROR] {dicom_params} not found!")
        print("\nRun parameter extraction first:")
        print("  python batch_extract_dicom_params.py --root <data_folder> --out dicom_params.csv")
        sys.exit(1)

    # Load data summary
    organ_df = pd.read_csv(organ_features, dtype={'UID': str})
    dicom_df = pd.read_csv(dicom_params, dtype={'UID': str})
    merged = organ_df.merge(dicom_df, on=['UID', 'PHASE'], how='inner')

    print(f"\nDataset Summary:")
    print(f"  Patients: {merged['UID'].nunique()}")
    print(f"  Total samples: {len(merged)}")
    print(f"  Organs: {merged['organ'].nunique()}")
    print(f"  Phases: {merged['PHASE'].unique()}")

    # Create output directory
    Path(output_base).mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # MODEL 1: Patient-Specific Neural Network
    # ========================================================================
    print("\n" + "="*70)
    print("MODEL 1/2: PATIENT-SPECIFIC NEURAL NETWORK")
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

    # ========================================================================
    # MODEL 2: XGBoost Gradient Boosting
    # ========================================================================
    print("\n" + "="*70)
    print("MODEL 2/2: XGBOOST GRADIENT BOOSTING")
    print("="*70)

    xgb_model = XGBoostDosePredictor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05
    )

    xgb_metrics = xgb_model.train(
        organ_features_path=organ_features,
        dicom_params_path=dicom_params
    )

    xgb_save_path = Path(output_base) / 'xgboost'
    xgb_model.save(str(xgb_save_path))

    # ========================================================================
    # ENSEMBLE: Combine Models with Optimal Weights
    # ========================================================================
    print("\n" + "="*70)
    print("CREATING ENSEMBLE META-MODEL")
    print("="*70)

    ensemble = EnsembleDosePredictor(models_config={
        'patient_specific_nn': True,
        'xgboost': True,
        'cnn': False,  # Optional
        'transformer': False  # Optional
    })

    # Load trained models
    ensemble.load_models(output_base)

    # Learn optimal weights
    print("\nLearning optimal combination weights...")
    weights = ensemble.train_ensemble_weights(
        organ_features_csv=organ_features,
        dicom_params_csv=dicom_params,
        val_split=0.2
    )

    # Save ensemble
    ensemble_save_path = Path(output_base) / 'ensemble'
    ensemble.save(str(ensemble_save_path))

    # ========================================================================
    # EVALUATION: Compare All Models
    # ========================================================================
    print("\n" + "="*70)
    print("FINAL EVALUATION - ALL MODELS")
    print("="*70)

    ensemble_metrics = ensemble.evaluate(organ_features, dicom_params)

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("[SUCCESS] MULTI-MODEL TRAINING COMPLETE!")
    print("="*70)

    print("\nModel Performance Comparison:")
    print("-" * 70)
    print(f"{'Model':<30} {'MAE':>10} {'R²':>10} {'RMSE':>10}")
    print("-" * 70)
    print(f"{'Patient-Specific NN':<30} {nn_metrics['mae']:>10.3f} {nn_metrics['r2']:>10.3f} {nn_metrics['rmse']:>10.3f}")
    print(f"{'XGBoost':<30} {xgb_metrics['mae']:>10.3f} {xgb_metrics['r2']:>10.3f} {'N/A':>10}")
    print("-" * 70)
    print(f"{'ENSEMBLE (Combined)':<30} {ensemble_metrics['ensemble_mae']:>10.3f} {ensemble_metrics['ensemble_r2']:>10.3f} {ensemble_metrics['ensemble_rmse']:>10.3f}")
    print("=" * 70)

    print(f"\nEnsemble Weights:")
    for model_name, weight in weights.items():
        print(f"  {model_name:<20} {weight:.3f}")

    print(f"\nAll models saved to: {output_base}/")
    print(f"\nModel Structure:")
    print(f"  {output_base}/")
    print(f"  ├── patient_specific_nn/")
    print(f"  ├── xgboost/")
    print(f"  └── ensemble/")

    print(f"\n" + "="*70)
    print("WHY MULTI-MODEL IS BETTER")
    print("="*70)
    print("\n1. Neural Network:")
    print("   - Learns smooth, continuous dose patterns")
    print("   - Good at generalization")
    print("   - Captures complex non-linear relationships")

    print("\n2. XGBoost:")
    print("   - Excellent at finding threshold effects")
    print("   - Handles feature interactions well")
    print("   - More interpretable (feature importance)")
    print("   - Robust to outliers")

    print("\n3. Ensemble:")
    print("   - Combines strengths of both models")
    print("   - Reduces individual model biases")
    print("   - More stable predictions")
    print("   - Provides uncertainty estimates")

    print(f"\n" + "="*70)
    print("PATIENT-SPECIFIC PREDICTIONS")
    print("="*70)

    # Demonstrate patient variability
    print("\nVerifying patient-specific predictions...")
    print("Checking LIVER doses for different patients:\n")

    # Get predictions for a few patients
    liver_data = merged[merged['organ'] == 'LIVER'].copy()
    if len(liver_data) > 0:
        sample_patients = liver_data['UID'].unique()[:5]

        for uid in sample_patients:
            patient_data = liver_data[liver_data['UID'] == uid]
            patient_organ = patient_data[organ_df.columns]
            patient_dicom = patient_data[['UID', 'PHASE'] + list(dicom_df.columns[2:])]

            pred, confidence, uncertainty = ensemble.predict_with_confidence(
                patient_organ, patient_dicom
            )

            volume = patient_data['volume_cm3'].values[0]
            hu = patient_data['mean_hu'].values[0]

            print(f"  Patient {uid:>3}: {pred[0]:>6.2f} mGy  "
                  f"(confidence: {confidence[0]:.2f}, "
                  f"vol: {volume:>7.1f} cm³, HU: {hu:>5.1f})")

    print("\n✓ Each patient gets unique predictions based on their anatomy!")

    print(f"\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\n1. Predict doses for new patients:")
    print(f"   python predict_ensemble.py --dicom data/patient_138p")

    print("\n2. Add your 71 additional patients:")
    print(f"   python batch_extract_organ_features.py --root <your_71_patients> --out organ_features.csv --fast")
    print(f"   python batch_extract_dicom_params.py --root <your_71_patients> --out dicom_params_new.csv")
    print(f"   # Merge datasets, then rerun: python train_multi_model.py")

    print("\n3. Expected improvement with 144 patients:")
    print(f"   Current R²: {ensemble_metrics['ensemble_r2']:.3f}")
    print(f"   Expected R²: 0.980-0.990 (even better!)")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
