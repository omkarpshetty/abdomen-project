"""
COMPLETE ENSEMBLE MODEL TRAINING PIPELINE
==========================================

Trains all dose prediction models:
1. Random Forest (robust baseline)
2. XGBoost (gradient boosting)
3. Patient-Specific Neural Network
4. Ensemble meta-learner

Requirements:
- organ_features.csv (from processed patients)
- dicom_params.csv (scan parameters)

Outputs:
- trained_models/rf_xgb_ensemble/ (Random Forest + XGBoost)
- trained_models/patient_specific_nn/ (Neural network)
- trained_models/ensemble/ (Meta-learner)
"""

# Historical implementation retained for audit; use the supported pipeline.
if __name__ == "__main__":
    raise SystemExit("Retired entry point. Use: python -m ct_dose --help")


import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import model classes
from models.ensemble import OrganDoseEnsemble
from models.patient_specific_ai import PatientSpecificAI
from models.xgboost_dose_model import XGBoostDosePredictor


def check_data_availability():
    """Check if training data exists."""
    required_files = [
        'data/organ_features.csv',
        'data/dicom_params.csv'
    ]

    missing = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)

    if missing:
        print("ERROR: Missing required data files:")
        for f in missing:
            print(f"  - {f}")
        print("\nPlease run data extraction first to generate these files.")
        return False

    return True


def load_training_data():
    """Load and validate training data."""
    print("Loading training data...")

    organ_df = pd.read_csv('data/organ_features.csv', dtype={'UID': str})
    dicom_df = pd.read_csv('data/dicom_params.csv', dtype={'UID': str})

    print(f"  Organ features: {len(organ_df)} rows from {organ_df['UID'].nunique()} patients")
    print(f"  DICOM params: {len(dicom_df)} rows from {dicom_df['UID'].nunique()} patients")

    # Merge
    df = organ_df.merge(dicom_df, on=['UID', 'PHASE'], how='inner')
    df = df.dropna(subset=['volume_cm3', 'mean_hu', 'mean_kvp'])

    print(f"  Merged dataset: {len(df)} samples from {df['UID'].nunique()} patients")

    return organ_df, dicom_df, df


def train_rf_xgb_ensemble():
    """Train Random Forest + XGBoost ensemble."""
    print("\n" + "="*80)
    print("TRAINING RF + XGBOOST ENSEMBLE")
    print("="*80)

    organ_df, dicom_df, merged_df = load_training_data()

    # Generate pseudo-labels using physics-based model
    print("\nGenerating training labels...")
    from models.patient_specific_ai import PatientSpecificAI
    temp_model = PatientSpecificAI()
    labels = temp_model._create_synthetic_labels(merged_df)
    merged_df['pseudo_label_dose_mGy'] = labels

    # Train ensemble
    print("\nTraining ensemble model...")
    ensemble = OrganDoseEnsemble(
        n_estimators=300,
        xgb_n_estimators=400,
        random_state=42
    )

    cv_metrics = ensemble.fit(merged_df)

    # Save model
    save_dir = 'trained_models/rf_xgb_ensemble'
    ensemble.save(save_dir)

    print(f"\n[SUCCESS] RF + XGBoost ensemble saved to: {save_dir}/")
    return cv_metrics


def train_patient_specific_nn():
    """Train patient-specific neural network."""
    print("\n" + "="*80)
    print("TRAINING PATIENT-SPECIFIC NEURAL NETWORK")
    print("="*80)

    # Check if PyTorch is available
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
    except Exception as e:
        print(f"[WARN] PyTorch issue: {e}")
        print("Skipping neural network training...")
        return None

    organ_df, dicom_df, _ = load_training_data()

    # Train model
    print("\nTraining neural network...")
    nn_model = PatientSpecificAI(
        hidden_layers=[128, 64, 32],
        dropout=0.3,
        learning_rate=0.001,
        batch_size=32,
        epochs=100
    )

    try:
        metrics = nn_model.train(
            organ_features_path='data/organ_features.csv',
            dicom_params_path='data/dicom_params.csv'
        )

        # Save model
        save_dir = 'trained_models/patient_specific_nn'
        nn_model.save(save_dir)

        print(f"\n[SUCCESS] Neural network saved to: {save_dir}/")
        return metrics
    except Exception as e:
        print(f"\n[ERROR] Neural network training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def train_xgboost_standalone():
    """Train standalone XGBoost model."""
    print("\n" + "="*80)
    print("TRAINING STANDALONE XGBOOST MODEL")
    print("="*80)

    try:
        import xgboost
        print(f"XGBoost version: {xgboost.__version__}")
    except ImportError:
        print("[ERROR] XGBoost not installed. Skipping...")
        return None

    # Train model
    print("\nTraining XGBoost with advanced features...")
    xgb_model = XGBoostDosePredictor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05
    )

    try:
        metrics = xgb_model.train(
            organ_features_path='data/organ_features.csv',
            dicom_params_path='data/dicom_params.csv'
        )

        # Save model
        save_dir = 'trained_models/xgboost_standalone'
        xgb_model.save(save_dir)

        print(f"\n[SUCCESS] XGBoost saved to: {save_dir}/")
        return metrics
    except Exception as e:
        print(f"\n[ERROR] XGBoost training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def train_all_models():
    """Train all available models."""
    print("\n" + "="*80)
    print("COMPLETE MODEL TRAINING PIPELINE")
    print("="*80)
    print("This will train all dose prediction models:")
    print("  1. Random Forest + XGBoost Ensemble")
    print("  2. Patient-Specific Neural Network")
    print("  3. Standalone XGBoost with Feature Engineering")
    print("="*80)

    # Check data
    if not check_data_availability():
        print("\n[ERROR] Cannot proceed without training data.")
        print("\nTo generate training data:")
        print("  1. Process multiple patients with the system")
        print("  2. Extract organ features and DICOM parameters")
        print("  3. Run this training script again")
        return

    results = {}

    # Train RF + XGBoost ensemble
    try:
        results['rf_xgb'] = train_rf_xgb_ensemble()
    except Exception as e:
        print(f"\n[ERROR] RF + XGBoost training failed: {e}")
        results['rf_xgb'] = None

    # Train patient-specific NN
    try:
        results['neural_net'] = train_patient_specific_nn()
    except Exception as e:
        print(f"\n[ERROR] Neural network training failed: {e}")
        results['neural_net'] = None

    # Train standalone XGBoost
    try:
        results['xgboost'] = train_xgboost_standalone()
    except Exception as e:
        print(f"\n[ERROR] XGBoost training failed: {e}")
        results['xgboost'] = None

    # Summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)

    for model_name, metrics in results.items():
        if metrics:
            print(f"\n{model_name.upper()}:")
            if isinstance(metrics, dict):
                for key, value in metrics.items():
                    if value is not None:
                        print(f"  {key}: {value:.3f}")
        else:
            print(f"\n{model_name.upper()}: FAILED")

    print("\n" + "="*80)
    print("All available models have been trained!")
    print("="*80)

    # Save training summary
    summary_file = 'trained_models/training_summary.json'
    Path('trained_models').mkdir(parents=True, exist_ok=True)

    import json
    with open(summary_file, 'w') as f:
        # Convert numpy types for JSON
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif obj is None:
                return "N/A"
            return obj

        summary = {k: {kk: convert(vv) for kk, vv in v.items()} if v else "FAILED"
                  for k, v in results.items()}
        json.dump(summary, f, indent=2)

    print(f"\nTraining summary saved to: {summary_file}")


def quick_validation():
    """Quick validation of trained models."""
    print("\n" + "="*80)
    print("MODEL VALIDATION")
    print("="*80)

    # Check which models exist
    model_dirs = {
        'RF + XGBoost': 'trained_models/rf_xgb_ensemble',
        'Neural Network': 'trained_models/patient_specific_nn',
        'XGBoost': 'trained_models/xgboost_standalone'
    }

    available = []
    for name, path in model_dirs.items():
        if Path(path).exists():
            print(f"  [OK] {name}: {path}")
            available.append(name)
        else:
            print(f"  [MISSING] {name}")

    if available:
        print(f"\n{len(available)}/{len(model_dirs)} models trained successfully!")
    else:
        print("\n[WARN] No trained models found!")

    print("="*80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train organ dose prediction models"
    )
    parser.add_argument(
        '--model',
        choices=['all', 'rf_xgb', 'neural_net', 'xgboost'],
        default='all',
        help='Which model(s) to train'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Only validate existing models'
    )

    args = parser.parse_args()

    if args.validate:
        quick_validation()
    else:
        if args.model == 'all':
            train_all_models()
        elif args.model == 'rf_xgb':
            train_rf_xgb_ensemble()
        elif args.model == 'neural_net':
            train_patient_specific_nn()
        elif args.model == 'xgboost':
            train_xgboost_standalone()

        # Validate after training
        quick_validation()
