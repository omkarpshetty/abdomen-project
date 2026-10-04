"""
Train Patient-Specific AI Dose Model

Simple script to train the AI model on your organ feature data.

Usage:
    python train_model.py
"""

# Historical implementation retained for audit; use the supported pipeline.
if __name__ == "__main__":
    raise SystemExit("Retired entry point. Use: python -m ct_dose --help")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from models.patient_specific_ai import PatientSpecificAI


def main():
    print("="*70)
    print("TRAINING PATIENT-SPECIFIC AI DOSE MODEL")
    print("="*70)

    # File paths
    organ_features = "organ_features.csv"
    dicom_params = "dicom_params.csv"
    model_output = "trained_model"

    # Check files exist
    if not Path(organ_features).exists():
        print(f"\n[ERROR] {organ_features} not found!")
        print("\nRun this first to extract features:")
        print(f"  python batch_extract_organ_features.py --root <your_data_folder> --out {organ_features} --fast --device cpu")
        sys.exit(1)

    if not Path(dicom_params).exists():
        print(f"\n[ERROR] {dicom_params} not found!")
        print("\nRun this first to extract DICOM parameters:")
        print(f"  python batch_extract_dicom_params.py --root <your_data_folder> --out {dicom_params}")
        sys.exit(1)

    # Initialize model
    print("\nInitializing AI model...")
    model = PatientSpecificAI(device='cpu')

    # Train
    metrics = model.train(
        organ_features_path=organ_features,
        dicom_params_path=dicom_params,
        epochs=150,
        batch_size=32,
        lr=0.001
    )

    # Save model
    model.save(model_output)

    # Summary
    print("\n" + "="*70)
    print("[SUCCESS] TRAINING COMPLETE!")
    print("="*70)
    print(f"\nModel Performance:")
    print(f"  MAE:      {metrics['mae']:.3f}")
    print(f"  RMSE:     {metrics['rmse']:.3f}")
    print(f"  R²:       {metrics['r2']:.3f}")
    print(f"\nDataset:")
    print(f"  Patients: {metrics['n_patients']}")
    print(f"  Samples:  {metrics['n_samples']}")
    print(f"  Organs:   {metrics['n_organs']}")
    print(f"\nModel saved to: {model_output}/")
    print("\nNext: Predict doses for a new patient")
    print(f"  python predict_dose.py --dicom data/patient_138p")


if __name__ == "__main__":
    main()
