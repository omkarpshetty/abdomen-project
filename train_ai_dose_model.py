"""
train_ai_dose_model.py

Trains the complete AI-driven dose estimation system:
  1. Extracts CNN features from CT images (optional but recommended)
  2. Trains neural network to predict organ doses
  3. Saves the complete model for inference

This is your MAIN AI training script — no hardcoded ratios,
pure learned patterns from medical images and dose data.

Usage:
    # Without CNN features (faster, slightly less accurate):
    python train_ai_dose_model.py --labels organ_dose_labels.csv --out ai_model/

    # With CNN features (slower, more accurate):
    python train_ai_dose_model.py --labels organ_dose_labels.csv --extract-cnn --out ai_model/
"""
import argparse
import pandas as pd
import numpy as np
import os
import pickle
from models.ai_dose_predictor import AIDosePredictor

# Optional: CNN feature extraction
try:
    from models.cnn_feature_extractor import CTFeatureExtractor, extract_features_from_dicom
    import torch
    CNN_AVAILABLE = True
except ImportError:
    CNN_AVAILABLE = False


def extract_cnn_embeddings(organ_features_path, dicom_root, device='cpu'):
    """
    Extract CNN features for all patients in the dataset.

    Args:
        organ_features_path: Path to organ_features.csv
        dicom_root: Root directory containing patient DICOM folders
        device: 'cpu' or 'cuda'

    Returns:
        embeddings: dict mapping (UID, PHASE) -> 128-dim numpy array
    """
    if not CNN_AVAILABLE:
        raise RuntimeError("PyTorch not available. Install with: pip install torch torchvision")

    df = pd.read_csv(organ_features_path, dtype={"UID": str})
    patients = df[['UID', 'PHASE']].drop_duplicates()

    print(f"\nExtracting CNN features from {len(patients)} patient scans...")
    print("This may take 10-30 minutes depending on your hardware.\n")

    # Initialize CNN
    cnn_model = CTFeatureExtractor(embedding_dim=128, pretrained=True)
    cnn_model.eval()
    cnn_model.to(device)

    embeddings = {}
    import re

    for i, (_, row) in enumerate(patients.iterrows(), 1):
        uid = row['UID']
        phase = row['PHASE']

        # Find matching DICOM folder
        # Pattern: "<uid> plain" or "<uid>_venous" or "<uid> v" etc.
        pattern = re.compile(rf"^\s*{uid}\s*[-_\s]*(plain|p|venous|v)", re.IGNORECASE)

        dicom_folder = None
        for folder in os.listdir(dicom_root):
            if pattern.match(folder):
                folder_phase = 'plain' if folder.lower().split()[-1].startswith('p') else 'venous'
                if folder_phase == phase:
                    dicom_folder = os.path.join(dicom_root, folder)
                    break

        if dicom_folder and os.path.isdir(dicom_folder):
            try:
                emb = extract_features_from_dicom(dicom_folder, cnn_model, device=device)
                embeddings[(uid, phase)] = emb
                print(f"[{i}/{len(patients)}] UID {uid} {phase}: ✓ extracted")
            except Exception as e:
                print(f"[{i}/{len(patients)}] UID {uid} {phase}: ✗ failed ({e})")
                embeddings[(uid, phase)] = np.zeros(128)
        else:
            print(f"[{i}/{len(patients)}] UID {uid} {phase}: ✗ DICOM folder not found")
            embeddings[(uid, phase)] = np.zeros(128)

    return embeddings


def main():
    parser = argparse.ArgumentParser(description="Train AI dose estimation system")
    parser.add_argument("--labels", required=True,
                        help="Path to organ_dose_labels.csv")
    parser.add_argument("--organ-features", default="organ_features.csv",
                        help="Path to organ_features.csv (needed if --extract-cnn)")
    parser.add_argument("--dicom-root", default=None,
                        help="Root directory with patient DICOM folders (needed if --extract-cnn)")
    parser.add_argument("--extract-cnn", action="store_true",
                        help="Extract CNN features from CT images (requires PyTorch, much slower)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="Device for CNN extraction (default: cpu)")
    parser.add_argument("--out", default="ai_model",
                        help="Output directory for trained model")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs (default: 50)")
    args = parser.parse_args()

    print("="*70)
    print("AI-DRIVEN DOSE ESTIMATION SYSTEM - TRAINING")
    print("="*70)

    # Load data
    print(f"\nLoading labels from: {args.labels}")
    df = pd.read_csv(args.labels, dtype={"UID": str})
    print(f"  {len(df)} rows, {df['UID'].nunique()} patients, {df['organ'].nunique()} organs")

    # Extract CNN features if requested
    cnn_embeddings = None
    if args.extract_cnn:
        if not CNN_AVAILABLE:
            print("\n⚠ WARNING: PyTorch not installed. Skipping CNN extraction.")
            print("  Install with: pip install torch torchvision")
        elif not args.dicom_root:
            print("\n⚠ WARNING: --dicom-root not specified. Skipping CNN extraction.")
        else:
            cnn_embeddings = extract_cnn_embeddings(
                args.organ_features,
                args.dicom_root,
                device=args.device
            )
            # Save embeddings for reuse
            emb_path = os.path.join(args.out, "cnn_embeddings.pkl")
            os.makedirs(args.out, exist_ok=True)
            pickle.dump(cnn_embeddings, open(emb_path, "wb"))
            print(f"\n✓ CNN embeddings saved to {emb_path}")

    # Train AI model
    print("\n" + "="*70)
    print("Training neural network...")
    print("="*70)

    device = args.device if args.extract_cnn and CNN_AVAILABLE else 'cpu'
    model = AIDosePredictor(cnn_dim=128, device=device)

    metrics = model.fit(df, cnn_embeddings=cnn_embeddings, epochs=args.epochs)

    # Save model
    model.save(args.out)

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"  MAE: {metrics['mae']:.2f} mGy")
    print(f"  R²:  {metrics['r2']:.3f}")
    print(f"\nModel saved to: {args.out}/")
    print("\nNext step:")
    print(f"  python predict_patient_dose.py --dicom <folder> --model {args.out}/")


if __name__ == "__main__":
    main()
