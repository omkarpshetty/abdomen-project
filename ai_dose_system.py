"""
Complete AI-Based Organ Dose Estimation System

End-to-end pipeline: DICOM → Segmentation → Feature Extraction →
                     Dose Prediction → Database Storage

Usage:
    python ai_dose_system.py --dicom-dir data/patient_001 --output results/
    python ai_dose_system.py --dicom-dir data/patient_001 --save-to-db
    python ai_dose_system.py --dicom-dir data/patient_001 --train-mode
"""
import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import json

# Import our modules
from src.organ_segmenter import OrganSegmenter, load_dicom_volume
from models.ensemble_dose_predictor import EnsembleDosePredictor
from src.patient_dose_calculator import PatientDoseCalculator, create_organ_dose_table
from src.database_connector import DatabaseConnector


class AIDoseEstimationSystem:
    """
    Complete AI system for organ dose estimation from CT DICOM scans.

    Pipeline:
    1. Load DICOM and extract metadata
    2. Segment organs with TotalSegmentator
    3. Extract organ features
    4. Predict doses with ensemble AI model
    5. Calculate patient total dose
    6. Store results in database (optional)
    """

    def __init__(self,
                 model_dir: str = 'trained_models',
                 fast_segmentation: bool = False,
                 device: str = 'gpu',
                 enable_database: bool = False,
                 supabase_url: str = None,
                 supabase_key: str = None):
        """
        Initialize AI dose estimation system.

        Args:
            model_dir: Directory containing trained models
            fast_segmentation: Use fast TotalSegmentator mode
            device: 'gpu', 'cpu', or 'mps'
            enable_database: Connect to Supabase database
            supabase_url: Supabase project URL
            supabase_key: Supabase API key
        """
        print("="*80)
        print("AI-BASED ORGAN DOSE ESTIMATION SYSTEM")
        print("="*80)
        print("Initializing components...\n")

        # Initialize organ segmenter
        print("[1/4] Loading organ segmenter...")
        self.segmenter = OrganSegmenter(fast=fast_segmentation, device=device)
        print("      [OK] TotalSegmentator ready\n")

        # Initialize dose prediction model
        print("[2/4] Loading AI dose prediction model...")
        self.dose_predictor = self._load_or_create_model(model_dir, device)
        print("      [OK] Ensemble model ready\n")

        # Initialize dose calculator
        print("[3/4] Loading patient dose calculator...")
        self.dose_calculator = PatientDoseCalculator()
        print("      [OK] Dose calculator ready\n")

        # Initialize database connector
        print("[4/4] Setting up database connection...")
        if enable_database:
            self.database = DatabaseConnector(url=supabase_url, key=supabase_key)
            if self.database.is_enabled():
                print("      [OK] Database connected\n")
            else:
                print("      [WARNING] Database not available (will save to files only)\n")
        else:
            self.database = None
            print("      [WARNING] Database disabled\n")

        print("="*80)
        print("System ready!\n")

    def _load_or_create_model(self, model_dir: str, device: str):
        """Load existing model or create new one."""
        model_path = Path(model_dir)

        if model_path.exists() and (model_path / 'metadata.json').exists():
            try:
                print(f"      Loading model from {model_dir}...")
                return EnsembleDosePredictor.load(model_dir, device=device)
            except Exception as e:
                print(f"      Warning: Could not load model: {e}")
                print(f"      Creating new model (will need training)...")
                return EnsembleDosePredictor(device=device)
        else:
            print(f"      No trained model found. Creating new model...")
            print(f"      (Model will need training before prediction)")
            return EnsembleDosePredictor(device=device)

    def process_patient(self,
                       dicom_folder: str,
                       output_dir: str = None,
                       save_to_database: bool = False,
                       patient_uid: str = None) -> Dict:
        """
        Process a patient scan end-to-end.

        Args:
            dicom_folder: Path to DICOM files
            output_dir: Output directory for results
            save_to_database: Save results to database
            patient_uid: Unique patient identifier

        Returns:
            Complete results dictionary
        """
        patient_name = Path(dicom_folder).name
        if patient_uid is None:
            patient_uid = patient_name

        if output_dir is None:
            output_dir = f"outputs/{patient_name}_dose_estimation"

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"PROCESSING PATIENT: {patient_name}")
        print(f"{'='*80}\n")

        start_time = time.time()

        results = {
            'patient_uid': patient_uid,
            'patient_name': patient_name,
            'dicom_folder': dicom_folder,
            'output_dir': output_dir,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'processing'
        }

        try:
            # Step 1: Load DICOM metadata
            print("[Step 1/5] Loading DICOM metadata...")
            _, metadata = load_dicom_volume(dicom_folder)
            results['scan_metadata'] = metadata
            print(f"            kVp: {metadata['kvp']:.0f}")
            print(f"            CTDIvol: {metadata['ctdivol_mgy']:.2f} mGy\n")

            # Step 2: Segment organs (returns both masks and CT volume from same NIfTI)
            print("[Step 2/5] Segmenting organs with TotalSegmentator...")
            seg_output = os.path.join(output_dir, 'segmentation')
            organ_masks, ct_volume = self.segmenter.segment_from_dicom(dicom_folder, seg_output)
            print(f"            Detected {len(organ_masks)} organs")
            print(f"            CT Volume shape: {ct_volume.shape}\n")

            # Step 3: Extract organ features
            print("[Step 3/5] Extracting organ features...")
            organ_features = {}
            organs_not_detected = []

            for organ_name, mask in organ_masks.items():
                features = self.segmenter.extract_organ_features(
                    ct_volume, mask, metadata['voxel_spacing_mm'], organ_name
                )
                if features:
                    organ_features[organ_name] = features
                else:
                    organs_not_detected.append(organ_name)

            results['organ_measurements'] = organ_features
            results['organs_not_detected'] = organs_not_detected

            print(f"            Extracted features for {len(organ_features)} organs")
            if organs_not_detected:
                print(f"            {len(organs_not_detected)} organs not detected in scan field\n")
            else:
                print()

            # Step 4: Predict organ doses
            print("[Step 4/5] Predicting organ doses with AI models...")
            organ_doses = self._predict_organ_doses(organ_features, metadata)
            results['organ_doses'] = organ_doses
            print(f"            Predicted doses for {len(organ_doses)} organs\n")

            # Step 5: Calculate patient total dose
            print("[Step 5/5] Calculating patient total dose...")
            patient_dose = self.dose_calculator.calculate_patient_dose(organ_doses)
            results['patient_dose'] = patient_dose
            print(f"            Total effective dose: {patient_dose['total_effective_dose_msv']:.2f} mSv")
            print(f"            Risk category: {patient_dose['risk_category']}\n")

            # Save results
            processing_time = time.time() - start_time
            results['processing_time_seconds'] = processing_time
            results['status'] = 'completed'

            self._save_results(results, output_dir)

            # Save to database if enabled
            if save_to_database and self.database and self.database.is_enabled():
                print("Saving to database...")
                self.database.save_complete_results(patient_uid, results)

            # Display summary
            self._display_summary(results)

            print(f"\n{'='*80}")
            print(f"PROCESSING COMPLETE")
            print(f"{'='*80}")
            print(f"Time: {processing_time:.1f}s")
            print(f"Output: {output_dir}")
            print(f"{'='*80}\n")

            return results

        except Exception as e:
            print(f"\n[ERROR] ERROR: {e}")
            import traceback
            traceback.print_exc()
            results['status'] = 'failed'
            results['error'] = str(e)
            return results

    def _predict_organ_doses(self, organ_features: Dict, scan_metadata: Dict) -> Dict:
        """Predict doses for each organ."""
        if not self.dose_predictor._is_fitted:
            print("            [WARNING] Model not trained. Using fallback estimation...")
            return self._fallback_dose_estimation(organ_features, scan_metadata)

        # Get all organs the model knows
        all_model_organs = self._load_model_organ_names()

        # Separate detected vs undetected organs
        ai_predictable = {}
        for organ_name, features in organ_features.items():
            if organ_name in all_model_organs:
                ai_predictable[organ_name] = features

        # Prepare features for AI prediction
        feature_list = []
        organ_names = []

        for organ_name, features in ai_predictable.items():
            row = {
                'organ': organ_name,
                'volume_cm3': features['volume_cm3'],
                'mean_hu': features['mean_hu'],
                'median_hu': features['median_hu'],
                'std_hu': features['std_hu'],
                'distance_from_isocenter_mm': features['distance_from_isocenter_mm'],
                'compactness': features['compactness'],
                'kvp': scan_metadata['kvp'],
                'ctdivol_mgy': scan_metadata['ctdivol_mgy'],
                'num_slices': scan_metadata['num_slices'],
                'slice_thickness_mm': scan_metadata['slice_thickness_mm'],
            }
            feature_list.append(row)
            organ_names.append(organ_name)

        # Predict doses using AI model for detected organs
        organ_doses = {}

        if feature_list:
            df = pd.DataFrame(feature_list)
            predictions = self.dose_predictor.predict(df)

            for organ_name, dose in zip(organ_names, predictions):
                tissue_weight = self.dose_calculator.TISSUE_WEIGHTS.get(organ_name, 0.04)
                organ_doses[organ_name] = {
                    'dose_mgy': float(dose),
                    'effective_dose_msv': float(dose * tissue_weight),
                    'tissue_weight': tissue_weight,
                    'volume_cm3': ai_predictable[organ_name]['volume_cm3'],
                    'mean_hu': ai_predictable[organ_name]['mean_hu'],
                    'detected': True,
                    'prediction_method': 'AI'
                }

        # Add entries for all model organs that weren't detected
        detected_organs = set(organ_doses.keys())
        undetected_organs = all_model_organs - detected_organs

        if undetected_organs:
            print(f"            {len(undetected_organs)} organs not in scan field (marked as not detected)")
            for organ_name in undetected_organs:
                tissue_weight = self.dose_calculator.TISSUE_WEIGHTS.get(organ_name, 0.04)
                organ_doses[organ_name] = {
                    'dose_mgy': 0.0,
                    'effective_dose_msv': 0.0,
                    'tissue_weight': tissue_weight,
                    'volume_cm3': 0.0,
                    'mean_hu': 0.0,
                    'detected': False,
                    'prediction_method': 'Not Detected'
                }

        return organ_doses

    def _load_model_organ_names(self):
        """Load the set of organ names the trained model knows."""
        import json
        metadata_path = os.path.join('trained_models', 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, encoding='utf-8') as f:
                return set(json.load(f).get('organs', []))
        return set([
            "BONES", "BOWEL", "GALL BLADDER", "HEART", "KIDNEYS", "LIVER", "LUNGS",
            "MUSCLE", "PANCREAS", "PROSTATE", "SPINAL CORD", "SPLEEN", "STOMACH",
            "URINARY BLADDER", "VESSELS"
        ])

    def _fallback_dose_estimation(self, organ_features: Dict, scan_metadata: Dict) -> Dict:
        """Fallback dose estimation when model is not trained."""
        organ_doses = {}
        ctdivol = scan_metadata.get('ctdivol_mgy', 10.0)

        # Simple empirical formula
        for organ_name, features in organ_features.items():
            volume = features['volume_cm3']
            mean_hu = features['mean_hu']

            # Base dose from CTDIvol
            base_dose = ctdivol * 0.8

            # Adjust for organ properties
            volume_factor = 1.0 + (volume / 1000.0) * 0.1
            density_factor = 1.0 + (mean_hu / 100.0) * 0.05

            absorbed_dose = base_dose * volume_factor * density_factor

            tissue_weight = self.dose_calculator.TISSUE_WEIGHTS.get(organ_name, 0.04)

            organ_doses[organ_name] = {
                'dose_mgy': float(absorbed_dose),
                'effective_dose_msv': float(absorbed_dose * tissue_weight),
                'tissue_weight': tissue_weight,
                'volume_cm3': volume,
                'mean_hu': mean_hu
            }

        return organ_doses

    def _save_results(self, results: Dict, output_dir: str):
        """Save results to files."""
        # Save organ doses as CSV
        if 'organ_doses' in results:
            df_data = []
            for organ_name, dose_info in results['organ_doses'].items():
                df_data.append({
                    'organ': organ_name,
                    'volume_cm3': dose_info['volume_cm3'],
                    'mean_hu': dose_info['mean_hu'],
                    'absorbed_dose_mgy': dose_info['dose_mgy'],
                    'effective_dose_msv': dose_info['effective_dose_msv'],
                    'tissue_weight': dose_info['tissue_weight']
                })

            df = pd.DataFrame(df_data)
            df = df.sort_values('absorbed_dose_mgy', ascending=False)
            csv_path = os.path.join(output_dir, 'organ_doses.csv')
            df.to_csv(csv_path, index=False)
            print(f"            Saved: {csv_path}")

        # Save complete results as JSON
        json_path = os.path.join(output_dir, 'complete_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"            Saved: {json_path}")

        # Save summary report
        if 'patient_dose' in results and 'organ_doses' in results:
            report = self.dose_calculator.generate_summary_report(
                results['patient_dose'],
                results['patient_name']
            )
            report += "\n\n"
            report += create_organ_dose_table(
                results['organ_doses'],
                results['patient_dose']
            )

            report_path = os.path.join(output_dir, 'dose_report.txt')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"            Saved: {report_path}")

    def _display_summary(self, results: Dict):
        """Display results summary."""
        if 'patient_dose' not in results:
            return

        print("\n" + "="*80)
        print("DOSE ESTIMATION SUMMARY")
        print("="*80)

        pd_info = results['patient_dose']
        print(f"Total Effective Dose: {pd_info['total_effective_dose_msv']:.2f} mSv")
        print(f"Total Absorbed Dose: {pd_info['total_absorbed_dose_mgy']:.2f} mGy")
        print(f"Risk Category: {pd_info['risk_category']}")
        print(f"Highest Dose Organ: {pd_info['highest_dose_organ']} "
              f"({pd_info['highest_dose_value_mgy']:.2f} mGy)")
        print(f"\nEquivalent to:")
        print(f"  • {pd_info['equivalent_chest_xrays']:.1f} chest X-rays")
        print(f"  • {pd_info['equivalent_background_days']:.1f} days of natural background")

    def train_model(self, training_data_csv: str, output_dir: str = 'trained_models'):
        """
        Train the dose prediction model.

        Args:
            training_data_csv: Path to training data CSV
            output_dir: Where to save trained model
        """
        print("\n" + "="*80)
        print("TRAINING DOSE PREDICTION MODEL")
        print("="*80 + "\n")

        # Load training data
        print(f"Loading training data from {training_data_csv}...")
        df = pd.read_csv(training_data_csv)
        print(f"  Loaded {len(df)} samples\n")

        # Train model
        metrics = self.dose_predictor.train(df, verbose=True)

        # Save model
        print(f"\nSaving model to {output_dir}...")
        self.dose_predictor.save(output_dir)

        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)

        return metrics


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='AI-Based Organ Dose Estimation System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single patient
  python ai_dose_system.py --dicom-dir data/patient_001

  # Process and save to database
  python ai_dose_system.py --dicom-dir data/patient_001 --save-to-db

  # Train model
  python ai_dose_system.py --train --training-data organ_training_data.csv

  # Batch process multiple patients
  python ai_dose_system.py --batch data/patients/ --output results/
        """
    )

    parser.add_argument('--dicom-dir', type=str,
                       help='Path to DICOM folder')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory')
    parser.add_argument('--patient-uid', type=str, default=None,
                       help='Unique patient identifier')

    parser.add_argument('--model-dir', type=str, default='trained_models',
                       help='Directory with trained models')
    parser.add_argument('--fast', action='store_true',
                       help='Use fast segmentation mode')
    parser.add_argument('--device', type=str, default='gpu',
                       choices=['gpu', 'cpu', 'mps'],
                       help='Device for computation')

    parser.add_argument('--save-to-db', action='store_true',
                       help='Save results to Supabase database')
    parser.add_argument('--supabase-url', type=str,
                       help='Supabase project URL')
    parser.add_argument('--supabase-key', type=str,
                       help='Supabase API key')

    parser.add_argument('--train', action='store_true',
                       help='Train the dose prediction model')
    parser.add_argument('--training-data', type=str,
                       help='Path to training data CSV')

    parser.add_argument('--batch', type=str,
                       help='Process all DICOM folders in directory')

    args = parser.parse_args()

    # Initialize system
    system = AIDoseEstimationSystem(
        model_dir=args.model_dir,
        fast_segmentation=args.fast,
        device=args.device,
        enable_database=args.save_to_db,
        supabase_url=args.supabase_url,
        supabase_key=args.supabase_key
    )

    # Training mode
    if args.train:
        if not args.training_data:
            print("Error: --training-data required for training mode")
            sys.exit(1)

        system.train_model(args.training_data, args.model_dir)
        sys.exit(0)

    # Batch processing mode
    if args.batch:
        batch_dir = Path(args.batch)
        patient_folders = [d for d in batch_dir.iterdir() if d.is_dir()]

        print(f"\nBatch processing {len(patient_folders)} patients...\n")

        results_list = []
        for patient_folder in patient_folders:
            try:
                result = system.process_patient(
                    dicom_folder=str(patient_folder),
                    output_dir=args.output,
                    save_to_database=args.save_to_db
                )
                results_list.append(result)
            except Exception as e:
                print(f"Error processing {patient_folder}: {e}")

        print(f"\nBatch processing complete: {len(results_list)} patients processed")
        sys.exit(0)

    # Single patient mode
    if not args.dicom_dir:
        parser.print_help()
        sys.exit(1)

    result = system.process_patient(
        dicom_folder=args.dicom_dir,
        output_dir=args.output,
        save_to_database=args.save_to_db,
        patient_uid=args.patient_uid
    )

    sys.exit(0 if result['status'] == 'completed' else 1)


if __name__ == "__main__":
    main()
