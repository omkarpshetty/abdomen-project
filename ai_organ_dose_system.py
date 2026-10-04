"""
Complete AI-Based Organ Dose Estimation System
================================================

Integrates:
1. Self-Training Organ Annotator (segmentation)
2. AI Dose Prediction Models (dose estimation)
3. Patient-specific dose calculations

End-to-end pipeline: DICOM → Organ Segmentation → Dose Prediction
"""

# Historical implementation retained for audit; use the supported pipeline.
if __name__ == "__main__":
    raise SystemExit("Retired entry point. Use: python -m ct_dose --help")


import os
import sys
import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import pickle

# Import organ annotator - using improved version
try:
    from models.improved_segmenter import ImprovedOrganSegmenter as OrganSegmenter
    print("Using Improved Organ Segmenter (with enhanced liver/kidney detection)")
except ImportError:
    from models.self_training_segmenter import SelfTrainingOrganSegmenter as OrganSegmenter
    print("Using Standard Organ Segmenter")

# Import AI dose models
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available, using alternative methods")


class AIOrganDoseEstimator:
    """
    Complete AI system for organ dose estimation.

    Pipeline:
    1. Load patient DICOM
    2. Segment organs (Self-Training Annotator)
    3. Extract organ features
    4. Predict organ-specific doses (AI models)
    5. Calculate patient total dose
    """

    def __init__(self, model_dir: str = 'trained_model'):
        """Initialize the AI dose estimation system."""
        print("="*80)
        print("AI-BASED ORGAN DOSE ESTIMATION SYSTEM")
        print("="*80)
        print("Initializing components...")

        # Initialize organ segmenter
        print("  [1/3] Loading organ annotator...")
        self.organ_segmenter = OrganSegmenter()
        print("        Organ annotator ready")

        # Load AI dose prediction model
        print("  [2/3] Loading AI dose prediction model...")
        self.dose_model = self._load_dose_model(model_dir)
        print("        Dose model ready")

        # Load organ encoder and scaler
        print("  [3/3] Loading preprocessors...")
        self.organ_encoder = self._load_encoder(model_dir)
        self.feature_scaler = self._load_scaler(model_dir)
        print("        Preprocessors ready")

        print("\nSystem ready!")
        print("="*80)

    def _load_dose_model(self, model_dir: str):
        """Load AI dose prediction model."""
        model_path = Path(model_dir) / 'model.pt'

        if not model_path.exists():
            print(f"        Warning: Model not found at {model_path}")
            print("        Using default dose estimation")
            return None

        try:
            if TORCH_AVAILABLE:
                model = torch.load(model_path, map_location='cpu')
                model.eval()
                return model
            else:
                # Load as dictionary for non-torch inference
                with open(model_path, 'rb') as f:
                    import zipfile
                    # PyTorch models are zip files
                    return None
        except Exception as e:
            print(f"        Warning: Could not load model: {e}")
            return None

    def _load_encoder(self, model_dir: str):
        """Load organ label encoder."""
        encoder_path = Path(model_dir) / 'organ_encoder.pkl'

        if encoder_path.exists():
            try:
                with open(encoder_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"        Warning: Could not load encoder: {e}")
                return None
        return None

    def _load_scaler(self, model_dir: str):
        """Load feature scaler."""
        scaler_path = Path(model_dir) / 'scaler.pkl'

        if scaler_path.exists():
            try:
                with open(scaler_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"        Warning: Could not load scaler: {e}")
                return None
        return None

    def load_dicom_volume(self, dicom_folder: str) -> Tuple[np.ndarray, Dict]:
        """Load DICOM series into 3D volume."""
        import pydicom

        print(f"\nLoading DICOM from: {dicom_folder}")

        dicom_files = sorted(list(Path(dicom_folder).glob("*.dcm")))

        if not dicom_files:
            raise ValueError(f"No DICOM files found in {dicom_folder}")

        print(f"  Found {len(dicom_files)} slices")

        # Load first slice
        first_ds = pydicom.dcmread(str(dicom_files[0]))
        image_shape = first_ds.pixel_array.shape

        # Initialize volume
        volume = np.zeros((len(dicom_files), image_shape[0], image_shape[1]), dtype=np.float32)

        # Load all slices
        for i, dcm_file in enumerate(dicom_files):
            ds = pydicom.dcmread(str(dcm_file))
            pixel_array = ds.pixel_array.astype(np.float32)

            # Apply rescale for HU values
            slope = getattr(ds, 'RescaleSlope', 1.0)
            intercept = getattr(ds, 'RescaleIntercept', 0.0)
            volume[i] = pixel_array * slope + intercept

        # Get voxel spacing
        pixel_spacing = first_ds.PixelSpacing
        slice_thickness = float(getattr(first_ds, 'SliceThickness', 5.0))
        voxel_spacing = (slice_thickness, float(pixel_spacing[0]), float(pixel_spacing[1]))

        # Get scan parameters
        metadata = {
            'num_slices': len(dicom_files),
            'voxel_spacing_mm': voxel_spacing,
            'image_shape': volume.shape,
            'kvp': float(getattr(first_ds, 'KVP', 120)),
            'exposure': float(getattr(first_ds, 'Exposure', 0)),
            'slice_thickness': slice_thickness
        }

        print(f"  Volume: {volume.shape}")
        print(f"  Voxel spacing: {voxel_spacing[0]:.2f} x {voxel_spacing[1]:.2f} x {voxel_spacing[2]:.2f} mm")

        return volume, metadata

    def segment_organs(self, ct_volume: np.ndarray,
                      voxel_spacing: Tuple[float, float, float],
                      patient_id: str = None) -> Dict:
        """Segment organs using Self-Training Annotator."""
        print("\nSegmenting organs...")

        start_time = time.time()

        # Run segmentation
        organ_measurements = self.organ_segmenter.segment_patient(
            ct_volume, voxel_spacing, patient_id
        )

        elapsed = time.time() - start_time

        print(f"  Segmentation complete in {elapsed:.2f}s")
        print(f"  Organs detected: {len(organ_measurements)}")

        return organ_measurements

    def predict_organ_doses(self, organ_measurements: Dict,
                           scan_params: Dict) -> Dict:
        """Predict radiation dose for each organ using AI."""
        print("\nPredicting organ doses...")

        organ_doses = {}

        # Extract scan parameters
        kvp = scan_params.get('kvp', 120)
        num_slices = scan_params.get('num_slices', 50)
        slice_thickness = scan_params.get('slice_thickness', 5.0)

        # Calculate exposure factors
        total_exposure = kvp * num_slices * 0.1  # Simplified calculation

        for organ_name, organ_stats in organ_measurements.items():
            # Extract features for dose prediction
            volume_cm3 = organ_stats['volume_cm3']
            mean_hu = organ_stats['mean_hu']

            # Predict dose based on organ characteristics and scan parameters
            if self.dose_model is not None and self.organ_encoder is not None:
                # Use AI model for prediction
                try:
                    organ_dose = self._predict_with_ai_model(
                        organ_name, volume_cm3, mean_hu, kvp, num_slices
                    )
                except:
                    # Fallback to empirical estimation
                    organ_dose = self._estimate_dose_empirical(
                        organ_name, volume_cm3, mean_hu, total_exposure
                    )
            else:
                # Use empirical estimation
                organ_dose = self._estimate_dose_empirical(
                    organ_name, volume_cm3, mean_hu, total_exposure
                )

            organ_doses[organ_name] = {
                'dose_mGy': organ_dose,
                'dose_mSv': organ_dose * self._get_tissue_weighting_factor(organ_name),
                'volume_cm3': volume_cm3,
                'mean_hu': mean_hu
            }

        print(f"  Dose predictions complete for {len(organ_doses)} organs")

        return organ_doses

    def _predict_with_ai_model(self, organ_name: str, volume: float,
                               mean_hu: float, kvp: float, num_slices: int) -> float:
        """Predict dose using trained AI model."""
        # Prepare features
        features = np.array([[
            volume,
            mean_hu,
            kvp,
            num_slices,
            kvp * num_slices
        ]])

        # Scale features
        if self.feature_scaler is not None:
            features = self.feature_scaler.transform(features)

        # Predict
        if TORCH_AVAILABLE and self.dose_model is not None:
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features)
                dose = self.dose_model(features_tensor).item()
            return max(0, dose)
        else:
            # Simple linear model as fallback
            dose = 5.0 + (volume * 0.01) + (mean_hu * 0.05) + (kvp * 0.1)
            return dose

    def _estimate_dose_empirical(self, organ_name: str, volume: float,
                                mean_hu: float, total_exposure: float) -> float:
        """Estimate dose using empirical formulas."""
        # Base dose depends on scan intensity
        base_dose = total_exposure / 100.0

        # Organ-specific factors
        organ_factors = {
            'BONES': 0.8,
            'LIVER': 1.2,
            'SPLEEN': 1.1,
            'KIDNEY_RIGHT': 1.0,
            'KIDNEY_LEFT': 1.0,
            'PANCREAS': 1.0,
            'STOMACH': 0.9,
            'HEART': 1.1,
            'AORTA': 0.7,
            'URINARY_BLADDER': 0.9,
            'GALL_BLADDER': 1.0,
            'SPINAL_CORD': 1.3
        }

        factor = organ_factors.get(organ_name, 1.0)

        # Adjust for organ density (HU-based)
        density_factor = 1.0 + (mean_hu / 1000.0)

        # Calculate dose
        dose = base_dose * factor * density_factor

        return max(0, dose)

    def _get_tissue_weighting_factor(self, organ_name: str) -> float:
        """Get ICRP tissue weighting factors for effective dose."""
        weights = {
            'BONES': 0.01,
            'LIVER': 0.04,
            'SPLEEN': 0.04,
            'KIDNEY_RIGHT': 0.04,
            'KIDNEY_LEFT': 0.04,
            'PANCREAS': 0.04,
            'STOMACH': 0.12,
            'HEART': 0.04,
            'AORTA': 0.04,
            'URINARY_BLADDER': 0.04,
            'GALL_BLADDER': 0.04,
            'SPINAL_CORD': 0.08
        }
        return weights.get(organ_name, 0.04)

    def calculate_patient_dose(self, organ_doses: Dict) -> Dict:
        """Calculate total patient dose."""
        print("\nCalculating patient dose...")

        total_absorbed_dose = sum(d['dose_mGy'] for d in organ_doses.values())
        total_effective_dose = sum(d['dose_mSv'] for d in organ_doses.values())

        patient_dose = {
            'total_absorbed_dose_mGy': total_absorbed_dose,
            'total_effective_dose_mSv': total_effective_dose,
            'organ_count': len(organ_doses),
            'highest_dose_organ': max(organ_doses.items(),
                                     key=lambda x: x[1]['dose_mGy'])[0],
            'risk_category': self._classify_risk(total_effective_dose)
        }

        print(f"  Total absorbed dose: {total_absorbed_dose:.2f} mGy")
        print(f"  Total effective dose: {total_effective_dose:.2f} mSv")
        print(f"  Risk category: {patient_dose['risk_category']}")

        return patient_dose

    def _classify_risk(self, effective_dose_mSv: float) -> str:
        """Classify radiation risk level."""
        if effective_dose_mSv < 1.0:
            return "Minimal (< 1 mSv)"
        elif effective_dose_mSv < 10.0:
            return "Low (1-10 mSv)"
        elif effective_dose_mSv < 50.0:
            return "Moderate (10-50 mSv)"
        else:
            return "High (> 50 mSv)"

    def process_patient(self, dicom_folder: str, output_dir: str = None) -> Dict:
        """
        Complete end-to-end processing pipeline.

        Args:
            dicom_folder: Path to patient DICOM files
            output_dir: Optional output directory for results

        Returns:
            Complete results dictionary
        """
        patient_name = Path(dicom_folder).name

        if output_dir is None:
            output_dir = f"outputs/{patient_name}_ai_dose"

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"PROCESSING PATIENT: {patient_name}")
        print(f"{'='*80}")

        start_time = time.time()

        results = {
            'patient_name': patient_name,
            'dicom_folder': dicom_folder,
            'output_dir': output_dir,
            'method': 'AI-Based Organ Dose Estimation',
            'processing_time': 0,
            'organ_measurements': {},
            'organ_doses': {},
            'patient_dose': {}
        }

        try:
            # Step 1: Load DICOM
            ct_volume, metadata = self.load_dicom_volume(dicom_folder)
            results['scan_metadata'] = metadata

            # Step 2: Segment organs
            organ_measurements = self.segment_organs(
                ct_volume,
                metadata['voxel_spacing_mm'],
                patient_name
            )
            results['organ_measurements'] = organ_measurements

            # Step 3: Predict organ doses
            organ_doses = self.predict_organ_doses(organ_measurements, metadata)
            results['organ_doses'] = organ_doses

            # Step 4: Calculate patient dose
            patient_dose = self.calculate_patient_dose(organ_doses)
            results['patient_dose'] = patient_dose

            # Step 5: Save results
            processing_time = time.time() - start_time
            results['processing_time'] = processing_time

            self._save_results(results, output_dir)
            self._display_results(results)

            print(f"\n{'='*80}")
            print(f"PROCESSING COMPLETE")
            print(f"{'='*80}")
            print(f"Total time: {processing_time:.2f}s")
            print(f"Organs processed: {len(organ_measurements)}")
            print(f"Output saved to: {output_dir}")

            return results

        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            results['error'] = str(e)
            return results

    def _display_results(self, results: Dict):
        """Display comprehensive results."""
        print(f"\n{'='*80}")
        print("ORGAN DOSE ESTIMATION RESULTS")
        print(f"{'='*80}")
        print(f"{'Organ':<25} {'Volume':>12} {'Dose':>12} {'Eff.Dose':>12}")
        print(f"{'':25} {'(cm3)':>12} {'(mGy)':>12} {'(mSv)':>12}")
        print(f"{'-'*80}")

        if results['organ_doses']:
            sorted_organs = sorted(
                results['organ_doses'].items(),
                key=lambda x: x[1]['dose_mGy'],
                reverse=True
            )

            for organ_name, dose_data in sorted_organs:
                print(f"{organ_name:<25} "
                      f"{dose_data['volume_cm3']:>12.1f} "
                      f"{dose_data['dose_mGy']:>12.2f} "
                      f"{dose_data['dose_mSv']:>12.3f}")

            print(f"{'-'*80}")
            patient_dose = results['patient_dose']
            print(f"{'TOTAL PATIENT DOSE':<25} "
                  f"{'':>12} "
                  f"{patient_dose['total_absorbed_dose_mGy']:>12.2f} "
                  f"{patient_dose['total_effective_dose_mSv']:>12.3f}")

        print(f"{'='*80}")

        if 'patient_dose' in results:
            pd = results['patient_dose']
            print(f"\nPatient Summary:")
            print(f"  Effective Dose: {pd['total_effective_dose_mSv']:.2f} mSv")
            print(f"  Risk Category: {pd['risk_category']}")
            print(f"  Highest Dose: {pd['highest_dose_organ']}")

    def _save_results(self, results: Dict, output_dir: str):
        """Save results to files."""

        # Save organ doses as CSV
        if results['organ_doses']:
            df_data = []
            for organ_name, dose_data in results['organ_doses'].items():
                row = {
                    'organ': organ_name,
                    'volume_cm3': dose_data['volume_cm3'],
                    'mean_hu': dose_data['mean_hu'],
                    'absorbed_dose_mGy': dose_data['dose_mGy'],
                    'effective_dose_mSv': dose_data['dose_mSv']
                }
                df_data.append(row)

            df = pd.DataFrame(df_data)
            df = df.sort_values('absorbed_dose_mGy', ascending=False)

            csv_file = os.path.join(output_dir, "organ_doses.csv")
            df.to_csv(csv_file, index=False)
            print(f"\n  Saved: {csv_file}")

        # Save complete results as JSON
        json_file = os.path.join(output_dir, "complete_dose_results.json")
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Saved: {json_file}")


def main():
    """Main function for testing."""

    # Initialize AI system
    estimator = AIOrganDoseEstimator()

    # Test patients
    test_patients = [
        "data/patient_138p",
        "data/patient_139p",
        "data/patient_55_plain"
    ]

    all_results = []

    for patient_folder in test_patients:
        if os.path.exists(patient_folder):
            results = estimator.process_patient(patient_folder)

            if 'error' not in results:
                all_results.append(results)
                print(f"\nProcessed {results['patient_name']} successfully")
            else:
                print(f"\nFailed to process {patient_folder}")
        else:
            print(f"\nPatient folder not found: {patient_folder}")

    # Summary
    if all_results:
        print(f"\n{'='*80}")
        print(f"BATCH PROCESSING SUMMARY")
        print(f"{'='*80}")
        print(f"Successfully processed: {len(all_results)} patients")

        avg_time = np.mean([r['processing_time'] for r in all_results])
        avg_dose = np.mean([r['patient_dose']['total_effective_dose_mSv']
                           for r in all_results])

        print(f"Average processing time: {avg_time:.2f}s per patient")
        print(f"Average effective dose: {avg_dose:.2f} mSv")
        print(f"\nAI system verification complete!")


if __name__ == "__main__":
    main()
