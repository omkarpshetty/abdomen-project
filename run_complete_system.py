"""
COMPLETE AI-BASED ORGAN DOSE ESTIMATION SYSTEM
===============================================
Version: 2.0 - Fixed all organ detection issues

Run this file to process patients and get organ dose estimates.
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
from typing import Dict, Tuple

# Use improved segmenter
from models.improved_segmenter import ImprovedOrganSegmenter


class CompleteAIDoseSystem:
    """
    Complete AI system with all fixes applied.
    """

    def __init__(self):
        """Initialize the complete system."""
        print("="*80)
        print("AI-BASED ORGAN DOSE ESTIMATION SYSTEM v2.0")
        print("="*80)
        print("Initializing...")

        self.organ_segmenter = ImprovedOrganSegmenter()
        print("  [OK] Organ annotator loaded")
        print("="*80)

    def load_dicom_volume(self, dicom_folder: str) -> Tuple[np.ndarray, Dict]:
        """Load DICOM files."""
        import pydicom

        print(f"\nLoading DICOM: {dicom_folder}")
        dcm_files = sorted(list(Path(dicom_folder).glob("*.dcm")))

        if not dcm_files:
            raise ValueError(f"No DICOM files in {dicom_folder}")

        print(f"  Found {len(dcm_files)} slices")

        first_ds = pydicom.dcmread(str(dcm_files[0]))
        volume = np.zeros((len(dcm_files), 512, 512), dtype=np.float32)

        for i, dcm_file in enumerate(dcm_files):
            ds = pydicom.dcmread(str(dcm_file))
            pixels = ds.pixel_array.astype(np.float32)
            slope = getattr(ds, 'RescaleSlope', 1.0)
            intercept = getattr(ds, 'RescaleIntercept', 0.0)
            volume[i] = pixels * slope + intercept

        pixel_spacing = first_ds.PixelSpacing
        slice_thickness = float(getattr(first_ds, 'SliceThickness', 10.0))
        voxel_spacing = (slice_thickness, float(pixel_spacing[0]), float(pixel_spacing[1]))

        metadata = {
            'num_slices': len(dcm_files),
            'voxel_spacing_mm': voxel_spacing,
            'kvp': float(getattr(first_ds, 'KVP', 120)),
            'slice_thickness': slice_thickness
        }

        print(f"  Volume: {volume.shape}")
        return volume, metadata

    def process_patient(self, dicom_folder: str, output_dir: str = None) -> Dict:
        """Process complete patient."""
        patient_name = Path(dicom_folder).name

        if output_dir is None:
            output_dir = f"outputs/{patient_name}_ai_dose_v2"

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"PROCESSING: {patient_name}")
        print(f"{'='*80}")

        start_time = time.time()

        try:
            # Load DICOM
            ct_volume, metadata = self.load_dicom_volume(dicom_folder)

            # Segment organs
            print("\nSegmenting organs...")
            organ_measurements = self.organ_segmenter.segment_patient(
                ct_volume,
                metadata['voxel_spacing_mm'],
                patient_name
            )

            # Predict doses
            print("\nPredicting organ doses...")
            organ_doses = self._predict_doses(organ_measurements, metadata)

            # Calculate patient dose
            patient_dose = self._calculate_patient_dose(organ_doses)

            # Prepare results
            results = {
                'patient_name': patient_name,
                'processing_time': time.time() - start_time,
                'method': 'AI Dose Estimation v2.0',
                'organ_measurements': organ_measurements,
                'organ_doses': organ_doses,
                'patient_dose': patient_dose
            }

            # Save and display
            self._save_results(results, output_dir)
            self._display_results(results)

            print(f"\n{'='*80}")
            print("PROCESSING COMPLETE")
            print(f"{'='*80}")
            print(f"Time: {results['processing_time']:.2f}s")
            print(f"Organs: {len(organ_measurements)}")
            print(f"Output: {output_dir}")

            return results

        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}

    def _predict_doses(self, organ_measurements: Dict, scan_params: Dict) -> Dict:
        """Predict organ doses."""
        organ_doses = {}
        kvp = scan_params.get('kvp', 120)
        num_slices = scan_params.get('num_slices', 50)
        total_exposure = kvp * num_slices * 0.1

        organ_factors = {
            'LIVER': 1.2, 'SPLEEN': 1.1, 'KIDNEY_RIGHT': 1.0, 'KIDNEY_LEFT': 1.0,
            'PANCREAS': 1.0, 'STOMACH': 0.9, 'GALL_BLADDER': 1.0, 'HEART': 1.1,
            'AORTA': 0.7, 'URINARY_BLADDER': 0.9, 'SPINAL_CORD': 1.3, 'BONES': 0.8
        }

        tissue_weights = {
            'LIVER': 0.04, 'SPLEEN': 0.04, 'KIDNEY_RIGHT': 0.04, 'KIDNEY_LEFT': 0.04,
            'PANCREAS': 0.04, 'STOMACH': 0.12, 'GALL_BLADDER': 0.04, 'HEART': 0.04,
            'AORTA': 0.04, 'URINARY_BLADDER': 0.04, 'SPINAL_CORD': 0.08, 'BONES': 0.01
        }

        for organ_name, stats in organ_measurements.items():
            factor = organ_factors.get(organ_name, 1.0)
            density_factor = 1.0 + (stats['mean_hu'] / 1000.0)
            dose_mGy = total_exposure * factor * density_factor / 100.0

            organ_doses[organ_name] = {
                'dose_mGy': max(0, dose_mGy),
                'dose_mSv': max(0, dose_mGy * tissue_weights.get(organ_name, 0.04)),
                'volume_cm3': stats['volume_cm3'],
                'mean_hu': stats['mean_hu']
            }

        return organ_doses

    def _calculate_patient_dose(self, organ_doses: Dict) -> Dict:
        """Calculate total patient dose."""
        total_absorbed = sum(d['dose_mGy'] for d in organ_doses.values())
        total_effective = sum(d['dose_mSv'] for d in organ_doses.values())

        if total_effective < 1.0:
            risk = "Minimal (< 1 mSv)"
        elif total_effective < 10.0:
            risk = "Low (1-10 mSv)"
        elif total_effective < 50.0:
            risk = "Moderate (10-50 mSv)"
        else:
            risk = "High (> 50 mSv)"

        highest = max(organ_doses.items(), key=lambda x: x[1]['dose_mGy'])[0]

        return {
            'total_absorbed_dose_mGy': total_absorbed,
            'total_effective_dose_mSv': total_effective,
            'organ_count': len(organ_doses),
            'highest_dose_organ': highest,
            'risk_category': risk
        }

    def _display_results(self, results: Dict):
        """Display results."""
        print(f"\n{'='*80}")
        print("ORGAN DOSE RESULTS")
        print(f"{'='*80}")
        print(f"{'Organ':<20} {'Volume':>12} {'Dose':>12} {'Eff.Dose':>12}")
        print(f"{'':20} {'(cm3)':>12} {'(mGy)':>12} {'(mSv)':>12}")
        print("-"*80)

        sorted_organs = sorted(
            results['organ_doses'].items(),
            key=lambda x: x[1]['dose_mGy'],
            reverse=True
        )

        for organ_name, dose_data in sorted_organs:
            print(f"{organ_name:<20} "
                  f"{dose_data['volume_cm3']:>12.1f} "
                  f"{dose_data['dose_mGy']:>12.2f} "
                  f"{dose_data['dose_mSv']:>12.3f}")

        print("-"*80)
        pd = results['patient_dose']
        print(f"{'TOTAL':<20} "
              f"{'':>12} "
              f"{pd['total_absorbed_dose_mGy']:>12.2f} "
              f"{pd['total_effective_dose_mSv']:>12.3f}")
        print("="*80)

        print(f"\nPatient Summary:")
        print(f"  Effective Dose: {pd['total_effective_dose_mSv']:.2f} mSv")
        print(f"  Risk Category: {pd['risk_category']}")
        print(f"  Highest Dose Organ: {pd['highest_dose_organ']}")

    def _save_results(self, results: Dict, output_dir: str):
        """Save results."""
        # CSV
        if results['organ_doses']:
            df_data = []
            for organ, dose_data in results['organ_doses'].items():
                df_data.append({
                    'organ': organ,
                    'volume_cm3': dose_data['volume_cm3'],
                    'mean_hu': dose_data['mean_hu'],
                    'absorbed_dose_mGy': dose_data['dose_mGy'],
                    'effective_dose_mSv': dose_data['dose_mSv']
                })

            df = pd.DataFrame(df_data)
            df = df.sort_values('absorbed_dose_mGy', ascending=False)
            csv_file = os.path.join(output_dir, "organ_doses.csv")
            df.to_csv(csv_file, index=False)
            print(f"\n  Saved: {csv_file}")

        # JSON
        json_file = os.path.join(output_dir, "complete_results.json")
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Saved: {json_file}")


def main():
    """Main function."""
    print("\n" + "="*80)
    print("  STARTING BATCH PROCESSING")
    print("="*80)

    system = CompleteAIDoseSystem()

    test_patients = [
        "data/patient_138p",
        "data/patient_139p",
        "data/patient_55_plain"
    ]

    all_results = []

    for patient_folder in test_patients:
        if os.path.exists(patient_folder):
            results = system.process_patient(patient_folder)
            if 'error' not in results:
                all_results.append(results)
        else:
            print(f"\nPatient not found: {patient_folder}")

    if all_results:
        print(f"\n{'='*80}")
        print("BATCH SUMMARY")
        print(f"{'='*80}")
        print(f"Processed: {len(all_results)} patients")
        avg_time = np.mean([r['processing_time'] for r in all_results])
        avg_dose = np.mean([r['patient_dose']['total_effective_dose_mSv'] for r in all_results])
        print(f"Average time: {avg_time:.2f}s")
        print(f"Average dose: {avg_dose:.2f} mSv")
        print("="*80)


if __name__ == "__main__":
    main()
