"""
Windows-Compatible Organ Annotator (No Unicode)
Fast, accurate organ segmentation with self-learning
"""

import os
import sys
import io

# Fix Windows console encoding issues
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.append('.')

import numpy as np
import pydicom
from pathlib import Path
from typing import Dict, Tuple
import time
import pandas as pd
import json

from models.self_training_segmenter import SelfTrainingOrganSegmenter


class WindowsOrganAnnotator:
    """Windows-compatible organ annotator without Unicode issues."""

    def __init__(self):
        print("=" * 80)
        print("ORGAN ANNOTATOR - PRODUCTION READY")
        print("=" * 80)
        print("Features:")
        print("  - Advanced computer vision algorithms")
        print("  - Anatomical knowledge and priors")
        print("  - Self-learning from your data")
        print("  - Fast processing (1-3 seconds)")
        print("=" * 80)

        self.segmenter = SelfTrainingOrganSegmenter()

    def load_dicom_volume(self, dicom_folder: str) -> Tuple[np.ndarray, Dict]:
        """Load DICOM series into 3D volume."""
        print(f"\n[LOADING] DICOM from: {dicom_folder}")

        dicom_files = sorted(list(Path(dicom_folder).glob("*.dcm")))

        if not dicom_files:
            raise ValueError(f"No DICOM files found in {dicom_folder}")

        print(f"   Found {len(dicom_files)} slices")

        # Load first slice
        first_ds = pydicom.dcmread(str(dicom_files[0]))
        image_shape = first_ds.pixel_array.shape

        # Initialize volume
        volume = np.zeros((len(dicom_files), image_shape[0], image_shape[1]), dtype=np.float32)

        # Load all slices
        slice_positions = []
        for i, dcm_file in enumerate(dicom_files):
            ds = pydicom.dcmread(str(dcm_file))
            pixel_array = ds.pixel_array.astype(np.float32)

            slope = getattr(ds, 'RescaleSlope', 1.0)
            intercept = getattr(ds, 'RescaleIntercept', 0.0)
            volume[i] = pixel_array * slope + intercept

            try:
                pos = float(ds.ImagePositionPatient[2])
                slice_positions.append((i, pos))
            except:
                pass

        # Sort by position
        if slice_positions:
            sorted_indices = [idx for idx, _ in sorted(slice_positions, key=lambda x: x[1])]
            volume = volume[sorted_indices]

        # Get voxel spacing
        pixel_spacing = first_ds.PixelSpacing
        slice_thickness = float(getattr(first_ds, 'SliceThickness', 5.0))
        voxel_spacing = (slice_thickness, float(pixel_spacing[0]), float(pixel_spacing[1]))

        metadata = {
            'num_slices': len(dicom_files),
            'voxel_spacing_mm': voxel_spacing,
            'image_shape': volume.shape,
            'pixel_spacing': pixel_spacing,
            'slice_thickness': slice_thickness
        }

        print(f"   [OK] Volume loaded: {volume.shape}")
        print(f"   Voxel spacing: {voxel_spacing[0]:.2f} x {voxel_spacing[1]:.2f} x {voxel_spacing[2]:.2f} mm")

        return volume, metadata

    def process_patient(self, dicom_folder: str, output_dir: str = None) -> Dict:
        """Complete processing pipeline."""
        patient_name = Path(dicom_folder).name

        if output_dir is None:
            output_dir = f"outputs/{patient_name}_annotated"

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"PROCESSING PATIENT: {patient_name}")
        print(f"{'='*80}")

        start_time = time.time()

        results = {
            'patient_name': patient_name,
            'dicom_folder': dicom_folder,
            'output_dir': output_dir,
            'method': 'Self-Training Organ Annotator',
            'processing_time': 0,
            'organ_measurements': {}
        }

        try:
            # Load DICOM
            ct_volume, metadata = self.load_dicom_volume(dicom_folder)
            results['metadata'] = metadata

            # Segment organs
            organ_measurements = self.segmenter.segment_patient(
                ct_volume,
                metadata['voxel_spacing_mm'],
                patient_id=patient_name
            )
            results['organ_measurements'] = organ_measurements

            # Save results
            processing_time = time.time() - start_time
            results['processing_time'] = processing_time

            self._save_results(results, output_dir)
            self._display_results(results)

            print(f"\n{'='*80}")
            print(f"[COMPLETE] Processing finished successfully")
            print(f"{'='*80}")
            print(f"Total time: {processing_time:.2f}s")
            print(f"Organs segmented: {len(organ_measurements)}")
            print(f"Output: {output_dir}")

            return results

        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()
            results['error'] = str(e)
            return results

    def _display_results(self, results: Dict):
        """Display results."""
        print(f"\n{'='*90}")
        print(f"ORGAN MEASUREMENTS")
        print(f"{'='*90}")
        print(f"{'Organ':<30} {'Volume (cm3)':>15} {'Mean HU':>12} {'Std HU':>12}")
        print(f"{'-'*90}")

        if results['organ_measurements']:
            sorted_organs = sorted(
                results['organ_measurements'].items(),
                key=lambda x: x[1]['volume_cm3'],
                reverse=True
            )

            for organ_name, stats in sorted_organs:
                print(f"{organ_name:<30} {stats['volume_cm3']:>15.1f} "
                      f"{stats['mean_hu']:>12.1f} {stats['std_hu']:>12.1f}")

            total_volume = sum(s['volume_cm3'] for s in results['organ_measurements'].values())
            print(f"{'-'*90}")
            print(f"{'TOTAL VOLUME':<30} {total_volume:>15.1f} cm3")

        print(f"{'='*90}")

    def _save_results(self, results: Dict, output_dir: str):
        """Save results to files."""
        if results['organ_measurements']:
            df_data = []
            for organ_name, stats in results['organ_measurements'].items():
                row = {'organ': organ_name}
                row.update(stats)
                df_data.append(row)

            df = pd.DataFrame(df_data)
            df = df.sort_values('volume_cm3', ascending=False)

            csv_file = os.path.join(output_dir, "organ_measurements.csv")
            df.to_csv(csv_file, index=False)
            print(f"\n   [SAVED] CSV: {csv_file}")

        json_file = os.path.join(output_dir, "complete_results.json")
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"   [SAVED] JSON: {json_file}")


def main():
    """Train on all 144 patients."""
    print("=" * 80)
    print("TRAINING ON 144 PATIENTS")
    print("=" * 80)

    # Find all patients
    base_dir = Path("C:/Users/omkar/OneDrive/Desktop/major project/1-144")

    if not base_dir.exists():
        print(f"[ERROR] Directory not found: {base_dir}")
        return

    patient_folders = sorted([str(f) for f in base_dir.iterdir() if f.is_dir()])
    print(f"\nFound {len(patient_folders)} patient folders\n")

    # Initialize
    annotator = WindowsOrganAnnotator()

    # Process all
    successful = 0
    failed = 0

    for i, patient_folder in enumerate(patient_folders, 1):
        patient_name = Path(patient_folder).name
        print(f"\n[{i}/{len(patient_folders)}] {patient_name}")
        print("-" * 60)

        try:
            results = annotator.process_patient(patient_folder)

            if 'error' not in results:
                successful += 1
                print(f"[SUCCESS] {len(results['organ_measurements'])} organs, {results['processing_time']:.1f}s")
            else:
                failed += 1
                print(f"[FAILED] {results['error']}")

        except Exception as e:
            failed += 1
            print(f"[ERROR] {e}")

    # Summary
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"Successfully processed: {successful}/{len(patient_folders)} patients")
    print(f"Failed: {failed}")
    print(f"\nThe system is now trained on {successful} patients!")
    print(f"Future segmentations will use learned parameters.")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
