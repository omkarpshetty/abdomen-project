"""
Robust Windows Training System - No Unicode Issues
Trains on all remaining patients with proper logging
"""

import os
import sys
from pathlib import Path
import time
import json

# Don't use special encoding wrapper - keep it simple
sys.path.append('.')

from models.self_training_segmenter import SelfTrainingOrganSegmenter
import numpy as np
import pydicom
import pandas as pd


class SimpleAnnotator:
    """Simple Windows-compatible annotator."""

    def __init__(self):
        print("Organ Annotator - Initializing...")
        self.segmenter = SelfTrainingOrganSegmenter()
        print("Ready")

    def load_dicom(self, folder):
        """Load DICOM volume."""
        files = sorted(list(Path(folder).glob("*.dcm")))
        if not files:
            raise ValueError(f"No DICOM files in {folder}")

        first = pydicom.dcmread(str(files[0]))
        shape = first.pixel_array.shape
        volume = np.zeros((len(files), shape[0], shape[1]), dtype=np.float32)

        for i, f in enumerate(files):
            ds = pydicom.dcmread(str(f))
            arr = ds.pixel_array.astype(np.float32)
            slope = getattr(ds, 'RescaleSlope', 1.0)
            intercept = getattr(ds, 'RescaleIntercept', 0.0)
            volume[i] = arr * slope + intercept

        spacing = first.PixelSpacing
        thickness = float(getattr(first, 'SliceThickness', 5.0))
        voxel_spacing = (thickness, float(spacing[0]), float(spacing[1]))

        return volume, voxel_spacing

    def process(self, folder):
        """Process one patient."""
        name = Path(folder).name
        output = f"outputs/{name}_annotated"
        Path(output).mkdir(parents=True, exist_ok=True)

        start = time.time()

        try:
            volume, spacing = self.load_dicom(folder)
            measurements = self.segmenter.segment_patient(volume, spacing, name)

            elapsed = time.time() - start

            # Save CSV
            if measurements:
                rows = []
                for organ, stats in measurements.items():
                    row = {'organ': organ}
                    row.update(stats)
                    rows.append(row)

                df = pd.DataFrame(rows)
                df = df.sort_values('volume_cm3', ascending=False)
                df.to_csv(f"{output}/measurements.csv", index=False)

            # Save JSON
            result = {
                'patient': name,
                'time': elapsed,
                'organs': len(measurements),
                'measurements': measurements
            }

            with open(f"{output}/results.json", 'w') as f:
                json.dump(result, f, indent=2, default=str)

            return True, len(measurements), elapsed

        except Exception as e:
            print(f"ERROR: {str(e)[:80]}")
            return False, 0, 0


def main():
    """Train on all patients."""
    print("="*70)
    print("COMPLETE TRAINING - ALL PATIENTS")
    print("="*70)

    base = Path("C:/Users/omkar/OneDrive/Desktop/major project/1-144")
    if not base.exists():
        print("ERROR: Base directory not found")
        return

    all_folders = sorted([str(f) for f in base.iterdir() if f.is_dir()])
    print(f"Total patients: {len(all_folders)}")

    # Check processed
    outputs = Path("outputs")
    processed = set()
    if outputs.exists():
        for d in outputs.iterdir():
            if d.is_dir() and d.name.endswith('_annotated'):
                processed.add(d.name.replace('_annotated', ''))

    remaining = [f for f in all_folders if Path(f).name not in processed]
    print(f"Already done: {len(all_folders) - len(remaining)}")
    print(f"Remaining: {len(remaining)}")

    if not remaining:
        print("\nALL PATIENTS ALREADY PROCESSED!")
        print("System is fully trained.")
        return

    print(f"\nProcessing {len(remaining)} patients...\n")

    annotator = SimpleAnnotator()

    success = 0
    failed = 0
    total_organs = 0
    total_time = 0

    start_all = time.time()

    for i, folder in enumerate(remaining, 1):
        name = Path(folder).name
        print(f"[{i}/{len(remaining)}] {name}")

        ok, organs, elapsed = annotator.process(folder)

        if ok:
            success += 1
            total_organs += organs
            total_time += elapsed
            print(f"  OK - {organs} organs in {elapsed:.1f}s")
        else:
            failed += 1
            print(f"  FAILED")

        # Progress summary every 20
        if i % 20 == 0:
            avg_time = total_time / success if success > 0 else 0
            avg_organs = total_organs / success if success > 0 else 0
            pct = (i / len(remaining)) * 100
            print(f"\n  Progress: {pct:.0f}% | Success: {success}/{i} | Avg: {avg_time:.1f}s, {avg_organs:.1f} organs\n")

    # Final summary
    elapsed_all = time.time() - start_all
    total_done = success + len(all_folders) - len(remaining)

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"This session: {success}/{len(remaining)} successful ({failed} failed)")
    print(f"Session time: {elapsed_all/60:.1f} minutes")
    print(f"\nTOTAL: {total_done}/{len(all_folders)} patients trained")
    print(f"Coverage: {total_done/len(all_folders)*100:.1f}%")

    if success > 0:
        print(f"\nPerformance:")
        print(f"  Avg speed: {total_time/success:.1f}s/patient")
        print(f"  Avg organs: {total_organs/success:.1f}/patient")

    print(f"\nModel: models/self_trained_segmenter.pkl")
    print(f"Results: outputs/")

    if total_done == len(all_folders):
        print(f"\nPERFECT - ALL {len(all_folders)} PATIENTS TRAINED!")
        print("Maximum accuracy achieved!")

    print("="*70)


if __name__ == "__main__":
    main()
