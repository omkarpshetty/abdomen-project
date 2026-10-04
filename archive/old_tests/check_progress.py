"""
Training Progress Monitor
Check status of ongoing training
"""

import os
from pathlib import Path
import time

def main():
    print("="*70)
    print("TRAINING PROGRESS MONITOR")
    print("="*70)

    # Count processed patients
    outputs_dir = Path("outputs")
    processed_count = 0

    if outputs_dir.exists():
        processed_count = len([d for d in outputs_dir.iterdir()
                              if d.is_dir() and d.name.endswith('_annotated')])

    # Total patients
    base_dir = Path("C:/Users/omkar/OneDrive/Desktop/major project/1-144")
    total_count = 0

    if base_dir.exists():
        total_count = len([d for d in base_dir.iterdir() if d.is_dir()])

    # Calculate progress
    if total_count > 0:
        progress_pct = (processed_count / total_count) * 100
        remaining = total_count - processed_count

        print(f"\nStatus:")
        print(f"  Total patients: {total_count}")
        print(f"  Processed: {processed_count}")
        print(f"  Remaining: {remaining}")
        print(f"  Progress: {progress_pct:.1f}%")

        # Progress bar
        bar_length = 50
        filled = int(bar_length * processed_count / total_count)
        bar = '#' * filled + '-' * (bar_length - filled)
        print(f"\n  [{bar}] {progress_pct:.1f}%")

        if remaining > 0:
            # Estimate time remaining (assume 2.5s per patient)
            est_minutes = (remaining * 2.5) / 60
            print(f"\n  Estimated time remaining: {est_minutes:.1f} minutes")
            print(f"  Status: TRAINING IN PROGRESS...")
        else:
            print(f"\n  Status: COMPLETE - All patients processed!")
            print(f"\n  Model file: models/self_trained_segmenter.pkl")
            print(f"  Results: outputs/")
    else:
        print("\nERROR: Base directory not found")

    print("="*70)

    # Check model file
    model_path = Path("models/self_trained_segmenter.pkl")
    if model_path.exists():
        size_kb = model_path.stat().st_size / 1024
        print(f"\nTrained model: {model_path} ({size_kb:.1f} KB)")
        print(f"Patients learned from: {processed_count}")

    print()

if __name__ == "__main__":
    main()
