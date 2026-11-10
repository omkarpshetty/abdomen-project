"""
Generate Synthetic Training Data for Dose Prediction Models

Creates realistic organ dose training data based on physics-based models
and ICRP reference values.
"""
import numpy as np
import pandas as pd
import argparse
from pathlib import Path


class SyntheticDoseGenerator:
    """Generate synthetic organ dose training data."""

    # Typical organ volumes (cm³)
    ORGAN_VOLUMES = {
        'LIVER': (1200, 1800),
        'SPLEEN': (150, 250),
        'KIDNEY_RIGHT': (120, 180),
        'KIDNEY_LEFT': (120, 180),
        'PANCREAS': (80, 120),
        'STOMACH': (250, 400),
        'HEART': (250, 350),
        'LUNG_LEFT': (1500, 2500),
        'LUNG_RIGHT': (1500, 2500),
        'COLON': (400, 600),
        'SMALL_BOWEL': (800, 1200),
        'URINARY_BLADDER': (150, 300),
        'GALL_BLADDER': (30, 60),
        'AORTA': (50, 100),
        'SPINAL_CORD': (20, 40),
    }

    # Typical HU values
    ORGAN_HU = {
        'LIVER': (50, 70),
        'SPLEEN': (50, 60),
        'KIDNEY_RIGHT': (30, 50),
        'KIDNEY_LEFT': (30, 50),
        'PANCREAS': (40, 50),
        'STOMACH': (20, 40),
        'HEART': (40, 60),
        'LUNG_LEFT': (-800, -600),
        'LUNG_RIGHT': (-800, -600),
        'COLON': (20, 40),
        'SMALL_BOWEL': (20, 40),
        'URINARY_BLADDER': (10, 30),
        'GALL_BLADDER': (0, 20),
        'AORTA': (40, 60),
        'SPINAL_CORD': (30, 50),
    }

    # Organ-specific dose factors (relative to CTDIvol)
    DOSE_FACTORS = {
        'LIVER': 1.0,
        'SPLEEN': 0.9,
        'KIDNEY_RIGHT': 0.85,
        'KIDNEY_LEFT': 0.85,
        'PANCREAS': 0.95,
        'STOMACH': 1.05,
        'HEART': 0.7,
        'LUNG_LEFT': 0.5,
        'LUNG_RIGHT': 0.5,
        'COLON': 1.1,
        'SMALL_BOWEL': 1.0,
        'URINARY_BLADDER': 0.95,
        'GALL_BLADDER': 0.9,
        'AORTA': 0.7,
        'SPINAL_CORD': 1.2,
    }

    def __init__(self, seed: int = 42):
        np.random.seed(seed)

    def generate_patient(self, patient_id: int) -> pd.DataFrame:
        """Generate organ data for one patient."""
        # Scan parameters
        kvp = np.random.choice([100, 120, 140])
        ctdivol = np.random.uniform(5, 25)  # mGy
        num_slices = np.random.randint(50, 250)
        slice_thickness = np.random.choice([2.5, 5.0])

        # Patient variation factors
        patient_size_factor = np.random.uniform(0.8, 1.2)
        patient_density_factor = np.random.uniform(0.9, 1.1)

        organs_data = []

        for organ_name in self.ORGAN_VOLUMES.keys():
            # Generate organ features
            vol_min, vol_max = self.ORGAN_VOLUMES[organ_name]
            volume_cm3 = np.random.uniform(vol_min, vol_max) * patient_size_factor

            hu_min, hu_max = self.ORGAN_HU[organ_name]
            mean_hu = np.random.uniform(hu_min, hu_max)
            median_hu = mean_hu + np.random.normal(0, 2)
            std_hu = np.random.uniform(5, 20)

            # Position (distance from isocenter)
            distance_from_isocenter = np.random.uniform(0, 200)

            # Shape
            compactness = np.random.uniform(0.3, 0.9)

            # Calculate absorbed dose
            base_dose = ctdivol * self.DOSE_FACTORS[organ_name]

            # Physics-based adjustments
            kvp_factor = (kvp / 120.0) ** 1.5
            thickness_factor = slice_thickness / 5.0
            volume_factor = 1.0 + (volume_cm3 / 1000.0) * 0.05
            hu_factor = 1.0 + (mean_hu / 100.0) * 0.03

            absorbed_dose = (
                base_dose *
                kvp_factor *
                thickness_factor *
                volume_factor *
                hu_factor *
                patient_density_factor *
                np.random.normal(1.0, 0.1)  # Random variation
            )

            # Ensure positive
            absorbed_dose = max(0.1, absorbed_dose)

            organs_data.append({
                'patient_id': patient_id,
                'organ': organ_name,
                'volume_cm3': volume_cm3,
                'mean_hu': mean_hu,
                'median_hu': median_hu,
                'std_hu': std_hu,
                'distance_from_isocenter_mm': distance_from_isocenter,
                'compactness': compactness,
                'kvp': kvp,
                'ctdivol_mgy': ctdivol,
                'num_slices': num_slices,
                'slice_thickness_mm': slice_thickness,
                'absorbed_dose_mgy': absorbed_dose,
            })

        return pd.DataFrame(organs_data)

    def generate_dataset(self, num_patients: int) -> pd.DataFrame:
        """Generate complete training dataset."""
        print(f"Generating synthetic training data for {num_patients} patients...")

        all_data = []
        for patient_id in range(1, num_patients + 1):
            if patient_id % 100 == 0:
                print(f"  Generated {patient_id}/{num_patients} patients")

            patient_data = self.generate_patient(patient_id)
            all_data.append(patient_data)

        df = pd.concat(all_data, ignore_index=True)

        print(f"\nDataset generated:")
        print(f"  Total samples: {len(df)}")
        print(f"  Patients: {num_patients}")
        print(f"  Organs per patient: {len(self.ORGAN_VOLUMES)}")
        print(f"  Features: {len(df.columns) - 1}")

        return df


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic training data for dose prediction'
    )
    parser.add_argument('--num-patients', type=int, default=1000,
                       help='Number of patients to generate')
    parser.add_argument('--output', type=str, default='training_data.csv',
                       help='Output CSV file path')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Generate data
    generator = SyntheticDoseGenerator(seed=args.seed)
    df = generator.generate_dataset(args.num_patients)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n✓ Training data saved to: {output_path}")
    print(f"\nTo train the model, run:")
    print(f"  python ai_dose_system.py --train --training-data {output_path}")

    # Show statistics
    print("\nData Statistics:")
    print(f"  Dose range: {df['absorbed_dose_mgy'].min():.2f} - {df['absorbed_dose_mgy'].max():.2f} mGy")
    print(f"  Mean dose: {df['absorbed_dose_mgy'].mean():.2f} mGy")
    print(f"  CTDIvol range: {df['ctdivol_mgy'].min():.2f} - {df['ctdivol_mgy'].max():.2f} mGy")

    print("\nOrgans included:")
    for organ in sorted(df['organ'].unique()):
        count = len(df[df['organ'] == organ])
        mean_dose = df[df['organ'] == organ]['absorbed_dose_mgy'].mean()
        print(f"  {organ:<20} {count:>4} samples  (mean: {mean_dose:.2f} mGy)")


if __name__ == "__main__":
    main()
