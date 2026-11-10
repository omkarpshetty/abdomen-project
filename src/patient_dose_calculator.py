"""
Patient Total Dose Calculator

Calculates total absorbed dose and effective dose for patients using
ICRP tissue weighting factors.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import os


class PatientDoseCalculator:
    """
    Calculate patient total doses from organ-specific doses.

    Uses ICRP 103 (2007) tissue weighting factors for effective dose calculation.
    Can also predict individual organ doses using trained AI ensemble model.
    """

    # ICRP 103 Tissue Weighting Factors
    TISSUE_WEIGHTS = {
        'LIVER': 0.04,
        'SPLEEN': 0.04,
        'KIDNEYS': 0.04,
        'PANCREAS': 0.04,
        'STOMACH': 0.12,
        'BOWEL': 0.12,
        'URINARY BLADDER': 0.04,
        'PROSTATE': 0.04,
        'UTERUS': 0.08,
        'HEART': 0.04,
        'LUNGS': 0.12,
        'ESOPHAGUS': 0.04,
        'THYROID': 0.04,
        'BRAIN': 0.01,
        'SALIVARY_GLANDS': 0.01,
        'BONE_SURFACE': 0.01,
        'SKIN': 0.01,
        'BREAST': 0.12,
        'GONADS': 0.08,
        'RED_MARROW': 0.12,
        'SPINAL CORD': 0.04,
        'ADRENAL': 0.04,
        'GALL BLADDER': 0.04,
        'VESSELS': 0.04,
        'BONES': 0.01,
        'MUSCLE': 0.04,
        # Remainder tissues
        'OTHER': 0.12,  # Split among remainder tissues
    }

    # Risk categories based on effective dose (mSv)
    RISK_CATEGORIES = [
        (0, 1, "Minimal", "Equivalent to natural background radiation"),
        (1, 10, "Low", "Comparable to annual natural background"),
        (10, 50, "Moderate", "Regular monitoring recommended"),
        (50, 100, "High", "Close monitoring required"),
        (100, float('inf'), "Very High", "Medical intervention may be needed"),
    ]

    def __init__(self, model_dir: str = 'trained_models'):
        """
        Initialize calculator with optional AI model for dose prediction.

        Args:
            model_dir: Directory containing trained ensemble model
        """
        self.model_dir = model_dir
        self._ensemble_model = None
        self._model_loaded = False

        # Map TotalSegmentator organ names to trained model organ names
        self.organ_name_mapping = {
            'ADRENAL GLANDS': 'KIDNEYS',  # Adrenal glands near kidneys, similar dose profile
            'ESOPHAGUS': 'STOMACH',       # Esophagus in similar region as stomach
            'KIDNEY_LEFT': 'KIDNEYS',
            'KIDNEY_RIGHT': 'KIDNEYS',
            'KIDNEYS': 'KIDNEYS',
            'LUNGS': 'LUNGS',
            'GALLBLADDER': 'GALL BLADDER',
            'GALL BLADDER': 'GALL BLADDER',
            'GALLBLADDER ': 'GALL BLADDER',
        }

    def _map_organ_name(self, organ_name: str) -> str:
        """Map organ name to trained model vocabulary."""
        # Direct match
        if hasattr(self, '_ensemble_model') and self._ensemble_model:
            if organ_name in self._ensemble_model.organ_encoder.classes_:
                return organ_name

        # Try mapped name
        mapped = self.organ_name_mapping.get(organ_name, organ_name)
        if hasattr(self, '_ensemble_model') and self._ensemble_model:
            if mapped in self._ensemble_model.organ_encoder.classes_:
                return mapped

        # Default fallback
        return 'BOWEL'  # Generic fallback for abdominal organs

    def _load_ensemble_model(self):
        """Lazy load the ensemble model when needed."""
        if self._model_loaded:
            return self._ensemble_model

        try:
            from models.ensemble_dose_predictor import EnsembleDosePredictor

            if os.path.exists(self.model_dir) and os.path.exists(os.path.join(self.model_dir, 'metadata.json')):
                self._ensemble_model = EnsembleDosePredictor.load(self.model_dir, device='cpu')
                self._model_loaded = True
                return self._ensemble_model
            else:
                self._model_loaded = True
                return None
        except Exception as e:
            print(f"Warning: Could not load AI model: {e}")
            self._model_loaded = True
            return None

    def predict_organ_dose(self,
                          organ_name: str,
                          organ_volume_cm3: float,
                          mean_hu: float,
                          distance_from_isocenter_mm: float,
                          kvp: float = 120.0,
                          mas: float = 200.0,
                          median_hu: float = None,
                          std_hu: float = None,
                          compactness: float = None,
                          ctdivol_mgy: float = None,
                          num_slices: int = None,
                          slice_thickness_mm: float = None) -> float:
        """
        Predict absorbed dose for a single organ using AI ensemble model.

        Args:
            organ_name: Name of the organ (e.g., 'LIVER', 'SPLEEN')
            organ_volume_cm3: Organ volume in cubic centimeters
            mean_hu: Mean Hounsfield Unit value
            distance_from_isocenter_mm: Distance from scan isocenter
            kvp: Tube voltage (default 120 kV)
            mas: Tube current-time product (default 200 mAs)
            median_hu: Median HU (defaults to mean_hu if not provided)
            std_hu: Standard deviation of HU (defaults to 50 if not provided)
            compactness: Shape compactness (defaults to 0.5 if not provided)
            ctdivol_mgy: CT dose index (estimated from kvp/mas if not provided)
            num_slices: Number of slices (defaults to 100 if not provided)
            slice_thickness_mm: Slice thickness (defaults to 5.0 if not provided)

        Returns:
            Predicted absorbed dose in mGy

        Raises:
            RuntimeError: If model is not available or not trained
        """
        # Load model if needed
        model = self._load_ensemble_model()

        if model is None or not model._is_fitted:
            raise RuntimeError(
                "AI model not available or not trained. "
                "Please train the model first using train_ai_dose_model.py"
            )

        # Set defaults for optional parameters
        if median_hu is None:
            median_hu = mean_hu
        if std_hu is None:
            std_hu = 50.0
        if compactness is None:
            compactness = 0.5
        if ctdivol_mgy is None:
            # Estimate CTDIvol from kvp and mAs
            ctdivol_mgy = (kvp / 120.0) * (mas / 200.0) * 10.0
        if num_slices is None:
            num_slices = 100
        if slice_thickness_mm is None:
            slice_thickness_mm = 5.0

        # Map organ name to trained model vocabulary
        mapped_organ_name = self._map_organ_name(organ_name)

        # Prepare features as DataFrame
        features_df = pd.DataFrame([{
            'organ': mapped_organ_name,
            'volume_cm3': organ_volume_cm3,
            'mean_hu': mean_hu,
            'median_hu': median_hu,
            'std_hu': std_hu,
            'distance_from_isocenter_mm': distance_from_isocenter_mm,
            'compactness': compactness,
            'kvp': kvp,
            'ctdivol_mgy': ctdivol_mgy,
            'num_slices': num_slices,
            'slice_thickness_mm': slice_thickness_mm
        }])

        # Predict dose
        try:
            predictions = model.predict(features_df)
            dose_mgy = float(predictions[0])

            # Ensure non-negative
            dose_mgy = max(0.0, dose_mgy)

            return dose_mgy

        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")

    def calculate_patient_dose(self, organ_doses: Dict[str, Dict]) -> Dict:
        """
        Calculate total patient dose from individual organ doses.

        Args:
            organ_doses: Dictionary mapping organ names to dose info
                        {organ_name: {'dose_mgy': float, 'volume_cm3': float, ...}}

        Returns:
            Dictionary with patient dose summary
        """
        if not organ_doses:
            return {
                'total_absorbed_dose_mgy': 0.0,
                'total_effective_dose_msv': 0.0,
                'organ_count': 0,
                'risk_category': 'Unknown',
                'risk_description': '',
                'organ_contributions': {}
            }

        # Calculate total absorbed dose (sum of all organ doses)
        total_absorbed_dose = sum(
            dose_info.get('dose_mgy', 0)
            for dose_info in organ_doses.values()
        )

        # Calculate effective dose using tissue weighting factors
        total_effective_dose = 0.0
        organ_contributions = {}

        for organ_name, dose_info in organ_doses.items():
            absorbed_dose = dose_info.get('dose_mgy', 0)

            # Get tissue weighting factor
            tissue_weight = self.TISSUE_WEIGHTS.get(organ_name, 0.04)

            # Effective dose for this organ = absorbed dose × tissue weight
            effective_dose = absorbed_dose * tissue_weight
            total_effective_dose += effective_dose

            # Store contribution
            organ_contributions[organ_name] = {
                'absorbed_dose_mgy': absorbed_dose,
                'tissue_weight': tissue_weight,
                'effective_dose_msv': effective_dose,
                'percentage': 0.0  # Will calculate after total known
            }

        # Calculate percentage contributions
        if total_effective_dose > 0:
            for organ_name in organ_contributions:
                organ_contributions[organ_name]['percentage'] = (
                    organ_contributions[organ_name]['effective_dose_msv'] /
                    total_effective_dose * 100
                )

        # Determine risk category
        risk_category, risk_description = self._classify_risk(total_effective_dose)

        # Find organ with highest dose
        if organ_doses:
            highest_dose_organ = max(
                organ_doses.items(),
                key=lambda x: x[1].get('dose_mgy', 0)
            )[0]
        else:
            highest_dose_organ = None

        return {
            'total_absorbed_dose_mgy': float(total_absorbed_dose),
            'total_effective_dose_msv': float(total_effective_dose),
            'organ_count': len(organ_doses),
            'risk_category': risk_category,
            'risk_description': risk_description,
            'highest_dose_organ': highest_dose_organ,
            'highest_dose_value_mgy': organ_doses[highest_dose_organ].get('dose_mgy', 0) if highest_dose_organ else 0,
            'organ_contributions': organ_contributions,
            'equivalent_chest_xrays': self._to_chest_xray_equivalent(total_effective_dose),
            'equivalent_background_days': self._to_background_equivalent(total_effective_dose),
        }

    def _classify_risk(self, effective_dose_msv: float) -> tuple:
        """Classify radiation risk level."""
        for min_dose, max_dose, category, description in self.RISK_CATEGORIES:
            if min_dose <= effective_dose_msv < max_dose:
                return category, description
        return "Unknown", "Dose classification not available"

    def _to_chest_xray_equivalent(self, effective_dose_msv: float) -> float:
        """Convert to chest X-ray equivalents (1 chest X-ray ≈ 0.02 mSv)."""
        return effective_dose_msv / 0.02

    def _to_background_equivalent(self, effective_dose_msv: float) -> float:
        """Convert to days of natural background radiation (≈ 0.0082 mSv/day)."""
        return effective_dose_msv / 0.0082

    def generate_summary_report(self, patient_dose: Dict, patient_name: str = None) -> str:
        """Generate a human-readable summary report."""
        lines = []
        lines.append("=" * 80)
        lines.append("PATIENT RADIATION DOSE REPORT")
        lines.append("=" * 80)

        if patient_name:
            lines.append(f"Patient: {patient_name}")

        lines.append(f"\nTotal Absorbed Dose: {patient_dose['total_absorbed_dose_mgy']:.2f} mGy")
        lines.append(f"Total Effective Dose: {patient_dose['total_effective_dose_msv']:.2f} mSv")
        lines.append(f"\nRisk Category: {patient_dose['risk_category']}")
        lines.append(f"Description: {patient_dose['risk_description']}")

        lines.append(f"\nEquivalent Radiation Exposure:")
        lines.append(f"  • Chest X-rays: {patient_dose['equivalent_chest_xrays']:.1f}")
        lines.append(f"  • Natural background: {patient_dose['equivalent_background_days']:.1f} days")

        lines.append(f"\nOrgans Analyzed: {patient_dose['organ_count']}")
        lines.append(f"Highest Dose Organ: {patient_dose['highest_dose_organ']} "
                    f"({patient_dose['highest_dose_value_mgy']:.2f} mGy)")

        # Top contributors
        lines.append(f"\nTop 5 Contributors to Effective Dose:")
        contributions = sorted(
            patient_dose['organ_contributions'].items(),
            key=lambda x: x[1]['effective_dose_msv'],
            reverse=True
        )[:5]

        for organ_name, contrib in contributions:
            lines.append(
                f"  {organ_name:<20} "
                f"{contrib['effective_dose_msv']:>8.3f} mSv "
                f"({contrib['percentage']:>5.1f}%)"
            )

        lines.append("=" * 80)

        return "\n".join(lines)

    def compare_to_reference_levels(self, effective_dose_msv: float) -> Dict:
        """Compare patient dose to common reference levels."""
        references = {
            'chest_xray': 0.02,
            'mammogram': 0.4,
            'head_ct': 2.0,
            'chest_ct': 7.0,
            'abdomen_ct': 10.0,
            'annual_natural_background': 3.0,
            'annual_radiation_worker_limit': 20.0,
            'max_single_ct_scan': 25.0,
        }

        comparisons = {}
        for ref_name, ref_dose in references.items():
            comparisons[ref_name] = {
                'reference_dose_msv': ref_dose,
                'ratio': effective_dose_msv / ref_dose if ref_dose > 0 else 0,
                'difference_msv': effective_dose_msv - ref_dose,
                'higher': effective_dose_msv > ref_dose
            }

        return comparisons


def create_organ_dose_table(organ_doses: Dict[str, Dict],
                            patient_dose: Dict) -> str:
    """Create formatted table of organ doses."""
    lines = []
    lines.append("=" * 100)
    lines.append("ORGAN-SPECIFIC DOSE DISTRIBUTION")
    lines.append("=" * 100)
    lines.append(f"{'Organ':<25} {'Volume':>10} {'Absorbed':>12} {'Tissue':>8} {'Effective':>12} {'Contrib':>8}")
    lines.append(f"{'':25} {'(cm³)':>10} {'Dose (mGy)':>12} {'Weight':>8} {'Dose (mSv)':>12} {'(%)':>8}")
    lines.append("-" * 100)

    # Sort by effective dose contribution
    contributions = patient_dose.get('organ_contributions', {})
    sorted_organs = sorted(
        organ_doses.keys(),
        key=lambda x: contributions.get(x, {}).get('effective_dose_msv', 0),
        reverse=True
    )

    for organ_name in sorted_organs:
        dose_info = organ_doses[organ_name]
        contrib = contributions.get(organ_name, {})

        lines.append(
            f"{organ_name:<25} "
            f"{dose_info.get('volume_cm3', 0):>10.1f} "
            f"{dose_info.get('dose_mgy', 0):>12.2f} "
            f"{contrib.get('tissue_weight', 0):>8.3f} "
            f"{contrib.get('effective_dose_msv', 0):>12.3f} "
            f"{contrib.get('percentage', 0):>8.1f}"
        )

    lines.append("-" * 100)
    lines.append(
        f"{'TOTAL':<25} "
        f"{'':>10} "
        f"{patient_dose['total_absorbed_dose_mgy']:>12.2f} "
        f"{'':>8} "
        f"{patient_dose['total_effective_dose_msv']:>12.3f} "
        f"{'100.0':>8}"
    )
    lines.append("=" * 100)

    return "\n".join(lines)


if __name__ == "__main__":
    # Test the calculator
    calculator = PatientDoseCalculator()

    # Sample organ doses
    test_organ_doses = {
        'LIVER': {'dose_mgy': 12.5, 'volume_cm3': 1500},
        'SPLEEN': {'dose_mgy': 8.3, 'volume_cm3': 200},
        'KIDNEY_RIGHT': {'dose_mgy': 10.1, 'volume_cm3': 150},
        'KIDNEY_LEFT': {'dose_mgy': 9.8, 'volume_cm3': 145},
        'STOMACH': {'dose_mgy': 11.2, 'volume_cm3': 350},
        'PANCREAS': {'dose_mgy': 7.5, 'volume_cm3': 100},
        'HEART': {'dose_mgy': 6.8, 'volume_cm3': 300},
    }

    # Calculate patient dose
    patient_dose = calculator.calculate_patient_dose(test_organ_doses)

    # Print report
    print(calculator.generate_summary_report(patient_dose, "Test Patient"))
    print("\n")
    print(create_organ_dose_table(test_organ_doses, patient_dose))

    # Compare to references
    print("\nComparison to Reference Levels:")
    comparisons = calculator.compare_to_reference_levels(
        patient_dose['total_effective_dose_msv']
    )
    for ref_name, comp in comparisons.items():
        symbol = ">" if comp['higher'] else "<"
        print(f"  {ref_name:30s} {symbol} {comp['ratio']:.2f}x")
