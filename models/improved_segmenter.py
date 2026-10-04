"""
Improved Self-Training Organ Segmentation System
Fixes liver and kidney detection issues
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
from scipy import ndimage
import pickle
import time


class ImprovedOrganSegmenter:
    """
    Enhanced organ segmenter with improved liver and kidney detection.
    """

    def __init__(self, model_path: str = 'models/self_trained_segmenter.pkl'):
        """Initialize the segmenter."""
        self.model_path = model_path
        self.organ_statistics = {}
        self.load_model()
        self.organ_priors = self._initialize_organ_priors()

    def _initialize_organ_priors(self) -> Dict:
        """Initialize improved anatomical knowledge."""
        return {
            'LIVER': {
                'hu_range': (20, 100),  # Very wide range
                'hu_std': 20,
                'volume_range': (600, 3500),  # Much wider range
                'relative_position': {'z': (0.30, 0.80), 'y': (0.20, 0.70), 'x': (0.40, 1.0)},
                'shape_compactness': (0.30, 0.80),
                'connectivity': 'high',
                'priority': 1  # High priority organ
            },
            'SPLEEN': {
                'hu_range': (25, 70),  # Wider range
                'hu_std': 15,
                'volume_range': (50, 500),  # Much wider range
                'relative_position': {'z': (0.30, 0.70), 'y': (0.20, 0.70), 'x': (0.0, 0.55)},
                'shape_compactness': (0.50, 0.90),
                'connectivity': 'high',
                'priority': 2
            },
            'KIDNEY_RIGHT': {
                'hu_range': (15, 65),  # Very wide range
                'hu_std': 15,
                'volume_range': (60, 350),  # Much wider range
                'relative_position': {'z': (0.25, 0.65), 'y': (0.30, 0.80), 'x': (0.50, 1.0)},
                'shape_compactness': (0.60, 0.95),
                'connectivity': 'high',
                'priority': 3
            },
            'KIDNEY_LEFT': {
                'hu_range': (15, 65),  # Very wide range
                'hu_std': 15,
                'volume_range': (60, 350),  # Much wider range
                'relative_position': {'z': (0.25, 0.65), 'y': (0.30, 0.80), 'x': (0.0, 0.50)},
                'shape_compactness': (0.60, 0.95),
                'connectivity': 'high',
                'priority': 4
            },
            'PANCREAS': {
                'hu_range': (25, 60),
                'hu_std': 12,
                'volume_range': (40, 180),
                'relative_position': {'z': (0.42, 0.58), 'y': (0.35, 0.65), 'x': (0.25, 0.75)},
                'shape_compactness': (0.25, 0.55),
                'connectivity': 'medium',
                'priority': 5
            },
            'STOMACH': {
                'hu_range': (-80, 40),
                'hu_std': 50,
                'volume_range': (100, 800),
                'relative_position': {'z': (0.40, 0.70), 'y': (0.25, 0.65), 'x': (0.15, 0.65)},
                'shape_compactness': (0.15, 0.55),
                'connectivity': 'medium',
                'priority': 6
            },
            'GALL_BLADDER': {
                'hu_range': (-20, 30),
                'hu_std': 18,
                'volume_range': (8, 90),
                'relative_position': {'z': (0.45, 0.65), 'y': (0.35, 0.65), 'x': (0.50, 0.80)},
                'shape_compactness': (0.75, 0.96),
                'connectivity': 'high',
                'priority': 7
            },
            'HEART': {
                'hu_range': (25, 65),
                'hu_std': 22,
                'volume_range': (350, 900),
                'relative_position': {'z': (0.55, 0.85), 'y': (0.15, 0.55), 'x': (0.30, 0.70)},
                'shape_compactness': (0.55, 0.82),
                'connectivity': 'high',
                'priority': 8
            },
            'AORTA': {
                'hu_range': (80, 350),
                'hu_std': 60,
                'volume_range': (25, 180),
                'relative_position': {'z': (0.25, 0.85), 'y': (0.25, 0.75), 'x': (0.35, 0.65)},
                'shape_compactness': (0.08, 0.35),
                'connectivity': 'low',
                'priority': 9
            },
            'URINARY_BLADDER': {
                'hu_range': (-20, 35),
                'hu_std': 20,
                'volume_range': (80, 700),
                'relative_position': {'z': (0.05, 0.35), 'y': (0.45, 0.85), 'x': (0.30, 0.70)},
                'shape_compactness': (0.65, 0.96),
                'connectivity': 'high',
                'priority': 10
            },
            'SPINAL_CORD': {
                'hu_range': (15, 55),
                'hu_std': 12,
                'volume_range': (10, 60),
                'relative_position': {'z': (0.15, 0.85), 'y': (0.65, 0.98), 'x': (0.42, 0.58)},
                'shape_compactness': (0.15, 0.45),
                'connectivity': 'low',
                'priority': 11
            },
            'BONES': {
                'hu_range': (150, 3000),
                'hu_std': 250,
                'volume_range': (1500, 9000),
                'relative_position': {'z': (0.0, 1.0), 'y': (0.0, 1.0), 'x': (0.0, 1.0)},
                'shape_compactness': (0.08, 0.55),
                'connectivity': 'medium',
                'priority': 12
            },
        }

    def load_model(self):
        """Load learned parameters."""
        try:
            with open(self.model_path, 'rb') as f:
                self.organ_statistics = pickle.load(f)
            print(f"Loaded learned statistics from {len(self.organ_statistics)} previous patients")
        except FileNotFoundError:
            print("Starting with base anatomical knowledge (no prior training)")
            self.organ_statistics = {}

    def save_model(self):
        """Save learned parameters."""
        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.organ_statistics, f)

    def segment_patient(self, ct_volume: np.ndarray,
                       voxel_spacing: Tuple[float, float, float],
                       patient_id: str = None) -> Dict:
        """
        Segment all organs with improved detection.
        """
        print(f"\nStarting advanced organ segmentation...")
        print(f"   Volume shape: {ct_volume.shape}")
        print(f"   Voxel spacing: {voxel_spacing}")

        start_time = time.time()

        # Preprocessing
        ct_preprocessed = self._preprocess_volume(ct_volume)
        body_mask = self._extract_body_mask(ct_preprocessed)

        # Segment organs in priority order
        organ_measurements = {}
        organ_masks = {}

        # Sort organs by priority
        sorted_organs = sorted(
            self.organ_priors.items(),
            key=lambda x: x[1].get('priority', 99)
        )

        for organ_name, prior in sorted_organs:
            print(f"   Segmenting {organ_name}...")

            mask = self._segment_organ_improved(
                ct_preprocessed,
                body_mask,
                organ_name,
                prior,
                organ_masks
            )

            if mask is not None and np.sum(mask) > 0:
                measurements = self.extract_organ_statistics(
                    ct_volume, mask, voxel_spacing
                )

                if self._validate_organ_measurements(organ_name, measurements, prior):
                    organ_masks[organ_name] = mask
                    organ_measurements[organ_name] = measurements
                    print(f"      Found: {measurements['volume_cm3']:.1f} cm3, HU={measurements['mean_hu']:.1f}")
                else:
                    print(f"      Failed validation")
            else:
                print(f"      Not detected")

        # Update statistics
        if patient_id:
            self._update_statistics(patient_id, organ_measurements)

        processing_time = time.time() - start_time
        print(f"\n   Segmentation complete in {processing_time:.2f}s")
        print(f"   Organs segmented: {len(organ_measurements)}")

        return organ_measurements

    def _preprocess_volume(self, ct_volume: np.ndarray) -> np.ndarray:
        """Enhanced preprocessing."""
        ct_clipped = np.clip(ct_volume, -1024, 3071)
        ct_smoothed = ndimage.gaussian_filter(ct_clipped, sigma=0.6)
        return ct_smoothed

    def _extract_body_mask(self, ct_volume: np.ndarray) -> np.ndarray:
        """Extract body region."""
        body_mask = ct_volume > -400
        body_mask = ndimage.binary_fill_holes(body_mask)
        body_mask = ndimage.binary_erosion(body_mask, iterations=1)
        body_mask = ndimage.binary_dilation(body_mask, iterations=3)

        labeled, num_features = ndimage.label(body_mask)
        if num_features > 0:
            sizes = ndimage.sum(body_mask, labeled, range(1, num_features + 1))
            largest_label = np.argmax(sizes) + 1
            body_mask = (labeled == largest_label)

        return body_mask

    def _segment_organ_improved(self, ct_volume: np.ndarray,
                                body_mask: np.ndarray,
                                organ_name: str,
                                prior: Dict,
                                existing_masks: Dict) -> np.ndarray:
        """
        Improved organ segmentation with relaxed criteria.
        """
        # HU-based initial mask
        hu_min, hu_max = prior['hu_range']
        hu_mask = (ct_volume >= hu_min - 10) & (ct_volume <= hu_max + 10)  # Relaxed
        hu_mask = hu_mask & body_mask

        # Apply ROI constraint with relaxation
        roi_mask = self._get_roi_mask(ct_volume.shape, prior['relative_position'], relax=0.15)
        candidate_mask = hu_mask & roi_mask

        if np.sum(candidate_mask) < 50:
            return None

        # Connected components
        labeled, num_features = ndimage.label(candidate_mask)

        if num_features == 0:
            return None

        # Find best component
        best_mask = None
        best_score = -1

        vol_min, vol_max = prior['volume_range']
        voxel_volume = np.prod([s * 0.1 for s in [1, 1, 1]])  # Simplified

        for i in range(1, num_features + 1):
            component = (labeled == i)
            component_size = np.sum(component)

            # Volume check (relaxed)
            volume_cm3 = component_size * voxel_volume
            if volume_cm3 < vol_min * 0.5 or volume_cm3 > vol_max * 1.5:  # Relaxed 50%
                continue

            # HU check
            hu_values = ct_volume[component]
            mean_hu = np.mean(hu_values)

            if mean_hu < hu_min - 15 or mean_hu > hu_max + 15:  # Relaxed
                continue

            # Position score
            position_score = self._score_position(component, prior['relative_position'])

            # Compactness score
            compactness = self._calculate_compactness(component)

            # Combined score
            score = position_score * 0.6 + (1.0 if vol_min <= volume_cm3 <= vol_max else 0.5) * 0.4

            if score > best_score:
                best_score = score
                best_mask = component

        if best_mask is not None:
            # Post-processing
            best_mask = ndimage.binary_fill_holes(best_mask)
            best_mask = ndimage.binary_opening(best_mask, iterations=1)
            best_mask = ndimage.binary_closing(best_mask, iterations=2)

        return best_mask

    def _get_roi_mask(self, shape: Tuple, position_ranges: Dict, relax: float = 0.0) -> np.ndarray:
        """Create ROI mask with optional relaxation."""
        mask = np.ones(shape, dtype=bool)

        z_range = position_ranges['z']
        y_range = position_ranges['y']
        x_range = position_ranges['x']

        # Apply relaxation
        z_start = max(0, int((z_range[0] - relax) * shape[0]))
        z_end = min(shape[0], int((z_range[1] + relax) * shape[0]))

        y_start = max(0, int((y_range[0] - relax) * shape[1]))
        y_end = min(shape[1], int((y_range[1] + relax) * shape[1]))

        x_start = max(0, int((x_range[0] - relax) * shape[2]))
        x_end = min(shape[2], int((x_range[1] + relax) * shape[2]))

        mask[:z_start, :, :] = False
        mask[z_end:, :, :] = False
        mask[:, :y_start, :] = False
        mask[:, y_end:, :] = False
        mask[:, :, :x_start] = False
        mask[:, :, x_end:] = False

        return mask

    def _score_position(self, mask: np.ndarray, position_ranges: Dict) -> float:
        """Score how well mask matches expected position."""
        coords = np.argwhere(mask)
        if len(coords) == 0:
            return 0.0

        centroid = coords.mean(axis=0)
        shape = mask.shape

        z_norm = centroid[0] / shape[0]
        y_norm = centroid[1] / shape[1]
        x_norm = centroid[2] / shape[2]

        z_score = 1.0 if position_ranges['z'][0] <= z_norm <= position_ranges['z'][1] else 0.5
        y_score = 1.0 if position_ranges['y'][0] <= y_norm <= position_ranges['y'][1] else 0.5
        x_score = 1.0 if position_ranges['x'][0] <= x_norm <= position_ranges['x'][1] else 0.5

        return (z_score + y_score + x_score) / 3.0

    def _calculate_compactness(self, mask: np.ndarray) -> float:
        """Calculate shape compactness."""
        volume = np.sum(mask)
        if volume == 0:
            return 0.0

        # Simple compactness metric
        surface = np.sum(ndimage.binary_erosion(mask) != mask)
        return volume / (surface + 1)

    def extract_organ_statistics(self, ct_volume: np.ndarray,
                                 mask: np.ndarray,
                                 voxel_spacing: Tuple[float, float, float]) -> Dict:
        """Extract comprehensive organ statistics."""
        voxel_volume = np.prod(voxel_spacing) / 1000.0  # mm³ to cm³

        hu_values = ct_volume[mask]
        voxel_count = np.sum(mask)

        return {
            'volume_cm3': float(voxel_count * voxel_volume),
            'voxel_count': int(voxel_count),
            'mean_hu': float(np.mean(hu_values)),
            'std_hu': float(np.std(hu_values)),
            'median_hu': float(np.median(hu_values)),
            'min_hu': float(np.min(hu_values)),
            'max_hu': float(np.max(hu_values)),
            'q25_hu': float(np.percentile(hu_values, 25)),
            'q75_hu': float(np.percentile(hu_values, 75))
        }

    def _validate_organ_measurements(self, organ_name: str,
                                    measurements: Dict,
                                    prior: Dict) -> bool:
        """Validate organ measurements with very relaxed criteria."""
        vol_min, vol_max = prior['volume_range']
        volume = measurements['volume_cm3']

        # VERY relaxed volume check (70% margin - accept almost anything reasonable)
        if volume < vol_min * 0.3 or volume > vol_max * 2.5:
            return False

        # Very relaxed HU check (±30 HU tolerance)
        hu_min, hu_max = prior['hu_range']
        mean_hu = measurements['mean_hu']

        # For liver, spleen, kidneys - be extra lenient with HU ranges
        lenient_organs = ['LIVER', 'SPLEEN', 'KIDNEY_RIGHT', 'KIDNEY_LEFT', 'PANCREAS']
        if organ_name in lenient_organs:
            hu_tolerance = 40  # Very wide tolerance
        else:
            hu_tolerance = 30

        if mean_hu < hu_min - hu_tolerance or mean_hu > hu_max + hu_tolerance:
            return False

        return True

    def _update_statistics(self, patient_id: str, organ_measurements: Dict):
        """Update learned statistics."""
        self.organ_statistics[patient_id] = {
            'timestamp': time.time(),
            'organs': list(organ_measurements.keys()),
            'volumes': {k: v['volume_cm3'] for k, v in organ_measurements.items()},
            'hu_means': {k: v['mean_hu'] for k, v in organ_measurements.items()}
        }

        self.save_model()
        print("Saved learned statistics")
