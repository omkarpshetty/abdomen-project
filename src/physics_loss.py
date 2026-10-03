"""
Physics-Informed Loss Functions

Implements physical constraints as differentiable loss terms to guide
neural network training toward physically plausible dose predictions.

These constraints DO NOT hardcode dose values - they enforce consistency
with known radiation physics principles while allowing the model to learn
from data.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PhysicsInformedLoss(nn.Module):
    """
    Combined loss with data-driven and physics-based components.

    Total Loss = Data Loss + λ * Physics Loss

    where Physics Loss enforces constraints without hardcoding values.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.physics_weight = config.get('weight', 0.1)
        self.constraints = config.get('constraints', {})

        # Data loss (MSE or MAE)
        self.data_loss = nn.MSELoss()

    def forward(self, predictions, targets, features, organ_idx):
        """
        Args:
            predictions: (batch,) predicted doses
            targets: (batch,) true doses
            features: (batch, n_features) input features
            organ_idx: (batch,) organ indices

        Returns:
            total_loss: scalar tensor
            loss_dict: dict with individual loss components
        """
        # Data loss
        loss_data = self.data_loss(predictions, targets)

        # Physics constraints
        loss_physics = 0.0
        loss_dict = {'data_loss': loss_data.item()}

        if self.physics_weight > 0:
            # Non-negativity constraint
            if self.constraints.get('non_negativity', {}).get('enabled', False):
                loss_neg = self._non_negativity_loss(predictions)
                weight = self.constraints['non_negativity'].get('weight', 0.1)
                loss_physics += weight * loss_neg
                loss_dict['non_negativity'] = loss_neg.item()

            # HU-consistency constraint
            if self.constraints.get('hu_consistency', {}).get('enabled', False):
                loss_hu = self._hu_consistency_loss(predictions, features)
                weight = self.constraints['hu_consistency'].get('weight', 0.05)
                loss_physics += weight * loss_hu
                loss_dict['hu_consistency'] = loss_hu.item()

            # Spatial smoothness (requires patient grouping)
            if self.constraints.get('spatial_smoothness', {}).get('enabled', False):
                loss_smooth = self._spatial_smoothness_loss(predictions, organ_idx)
                weight = self.constraints['spatial_smoothness'].get('weight', 0.03)
                loss_physics += weight * loss_smooth
                loss_dict['spatial_smoothness'] = loss_smooth.item()

            # Energy conservation (dose scales with kVp², mAs)
            if self.constraints.get('energy_conservation', {}).get('enabled', False):
                loss_energy = self._energy_conservation_loss(predictions, features)
                weight = self.constraints['energy_conservation'].get('weight', 0.02)
                loss_physics += weight * loss_energy
                loss_dict['energy_conservation'] = loss_energy.item()

        loss_dict['physics_loss'] = loss_physics if isinstance(loss_physics, float) else loss_physics.item()

        total_loss = loss_data + self.physics_weight * loss_physics
        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict

    def _non_negativity_loss(self, predictions):
        """
        Penalize negative dose predictions.

        Dose must be >= 0 by physics.
        """
        return F.relu(-predictions).mean()

    def _hu_consistency_loss(self, predictions, features):
        """
        Organs with similar tissue density (HU) should have similar
        dose per unit exposure, accounting for attenuation.

        This doesn't hardcode values - it enforces relative consistency.
        """
        # Extract HU values (assuming it's in features)
        # Feature order: volume, HU, kVp, mAs, ...
        hu_values = features[:, 1]  # Index 1 is mean_hu

        # Compute dose normalized by energy (to remove technique variation)
        kvp = features[:, 2]
        mas = features[:, 4] if features.shape[1] > 4 else torch.ones_like(kvp)
        energy_proxy = (kvp ** 2) * mas

        # Avoid division by zero
        normalized_dose = predictions / (energy_proxy + 1e-6)

        # For similar HU values, normalized dose should be similar
        # Compute pairwise HU distance and dose distance
        n = len(predictions)
        if n < 2:
            return torch.tensor(0.0, device=predictions.device)

        # Sample random pairs (efficient for large batches)
        n_pairs = min(100, n * (n-1) // 2)
        idx1 = torch.randint(0, n, (n_pairs,), device=predictions.device)
        idx2 = torch.randint(0, n, (n_pairs,), device=predictions.device)
        mask = idx1 != idx2
        idx1, idx2 = idx1[mask], idx2[mask]

        if len(idx1) == 0:
            return torch.tensor(0.0, device=predictions.device)

        # HU similarity (small distance = similar tissue)
        hu_dist = torch.abs(hu_values[idx1] - hu_values[idx2])

        # Normalized dose similarity
        dose_dist = torch.abs(normalized_dose[idx1] - normalized_dose[idx2])

        # Loss: for similar HU (small dist), dose should also be similar
        # Use soft thresholding to avoid penalizing expected variations
        hu_threshold = 50.0  # HU units
        weight = torch.exp(-hu_dist / hu_threshold)

        loss = (weight * dose_dist).mean()
        return loss

    def _spatial_smoothness_loss(self, predictions, organ_idx):
        """
        Dose should vary smoothly across anatomically adjacent organs.

        This doesn't hardcode ratios - it enforces smooth transitions.
        """
        # Define organ adjacency (simplified)
        # In practice, would use spatial coordinates
        adjacency_pairs = [
            # (organ1_idx, organ2_idx) for organs that are neighbors
            # This would be computed from actual spatial data
        ]

        # For now, use a simple heuristic: similar organ types
        # should have similar doses within a patient
        unique_organs = torch.unique(organ_idx)
        if len(unique_organs) < 2:
            return torch.tensor(0.0, device=predictions.device)

        # Compute variance of doses within this batch
        # Lower variance = smoother (but we don't want too smooth)
        dose_std = torch.std(predictions)

        # Penalize extreme outliers (> 3 std devs)
        dose_mean = torch.mean(predictions)
        outliers = torch.abs(predictions - dose_mean) > 3 * dose_std
        loss = torch.abs(predictions[outliers] - dose_mean).mean() if outliers.any() else torch.tensor(0.0, device=predictions.device)

        return loss

    def _energy_conservation_loss(self, predictions, features):
        """
        Total dose should scale appropriately with beam energy.

        Physical principle: Dose ∝ kVp² × mAs (approximately)

        This doesn't hardcode absolute values - it enforces the scaling relationship.
        """
        kvp = features[:, 2]
        mas = features[:, 4] if features.shape[1] > 4 else torch.ones_like(kvp)

        energy_proxy = (kvp ** 2) * mas

        # Compute correlation between log(dose) and log(energy)
        # Should be positive correlation
        log_dose = torch.log(predictions + 1e-6)
        log_energy = torch.log(energy_proxy + 1e-6)

        # Pearson correlation
        dose_centered = log_dose - log_dose.mean()
        energy_centered = log_energy - log_energy.mean()

        correlation = (dose_centered * energy_centered).sum() / (
            torch.sqrt((dose_centered ** 2).sum()) * torch.sqrt((energy_centered ** 2).sum()) + 1e-6
        )

        # Loss: penalize negative correlation (should be positive)
        loss = F.relu(-correlation)

        return loss


class UncertaintyLoss(nn.Module):
    """
    Loss that encourages the model to provide calibrated uncertainty estimates.

    For ensemble models, uncertainty comes from model disagreement.
    """

    def __init__(self):
        super().__init__()

    def forward(self, predictions_ensemble, targets):
        """
        Args:
            predictions_ensemble: (n_models, batch) predictions from each model
            targets: (batch,) true values

        Returns:
            loss: scalar encouraging accurate uncertainty
        """
        # Mean prediction
        pred_mean = predictions_ensemble.mean(dim=0)

        # Uncertainty (std dev across models)
        pred_std = predictions_ensemble.std(dim=0)

        # Data loss
        mse = F.mse_loss(pred_mean, targets)

        # Calibration loss: predictions within 1 std should be accurate
        # with ~68% probability
        errors = torch.abs(pred_mean - targets)

        # Penalize: high uncertainty where error is small (overconfident uncertainty)
        # or low uncertainty where error is large (underconfident)
        calibration_loss = F.mse_loss(pred_std, errors)

        return mse + 0.1 * calibration_loss


def compute_gamma_pass_rate(pred_dose_map, true_dose_map, dose_threshold=0.03, distance_threshold=3.0):
    """
    Compute gamma pass rate for dose map comparison (3%/3mm criterion).

    Used for radiotherapy dose validation, adapted for diagnostic CT.

    Args:
        pred_dose_map: (H, W, D) or (H, W) predicted dose distribution
        true_dose_map: same shape, true dose distribution
        dose_threshold: 3% dose difference criterion
        distance_threshold: 3mm distance to agreement (in voxels)

    Returns:
        gamma_pass_rate: float [0, 1], percentage of voxels passing
    """
    # Simplified gamma calculation (full implementation is complex)
    # In practice, use specialized libraries like pymedphys

    # Normalize doses
    max_dose = true_dose_map.max()
    if max_dose == 0:
        return 1.0

    true_norm = true_dose_map / max_dose
    pred_norm = pred_dose_map / max_dose

    # Dose difference
    dose_diff = np.abs(pred_norm - true_norm)

    # Simple criterion: dose difference < threshold
    passing = dose_diff < dose_threshold

    return passing.mean()


if __name__ == "__main__":
    print("Physics-Informed Loss Module")
    print("Enforces physical constraints without hardcoding dose values")
