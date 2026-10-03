"""
True AI Dose Model - NO Hardcoded Values

Trains purely on relationships between:
- Patient anatomy (organ volumes, tissue density)
- Scan parameters (kVp, mAs, exposure time)
- Physics principles (via loss constraints, not fixed values)

The model learns dose patterns from data, not from lookup tables.

Key Innovation: Uses ICRP labels ONLY for initial training,
then learns to predict from scan parameters + anatomy alone.
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import json
import sys

sys.path.append('.')
from src.physics_loss import PhysicsInformedLoss


class TrueAIDoseModel(nn.Module):
    """
    Deep neural network that learns dose from features.
    NO hardcoded ratios or lookup tables.
    """

    def __init__(self, n_features, n_organs):
        super().__init__()

        # Organ embedding - learns organ-specific sensitivity
        self.organ_embedding = nn.Embedding(n_organs, 32)

        # Main network - learns dose from physics + anatomy
        input_dim = n_features + 32

        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 1)
        )

    def forward(self, features, organ_idx):
        """Predict dose from features + organ type."""
        organ_emb = self.organ_embedding(organ_idx)
        x = torch.cat([features, organ_emb], dim=1)
        dose = self.network(x)
        return dose.squeeze(-1)


class TrueAIDosePredictor:
    """Complete AI dose estimation system - no hardcoding."""

    def __init__(self, device='cpu'):
        self.device = device
        self.model = None
        self.organ_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_cols = []
        self._is_fitted = False

    def prepare_features(self, df):
        """Extract features WITHOUT using any dose ratios."""
        features = []

        # Anatomy features
        if 'volume_cm3' in df.columns:
            features.append('volume_cm3')
        if 'mean_hu' in df.columns:
            features.append('mean_hu')

        # Scan parameters (these determine dose via physics)
        if 'mean_kvp' in df.columns:
            features.append('mean_kvp')
        if 'mean_tube_current_mA' in df.columns:
            features.append('mean_tube_current_mA')
        if 'mean_exposure_mAs' in df.columns:
            features.append('mean_exposure_mAs')
        if 'mean_exposure_time_ms' in df.columns:
            features.append('mean_exposure_time_ms')
        if 'pitch_factor' in df.columns:
            features.append('pitch_factor')
        if 'mean_slice_thickness_mm' in df.columns:
            features.append('mean_slice_thickness_mm')
        if 'scan_length_cm' in df.columns:
            features.append('scan_length_cm')

        # Phase
        df['phase_venous'] = (df['PHASE'] == 'venous').astype(int)
        features.append('phase_venous')

        self.feature_cols = features
        X = df[features].values

        # Organ encoding
        organ_idx = self.organ_encoder.transform(df['organ'])

        # Patient groups for CV
        patient_groups = df['UID'].values

        return X, organ_idx, patient_groups

    def train(self, organ_features_path, dicom_params_path, labels_path,
              epochs=200, batch_size=32, lr=0.001, use_physics_loss=True):
        """
        Train AI model on REAL physics-based labels.

        Args:
            organ_features_path: Real organ features
            dicom_params_path: Real scan parameters
            labels_path: Physics-based labels (ICRP or real measurements)
            epochs: Training epochs
            batch_size: Batch size
            lr: Learning rate
            use_physics_loss: Add physics constraints
        """
        print("="*70)
        print("TRUE AI DOSE ESTIMATION - NO HARDCODED VALUES")
        print("="*70)

        # Load data
        organ_df = pd.read_csv(organ_features_path, dtype={'UID': str})
        dicom_df = pd.read_csv(dicom_params_path, dtype={'UID': str})
        labels_df = pd.read_csv(labels_path, dtype={'UID': str})

        print(f"\nData loaded:")
        print(f"  Organ features: {len(organ_df)} rows")
        print(f"  DICOM params: {len(dicom_df)} rows")
        print(f"  Dose labels: {len(labels_df)} rows")

        # Merge
        df = organ_df.merge(dicom_df, on=['UID', 'PHASE'], how='inner')
        df = df.merge(labels_df[['UID', 'PHASE', 'organ', 'icrp_dose_mGy']],
                     on=['UID', 'PHASE', 'organ'], how='inner')

        df = df.dropna(subset=['volume_cm3', 'mean_hu', 'mean_kvp', 'icrp_dose_mGy'])

        print(f"\nMerged dataset: {len(df)} samples from {df['UID'].nunique()} patients")

        if len(df) < 100:
            raise ValueError(f"Insufficient training data: {len(df)} samples")

        # Fit encoders
        self.organ_encoder.fit(df['organ'])
        n_organs = len(self.organ_encoder.classes_)
        print(f"\nOrgans: {n_organs} types")
        print(f"  {list(self.organ_encoder.classes_)}")

        # Target labels
        y = df['icrp_dose_mGy'].values
        print(f"\nDose range: {y.min():.2f} - {y.max():.2f} mGy")
        print(f"Mean dose: {y.mean():.2f} mGy")

        # Prepare features
        X, organ_idx, patient_groups = self.prepare_features(df)
        print(f"\nFeatures ({X.shape[1]}): {self.feature_cols}")

        # Scale features
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        # Initialize model
        n_features = X.shape[1]
        self.model = TrueAIDoseModel(n_features, n_organs).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"\nModel: {total_params:,} parameters")

        # Physics-informed loss
        if use_physics_loss:
            physics_config = {
                'weight': 0.1,
                'constraints': {
                    'non_negativity': {'enabled': True, 'weight': 0.1},
                    'hu_consistency': {'enabled': True, 'weight': 0.05},
                    'energy_conservation': {'enabled': True, 'weight': 0.02}
                }
            }
            criterion = PhysicsInformedLoss(physics_config)
            print("\n✓ Physics-informed loss enabled")
        else:
            criterion = nn.MSELoss()

        # Cross-validation
        print("\n" + "="*70)
        print("5-FOLD PATIENT-WISE CROSS-VALIDATION")
        print("="*70)

        cv = GroupKFold(n_splits=5)
        oof_preds = np.zeros(len(y))

        for fold, (train_idx, val_idx) in enumerate(cv.split(X_scaled, y, groups=patient_groups), 1):
            print(f"\nFold {fold}/5:")

            # Data
            X_train = torch.FloatTensor(X_scaled[train_idx]).to(self.device)
            y_train = torch.FloatTensor(y[train_idx]).to(self.device)
            organ_train = torch.LongTensor(organ_idx[train_idx]).to(self.device)

            X_val = torch.FloatTensor(X_scaled[val_idx]).to(self.device)
            y_val = torch.FloatTensor(y[val_idx]).to(self.device)
            organ_val = torch.LongTensor(organ_idx[val_idx]).to(self.device)

            # Model for this fold
            fold_model = TrueAIDoseModel(n_features, n_organs).to(self.device)
            optimizer = torch.optim.Adam(fold_model.parameters(), lr=lr, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

            best_val_loss = float('inf')
            patience = 0
            max_patience = 20

            for epoch in range(epochs):
                fold_model.train()
                perm = torch.randperm(len(X_train))

                epoch_loss = 0
                for i in range(0, len(X_train), batch_size):
                    batch_idx = perm[i:i+batch_size]
                    X_batch = X_train[batch_idx]
                    y_batch = y_train[batch_idx]
                    organ_batch = organ_train[batch_idx]

                    optimizer.zero_grad()
                    pred = fold_model(X_batch, organ_batch)

                    if use_physics_loss:
                        loss, loss_dict = criterion(pred, y_batch, X_batch, organ_batch)
                    else:
                        loss = criterion(pred, y_batch)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(fold_model.parameters(), 1.0)
                    optimizer.step()

                    epoch_loss += loss.item()

                # Validation
                fold_model.eval()
                with torch.no_grad():
                    val_pred = fold_model(X_val, organ_val).cpu().numpy()
                    val_loss = mean_absolute_error(y_val.cpu().numpy(), val_pred)

                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience = 0
                else:
                    patience += 1

                if (epoch + 1) % 25 == 0:
                    print(f"  Epoch {epoch+1:3d}: Val MAE = {val_loss:.3f}")

                if patience >= max_patience:
                    print(f"  Early stop at epoch {epoch+1}")
                    break

            # OOF predictions
            fold_model.eval()
            with torch.no_grad():
                oof_preds[val_idx] = fold_model(X_val, organ_val).cpu().numpy()

        # Metrics
        mae = mean_absolute_error(y, oof_preds)
        rmse = np.sqrt(np.mean((y - oof_preds) ** 2))
        r2 = r2_score(y, oof_preds)

        print("\n" + "="*70)
        print("CROSS-VALIDATION RESULTS")
        print("="*70)
        print(f"  MAE:  {mae:.3f} mGy")
        print(f"  RMSE: {rmse:.3f} mGy")
        print(f"  R²:   {r2:.3f}")

        # Per-organ breakdown
        print("\nPer-Organ Performance:")
        for organ in sorted(df['organ'].unique()):
            mask = df['organ'] == organ
            organ_mae = mean_absolute_error(y[mask], oof_preds[mask])
            organ_r2 = r2_score(y[mask], oof_preds[mask])
            print(f"  {organ:20} MAE: {organ_mae:6.2f}  R²: {organ_r2:.3f}  (n={mask.sum()})")

        # Retrain on full data
        print("\nRetraining on full dataset...")
        X_all = torch.FloatTensor(X_scaled).to(self.device)
        y_all = torch.FloatTensor(y).to(self.device)
        organ_all = torch.LongTensor(organ_idx).to(self.device)

        self.model = TrueAIDoseModel(n_features, n_organs).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)

        for epoch in range(epochs):
            self.model.train()
            perm = torch.randperm(len(X_all))

            for i in range(0, len(X_all), batch_size):
                batch_idx = perm[i:i+batch_size]
                X_batch = X_all[batch_idx]
                y_batch = y_all[batch_idx]
                organ_batch = organ_all[batch_idx]

                optimizer.zero_grad()
                pred = self.model(X_batch, organ_batch)

                if use_physics_loss:
                    loss, _ = criterion(pred, y_batch, X_batch, organ_batch)
                else:
                    loss = criterion(pred, y_batch)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

        self._is_fitted = True

        print("\n✓ Training complete!")

        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'n_samples': len(df),
            'n_patients': df['UID'].nunique(),
            'n_organs': n_organs
        }

    def predict(self, organ_features_df, dicom_params_df):
        """Predict doses for new patients."""
        if not self._is_fitted:
            raise RuntimeError("Model not trained!")

        df = organ_features_df.merge(dicom_params_df, on=['UID', 'PHASE'], how='inner')
        df['phase_venous'] = (df['PHASE'] == 'venous').astype(int)

        X = df[self.feature_cols].fillna(df[self.feature_cols].mean()).values
        X_scaled = self.scaler.transform(X)
        organ_idx = self.organ_encoder.transform(df['organ'])

        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        organ_tensor = torch.LongTensor(organ_idx).to(self.device)

        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_tensor, organ_tensor).cpu().numpy()

        return preds

    def save(self, save_dir):
        """Save model."""
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), Path(save_dir) / 'model.pt')
        pickle.dump(self.organ_encoder, open(Path(save_dir) / 'organ_encoder.pkl', 'wb'))
        pickle.dump(self.scaler, open(Path(save_dir) / 'scaler.pkl', 'wb'))

        meta = {
            'feature_cols': self.feature_cols,
            'organs': list(self.organ_encoder.classes_),
            'n_features': len(self.feature_cols),
            'n_organs': len(self.organ_encoder.classes_)
        }
        json.dump(meta, open(Path(save_dir) / 'meta.json', 'w'), indent=2)

        print(f"\n✓ Model saved: {save_dir}/")

    @classmethod
    def load(cls, save_dir, device='cpu'):
        """Load model."""
        with open(Path(save_dir) / 'meta.json') as f:
            meta = json.load(f)

        obj = cls(device=device)
        obj.feature_cols = meta['feature_cols']
        obj.organ_encoder = pickle.load(open(Path(save_dir) / 'organ_encoder.pkl', 'rb'))
        obj.scaler = pickle.load(open(Path(save_dir) / 'scaler.pkl', 'rb'))

        obj.model = TrueAIDoseModel(meta['n_features'], meta['n_organs']).to(device)
        obj.model.load_state_dict(torch.load(Path(save_dir) / 'model.pt', map_location=device))
        obj.model.eval()
        obj._is_fitted = True

        print(f"✓ Model loaded: {save_dir}/")
        return obj


if __name__ == "__main__":
    print("True AI Dose Model - NO Hardcoded Values")
