"""
models/ai_dose_predictor.py

Complete AI-driven dose estimation system that combines:
  1. CNN visual features from CT images (learned patient anatomy)
  2. Organ-level geometric features (volume, mean HU)
  3. Multi-task neural network that predicts organ-specific doses

This is a TRUE AI approach - no hardcoded ratios, learns dose patterns
from data end-to-end.
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import json
import os


class DoseEstimationNetwork(nn.Module):
    """
    Multi-layer perceptron that predicts organ dose from:
      - CNN visual features (128-dim)
      - Organ geometric features (volume, HU, organ type)
      - Scan parameters (CTDIvol)
    """

    def __init__(self, cnn_dim=128, n_organs=15, hidden_dims=[256, 128, 64]):
        super().__init__()

        # Input: [cnn_features(128) + organ_encoded(1) + volume(1) + mean_hu(1) + ctdivol(1)] = 132
        input_dim = cnn_dim + 4

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.3),
            ])
            prev_dim = hidden_dim

        # Output layer: single dose value
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, input_dim)
        Returns:
            dose: Tensor of shape (batch_size, 1)
        """
        return self.network(x).squeeze(-1)


class AIDosePredictor:
    """
    Complete AI dose estimation system.
    """

    def __init__(self, cnn_dim=128, device='cpu'):
        self.cnn_dim = cnn_dim
        self.device = device

        self.organ_encoder = LabelEncoder()
        self.scaler = StandardScaler()

        self.model = None
        self.feature_cols = ["organ_encoded", "volume_cm3", "mean_hu", "mean_ctdivol_mGy"]
        self.cnn_cols = [f"cnn_{i}" for i in range(cnn_dim)]

        self._is_fitted = False

    def _prepare_features(self, df, cnn_embeddings=None):
        """
        Combine organ features + CNN embeddings into model input.

        Args:
            df: DataFrame with organ_encoded, volume_cm3, mean_hu, mean_ctdivol_mGy
            cnn_embeddings: dict mapping (UID, PHASE) -> 128-dim numpy array

        Returns:
            X: numpy array of shape (n_samples, 132)
        """
        # Get tabular features
        X_tabular = df[self.feature_cols].values

        # Get CNN features
        if cnn_embeddings is not None:
            X_cnn = np.array([
                cnn_embeddings.get((row['UID'], row['PHASE']), np.zeros(self.cnn_dim))
                for _, row in df.iterrows()
            ])
        else:
            X_cnn = np.zeros((len(df), self.cnn_dim))

        # Concatenate
        X = np.concatenate([X_cnn, X_tabular], axis=1)
        return X

    def fit(self, df, cnn_embeddings=None, epochs=50, batch_size=64, lr=0.001):
        """
        Train the AI dose predictor.

        Args:
            df: organ_dose_labels.csv with pseudo_label_dose_mGy
            cnn_embeddings: dict mapping (UID, PHASE) -> 128-dim embedding (optional)
            epochs: training epochs
            batch_size: batch size
            lr: learning rate
        """
        df = df.dropna(subset=["volume_cm3", "mean_hu", "mean_ctdivol_mGy", "pseudo_label_dose_mGy"])

        self.organ_encoder.fit(df["organ"])
        df["organ_encoded"] = self.organ_encoder.transform(df["organ"])

        n_organs = len(self.organ_encoder.classes_)
        self.model = DoseEstimationNetwork(
            cnn_dim=self.cnn_dim,
            n_organs=n_organs,
        ).to(self.device)

        print(f"Training AI dose predictor on {len(df)} organ-rows from {df['UID'].nunique()} patients.")
        print(f"Using CNN features: {cnn_embeddings is not None}")

        # Prepare features
        X = self._prepare_features(df, cnn_embeddings)
        y = df["pseudo_label_dose_mGy"].values

        # Fit scaler
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        # 5-fold CV training
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        oof_preds = np.zeros(len(y))

        for fold, (tr_idx, val_idx) in enumerate(cv.split(X_scaled), 1):
            print(f"\nFold {fold}/5:")

            X_tr = torch.FloatTensor(X_scaled[tr_idx]).to(self.device)
            y_tr = torch.FloatTensor(y[tr_idx]).to(self.device)
            X_val = torch.FloatTensor(X_scaled[val_idx]).to(self.device)
            y_val = torch.FloatTensor(y[val_idx]).to(self.device)

            fold_model = DoseEstimationNetwork(cnn_dim=self.cnn_dim, n_organs=n_organs).to(self.device)
            optimizer = torch.optim.Adam(fold_model.parameters(), lr=lr)
            criterion = nn.MSELoss()

            # Training loop
            for epoch in range(epochs):
                fold_model.train()
                perm = torch.randperm(len(X_tr))

                for i in range(0, len(X_tr), batch_size):
                    batch_idx = perm[i:i+batch_size]
                    X_batch = X_tr[batch_idx]
                    y_batch = y_tr[batch_idx]

                    optimizer.zero_grad()
                    pred = fold_model(X_batch)
                    loss = criterion(pred, y_batch)
                    loss.backward()
                    optimizer.step()

                if (epoch + 1) % 10 == 0:
                    fold_model.eval()
                    with torch.no_grad():
                        val_pred = fold_model(X_val).cpu().numpy()
                        val_mae = mean_absolute_error(y_val.cpu().numpy(), val_pred)
                        print(f"  Epoch {epoch+1}/{epochs} - Val MAE: {val_mae:.3f} mGy")

            # Get OOF predictions
            fold_model.eval()
            with torch.no_grad():
                oof_preds[val_idx] = fold_model(X_val).cpu().numpy()

        # Calculate OOF metrics
        mae = mean_absolute_error(y, oof_preds)
        r2 = r2_score(y, oof_preds)

        print(f"\n{'='*70}")
        print(f"AI Dose Predictor OOF Performance:")
        print(f"  MAE: {mae:.2f} mGy")
        print(f"  R²:  {r2:.3f}")
        print(f"{'='*70}")

        # Retrain on full data for deployment
        X_all = torch.FloatTensor(X_scaled).to(self.device)
        y_all = torch.FloatTensor(y).to(self.device)

        self.model = DoseEstimationNetwork(cnn_dim=self.cnn_dim, n_organs=n_organs).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        print("\nRetraining on full dataset...")
        for epoch in range(epochs):
            self.model.train()
            perm = torch.randperm(len(X_all))

            for i in range(0, len(X_all), batch_size):
                batch_idx = perm[i:i+batch_size]
                X_batch = X_all[batch_idx]
                y_batch = y_all[batch_idx]

                optimizer.zero_grad()
                pred = self.model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()

        self._is_fitted = True

        return {'mae': mae, 'r2': r2}

    def predict(self, df, cnn_embeddings=None):
        """
        Predict organ doses.

        Args:
            df: DataFrame with organ_encoded, volume_cm3, mean_hu, mean_ctdivol_mGy
            cnn_embeddings: dict mapping (UID, PHASE) -> 128-dim embedding (optional)

        Returns:
            predictions: numpy array of doses
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() or load() first.")

        df = df.copy()
        df["organ_encoded"] = self.organ_encoder.transform(df["organ"])

        X = self._prepare_features(df, cnn_embeddings)
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_tensor).cpu().numpy()

        return preds

    def save(self, directory):
        """Save all model components"""
        os.makedirs(directory, exist_ok=True)

        torch.save(self.model.state_dict(), os.path.join(directory, "neural_net.pt"))
        pickle.dump(self.organ_encoder, open(os.path.join(directory, "encoder.pkl"), "wb"))
        pickle.dump(self.scaler, open(os.path.join(directory, "scaler.pkl"), "wb"))

        with open(os.path.join(directory, "meta.json"), "w") as f:
            json.dump({
                "cnn_dim": self.cnn_dim,
                "feature_cols": self.feature_cols,
                "organs": list(self.organ_encoder.classes_),
            }, f, indent=2)

        print(f"AI dose predictor saved to {directory}/")

    @classmethod
    def load(cls, directory, device='cpu'):
        """Load saved model"""
        with open(os.path.join(directory, "meta.json")) as f:
            meta = json.load(f)

        obj = cls(cnn_dim=meta["cnn_dim"], device=device)
        obj.feature_cols = meta["feature_cols"]
        obj.organ_encoder = pickle.load(open(os.path.join(directory, "encoder.pkl"), "rb"))
        obj.scaler = pickle.load(open(os.path.join(directory, "scaler.pkl"), "rb"))

        n_organs = len(obj.organ_encoder.classes_)
        obj.model = DoseEstimationNetwork(cnn_dim=meta["cnn_dim"], n_organs=n_organs).to(device)
        obj.model.load_state_dict(torch.load(os.path.join(directory, "neural_net.pt"), map_location=device))
        obj.model.eval()

        obj._is_fitted = True
        print(f"AI dose predictor loaded from {directory}/")
        return obj
