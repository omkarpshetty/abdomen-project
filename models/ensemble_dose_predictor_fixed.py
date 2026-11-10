"""
Fixed Ensemble AI Dose Prediction Model
"""
import numpy as np
import pandas as pd
import pickle
import json
import os
from pathlib import Path
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Traditional ML
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# PyTorch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class DoseNeuralNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128, 64]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


class EnsembleDosePredictor:
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.rf_model = None
        self.xgb_model = None
        self.nn_model = None
        self.scaler = StandardScaler()
        self.organ_encoder = LabelEncoder()

        self.weights = {
            'rf': 0.25,
            'xgb': 0.35,
            'nn': 0.40
        }

        self.feature_columns = [
            'organ_encoded',
            'volume_cm3',
            'mean_hu',
            'median_hu',
            'std_hu',
            'distance_from_isocenter_mm',
            'compactness',
            'kvp',
            'ctdivol_mgy',
            'num_slices',
            'slice_thickness_mm'
        ]

        self._is_fitted = False

    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        if 'organ' in df.columns and 'organ_encoded' not in df.columns:
            df = df.copy()
            df['organ_encoded'] = self.organ_encoder.transform(df['organ'])
        X = df[self.feature_columns].values
        return X

    def train(self, df: pd.DataFrame, target_col: str = 'absorbed_dose_mgy',
              cv_folds: int = 5, verbose: bool = True) -> Dict:

        if verbose:
            print("="*70)
            print("TRAINING ENSEMBLE DOSE PREDICTION MODEL")
            print("="*70)

        # FIXED: Encode organs BEFORE dropping NA
        df = df.copy()
        if 'organ' not in df.columns:
            raise ValueError("DataFrame must have 'organ' column")

        self.organ_encoder.fit(df['organ'])
        df['organ_encoded'] = self.organ_encoder.transform(df['organ'])

        # Now drop NA (organ_encoded exists)
        df = df.dropna(subset=self.feature_columns + [target_col])

        X = df[self.feature_columns].values
        y = df[target_col].values

        if verbose:
            print(f"\nTraining data: {len(df)} samples")
            print(f"Features: {len(self.feature_columns)}")
            print(f"Organs: {len(self.organ_encoder.classes_)}")

        # Fit scaler
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        # Cross-validation
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

        oof_predictions = {
            'rf': np.zeros(len(y)),
            'xgb': np.zeros(len(y)),
            'nn': np.zeros(len(y)),
            'ensemble': np.zeros(len(y))
        }

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled), 1):
            if verbose:
                print(f"\n--- Fold {fold}/{cv_folds} ---")

            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Random Forest
            rf = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            oof_predictions['rf'][val_idx] = rf.predict(X_val)

            # XGBoost
            if XGBOOST_AVAILABLE:
                xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.05, random_state=42)
                xgb_model.fit(X_train, y_train, verbose=False)
                oof_predictions['xgb'][val_idx] = xgb_model.predict(X_val)
            else:
                oof_predictions['xgb'][val_idx] = oof_predictions['rf'][val_idx]

            # Neural Network
            if TORCH_AVAILABLE:
                nn_model = DoseNeuralNetwork(input_dim=X_train.shape[1]).to(self.device)
                optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.001)
                criterion = nn.MSELoss()

                X_train_t = torch.FloatTensor(X_train).to(self.device)
                y_train_t = torch.FloatTensor(y_train).to(self.device)
                X_val_t = torch.FloatTensor(X_val).to(self.device)

                nn_model.train()
                for epoch in range(50):
                    optimizer.zero_grad()
                    pred = nn_model(X_train_t)
                    loss = criterion(pred, y_train_t)
                    loss.backward()
                    optimizer.step()

                nn_model.eval()
                with torch.no_grad():
                    oof_predictions['nn'][val_idx] = nn_model(X_val_t).cpu().numpy()
            else:
                oof_predictions['nn'][val_idx] = oof_predictions['rf'][val_idx]

        # Ensemble
        oof_predictions['ensemble'] = (
            self.weights['rf'] * oof_predictions['rf'] +
            self.weights['xgb'] * oof_predictions['xgb'] +
            self.weights['nn'] * oof_predictions['nn']
        )

        # Calculate metrics
        metrics = {}
        for model_name, preds in oof_predictions.items():
            mae = mean_absolute_error(y, preds)
            rmse = np.sqrt(mean_squared_error(y, preds))
            r2 = r2_score(y, preds)
            metrics[model_name] = {'mae': mae, 'rmse': rmse, 'r2': r2}

        if verbose:
            print("\n" + "="*70)
            print("CROSS-VALIDATION RESULTS")
            print("="*70)
            print(f"{'Model':<15} {'MAE (mGy)':<12} {'RMSE (mGy)':<12} {'R² Score':<10}")
            print("-"*70)
            for model_name, m in metrics.items():
                print(f"{model_name:<15} {m['mae']:>10.2f}  {m['rmse']:>11.2f}  {m['r2']:>9.3f}")
            print("="*70)

        # Train final models on full dataset
        if verbose:
            print("\nTraining final models on full dataset...")

        self.rf_model = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
        self.rf_model.fit(X_scaled, y)

        if XGBOOST_AVAILABLE:
            self.xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.05, random_state=42)
            self.xgb_model.fit(X_scaled, y, verbose=False)

        if TORCH_AVAILABLE:
            self.nn_model = DoseNeuralNetwork(input_dim=X_scaled.shape[1]).to(self.device)
            optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=0.001)
            criterion = nn.MSELoss()

            X_t = torch.FloatTensor(X_scaled).to(self.device)
            y_t = torch.FloatTensor(y).to(self.device)

            self.nn_model.train()
            for epoch in range(100):
                optimizer.zero_grad()
                pred = self.nn_model(X_t)
                loss = criterion(pred, y_t)
                loss.backward()
                optimizer.step()

        self._is_fitted = True

        if verbose:
            print("[SUCCESS] Training complete!")

        return metrics

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model not trained. Call train() first.")

        X = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)

        rf_pred = self.rf_model.predict(X_scaled)

        if self.xgb_model is not None:
            xgb_pred = self.xgb_model.predict(X_scaled)
        else:
            xgb_pred = rf_pred

        if self.nn_model is not None and TORCH_AVAILABLE:
            self.nn_model.eval()
            with torch.no_grad():
                X_t = torch.FloatTensor(X_scaled).to(self.device)
                nn_pred = self.nn_model(X_t).cpu().numpy()
        else:
            nn_pred = rf_pred

        ensemble_pred = (
            self.weights['rf'] * rf_pred +
            self.weights['xgb'] * xgb_pred +
            self.weights['nn'] * nn_pred
        )

        return np.maximum(ensemble_pred, 0)

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)

        with open(os.path.join(directory, 'rf_model.pkl'), 'wb') as f:
            pickle.dump(self.rf_model, f)

        if self.xgb_model is not None:
            with open(os.path.join(directory, 'xgb_model.pkl'), 'wb') as f:
                pickle.dump(self.xgb_model, f)

        if self.nn_model is not None and TORCH_AVAILABLE:
            torch.save(self.nn_model.state_dict(), os.path.join(directory, 'nn_model.pt'))

        with open(os.path.join(directory, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)

        with open(os.path.join(directory, 'organ_encoder.pkl'), 'wb') as f:
            pickle.dump(self.organ_encoder, f)

        metadata = {
            'feature_columns': self.feature_columns,
            'weights': self.weights,
            'organs': list(self.organ_encoder.classes_),
            'device': self.device
        }
        with open(os.path.join(directory, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Model saved to {directory}/")

    @classmethod
    def load(cls, directory: str, device: str = 'cpu'):
        with open(os.path.join(directory, 'metadata.json'), 'r') as f:
            metadata = json.load(f)

        model = cls(device=device)
        model.feature_columns = metadata['feature_columns']
        model.weights = metadata['weights']

        with open(os.path.join(directory, 'scaler.pkl'), 'rb') as f:
            model.scaler = pickle.load(f)

        with open(os.path.join(directory, 'organ_encoder.pkl'), 'rb') as f:
            model.organ_encoder = pickle.load(f)

        with open(os.path.join(directory, 'rf_model.pkl'), 'rb') as f:
            model.rf_model = pickle.load(f)

        xgb_path = os.path.join(directory, 'xgb_model.pkl')
        if os.path.exists(xgb_path) and XGBOOST_AVAILABLE:
            with open(xgb_path, 'rb') as f:
                model.xgb_model = pickle.load(f)

        nn_path = os.path.join(directory, 'nn_model.pt')
        if os.path.exists(nn_path) and TORCH_AVAILABLE:
            input_dim = len(model.feature_columns)
            model.nn_model = DoseNeuralNetwork(input_dim=input_dim).to(device)
            model.nn_model.load_state_dict(torch.load(nn_path, map_location=device))
            model.nn_model.eval()

        model._is_fitted = True
        print(f"Model loaded from {directory}/")
        return model
