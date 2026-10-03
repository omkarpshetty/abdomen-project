"""
XGBoost-Based Dose Predictor

Gradient boosting model that excels at learning complex non-linear
relationships in tabular data. Particularly good at:
- Capturing threshold effects (e.g., dose increases sharply above certain kVp)
- Learning patient-specific correction factors
- Handling missing values naturally
- Feature importance analysis
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import json
from pathlib import Path


class XGBoostDosePredictor:
    """
    XGBoost-based organ dose predictor.
    """

    def __init__(self, n_estimators=500, max_depth=8, learning_rate=0.05):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate

        self.model = None
        self.feature_names = None
        self._is_fitted = False

    def _engineer_features(self, df):
        """
        Create advanced features that XGBoost can learn from.

        Includes:
        - Interaction terms
        - Polynomial features
        - Patient-specific ratios
        - Physics-informed features
        """
        df = df.copy()

        # Energy-related features
        df['energy_proxy'] = df['mean_kvp'] ** 2 * df['mean_exposure_mAs'] / 10000

        # Organ density features
        df['is_air_filled'] = (df['mean_hu'] < -500).astype(int)  # Lungs, bowel gas
        df['is_dense'] = (df['mean_hu'] > 100).astype(int)  # Bones
        df['is_soft_tissue'] = ((df['mean_hu'] >= -50) & (df['mean_hu'] <= 100)).astype(int)

        # Volume categories
        df['volume_log'] = np.log1p(df['volume_cm3'])
        df['volume_category'] = pd.cut(df['volume_cm3'],
                                       bins=[0, 100, 500, 1500, 10000],
                                       labels=['small', 'medium', 'large', 'very_large'])

        # Organ-specific features
        organ_groups = {
            'central': ['LIVER', 'STOMACH', 'PANCREAS', 'SPLEEN'],
            'posterior': ['KIDNEYS', 'SPINAL CORD'],
            'anterior': ['BOWEL', 'URINARY BLADDER'],
            'thorax': ['HEART', 'LUNGS'],
            'skeletal': ['BONES']
        }

        for group_name, organs in organ_groups.items():
            df[f'is_{group_name}'] = df['organ'].isin(organs).astype(int)

        # Scan technique interactions
        df['kvp_mas_interaction'] = df['mean_kvp'] * df['mean_exposure_mAs'] / 1000
        df['pitch_thickness'] = df['pitch_factor'] * df['mean_slice_thickness_mm']

        # Patient-level features (aggregated per patient)
        patient_stats = df.groupby('UID').agg({
            'volume_cm3': ['sum', 'mean', 'std'],
            'mean_hu': ['mean', 'std']
        })
        patient_stats.columns = ['_'.join(col).strip() for col in patient_stats.columns]
        patient_stats = patient_stats.add_prefix('patient_')

        df = df.merge(patient_stats, left_on='UID', right_index=True, how='left')

        # Relative organ size within patient
        df['organ_volume_ratio'] = df['volume_cm3'] / (df['patient_volume_cm3_sum'] + 1)

        return df

    def train(self, organ_features_path, dicom_params_path, labels=None):
        """
        Train XGBoost model.

        Args:
            organ_features_path: Path to organ_features.csv
            dicom_params_path: Path to dicom_params.csv
            labels: Optional pre-computed labels. If None, generates physics-based labels.
        """
        print("="*70)
        print("XGBOOST DOSE PREDICTOR - TRAINING")
        print("="*70)

        # Load data
        print(f"\nLoading data...")
        organ_df = pd.read_csv(organ_features_path, dtype={'UID': str})
        dicom_df = pd.read_csv(dicom_params_path, dtype={'UID': str})

        # Merge
        df = organ_df.merge(dicom_df, on=['UID', 'PHASE'], how='inner')
        df = df.dropna(subset=['volume_cm3', 'mean_hu', 'mean_kvp'])

        print(f"  {len(df)} samples from {df['UID'].nunique()} patients")

        # Generate labels if not provided
        if labels is None:
            print("\nGenerating training labels...")
            from models.patient_specific_ai import PatientSpecificAI
            temp_model = PatientSpecificAI()
            labels = temp_model._create_synthetic_labels(df)

        # Engineer features
        print("\nEngineering features...")
        df = self._engineer_features(df)

        # Select features for XGBoost
        feature_cols = [col for col in df.columns if col not in
                       ['UID', 'PHASE', 'organ', 'volume_category']]

        # Handle categorical features
        for col in feature_cols:
            if df[col].dtype == 'object':
                df[col] = pd.Categorical(df[col]).codes

        X = df[feature_cols].fillna(-999).values
        y = labels
        groups = df['UID'].values

        self.feature_names = feature_cols
        print(f"  Features: {len(feature_cols)}")

        # Cross-validation
        print("\nTraining with 5-fold CV...")
        cv = GroupKFold(n_splits=5)
        oof_preds = np.zeros(len(y))

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups), 1):
            print(f"\nFold {fold}/5:")

            dtrain = xgb.DMatrix(X[train_idx], label=y[train_idx],
                                feature_names=self.feature_names)
            dval = xgb.DMatrix(X[val_idx], label=y[val_idx],
                              feature_names=self.feature_names)

            params = {
                'objective': 'reg:squarederror',
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'gamma': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'tree_method': 'hist',
                'eval_metric': 'mae'
            }

            model = xgb.train(
                params,
                dtrain,
                num_boost_round=self.n_estimators,
                evals=[(dval, 'val')],
                early_stopping_rounds=50,
                verbose_eval=100
            )

            oof_preds[val_idx] = model.predict(dval)

        # Calculate metrics
        mae = mean_absolute_error(y, oof_preds)
        r2 = r2_score(y, oof_preds)

        print("\n" + "="*70)
        print("CROSS-VALIDATION RESULTS")
        print("="*70)
        print(f"  MAE:  {mae:.3f}")
        print(f"  R²:   {r2:.3f}")

        # Train final model on all data
        print("\nTraining final model on full dataset...")
        dtrain_full = xgb.DMatrix(X, label=y, feature_names=self.feature_names)

        self.model = xgb.train(
            params,
            dtrain_full,
            num_boost_round=self.n_estimators,
            evals=[(dtrain_full, 'train')],
            verbose_eval=False
        )

        self._is_fitted = True

        # Feature importance
        importance = self.model.get_score(importance_type='gain')
        print("\nTop 10 Most Important Features:")
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for feat, score in sorted_importance[:10]:
            print(f"  {feat:<30} {score:>10.1f}")

        return {'mae': mae, 'r2': r2}

    def predict(self, organ_features_df, dicom_params_df):
        """Predict doses for new patients."""
        if not self._is_fitted:
            raise RuntimeError("Model not trained!")

        # Merge and engineer features
        df = organ_features_df.merge(dicom_params_df, on=['UID', 'PHASE'], how='inner')
        df = self._engineer_features(df)

        # Prepare features
        X = df[self.feature_names].fillna(-999).values
        dtest = xgb.DMatrix(X, feature_names=self.feature_names)

        return self.model.predict(dtest)

    def save(self, save_dir):
        """Save model."""
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        self.model.save_model(str(Path(save_dir) / 'xgboost_model.json'))

        meta = {
            'feature_names': self.feature_names,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate
        }
        json.dump(meta, open(Path(save_dir) / 'meta.json', 'w'), indent=2)

        print(f"\n[OK] XGBoost model saved to: {save_dir}/")

    @classmethod
    def load(cls, save_dir):
        """Load model."""
        with open(Path(save_dir) / 'meta.json') as f:
            meta = json.load(f)

        obj = cls(
            n_estimators=meta['n_estimators'],
            max_depth=meta['max_depth'],
            learning_rate=meta['learning_rate']
        )

        obj.model = xgb.Booster()
        obj.model.load_model(str(Path(save_dir) / 'xgboost_model.json'))
        obj.feature_names = meta['feature_names']
        obj._is_fitted = True

        print(f"[OK] XGBoost model loaded from: {save_dir}/")
        return obj


if __name__ == "__main__":
    print("XGBoost Dose Predictor initialized.")
    print("Use train() method to train on your data.")
