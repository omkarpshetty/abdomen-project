"""
models/ensemble_full.py

Complete multi-model ensemble: Random Forest + XGBoost + LightGBM + SVR
Trains all 4 models and compares their performance, then stacks the best ones.

For medical dose estimation with small datasets (73-144 patients), this
combination gives the most robust predictions:
  - RandomForest: baseline, handles outliers
  - XGBoost: non-linear interactions
  - LightGBM: fast boosting, good for imbalanced data
  - SVR: specialized for small datasets

Saved artifacts: rf.pkl, xgb.pkl, lgb.pkl, svr.pkl, meta.pkl, encoder.pkl, etc.
"""
import json
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from xgboost import XGBRegressor
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    _LGB_AVAILABLE = True
except ImportError:
    _LGB_AVAILABLE = False


TABULAR_FEATURES = ["organ_encoded", "volume_cm3", "mean_hu", "mean_ctdivol_mGy"]


class FullEnsemble:
    """
    Trains RF + XGB + LGB + SVR, compares them, then stacks the best performers.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.random_state = random_state

        self.organ_encoder = LabelEncoder()
        self.scaler = StandardScaler()  # SVR needs scaled features
        self.feature_cols = list(TABULAR_FEATURES)

        # Initialize all models
        self.rf = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )

        self.xgb = XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            random_state=random_state,
            verbosity=0,
            n_jobs=-1,
        ) if _XGB_AVAILABLE else None

        self.lgb = LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            random_state=random_state,
            verbose=-1,
            n_jobs=-1,
        ) if _LGB_AVAILABLE else None

        self.svr = SVR(
            kernel='rbf',
            C=10.0,
            epsilon=0.1,
            gamma='scale',
        )

        self.meta = Ridge(alpha=1.0)
        self.use_models = []  # Will be populated after CV
        self._is_fitted = False

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["organ_encoded"] = self.organ_encoder.transform(df["organ"])
        return df

    def fit(self, df: pd.DataFrame) -> dict:
        """
        Train all models with 5-fold CV, compare performance, stack the best.
        """
        df = df.dropna(subset=["volume_cm3", "mean_hu", "mean_ctdivol_mGy", "pseudo_label_dose_mGy"])

        self.organ_encoder.fit(df["organ"])
        df = self._prepare(df)

        X = df[self.feature_cols]
        y = df["pseudo_label_dose_mGy"]

        print(f"Training on {len(df)} organ-rows from {df['UID'].nunique()} patients.")
        print(f"Organs: {sorted(df['organ'].unique())}")

        # Fit scaler on all data
        self.scaler.fit(X)
        X_scaled = pd.DataFrame(self.scaler.transform(X), columns=X.columns, index=X.index)

        # 5-fold CV for all models
        cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)

        oof_preds = {
            'rf': np.zeros(len(y)),
            'xgb': np.zeros(len(y)),
            'lgb': np.zeros(len(y)),
            'svr': np.zeros(len(y)),
        }

        print("\n" + "="*70)
        print("Training individual models with 5-fold CV...")
        print("="*70)

        for fold, (tr_idx, val_idx) in enumerate(cv.split(X), 1):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            X_tr_scaled, X_val_scaled = X_scaled.iloc[tr_idx], X_scaled.iloc[val_idx]
            y_tr = y.iloc[tr_idx]

            # Random Forest
            rf_fold = RandomForestRegressor(
                n_estimators=self.n_estimators,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1,
            )
            rf_fold.fit(X_tr, y_tr)
            oof_preds['rf'][val_idx] = rf_fold.predict(X_val)

            # XGBoost
            if self.xgb is not None:
                xgb_fold = XGBRegressor(
                    n_estimators=400,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    random_state=self.random_state,
                    verbosity=0,
                    n_jobs=-1,
                )
                xgb_fold.fit(X_tr, y_tr)
                oof_preds['xgb'][val_idx] = xgb_fold.predict(X_val)

            # LightGBM
            if self.lgb is not None:
                lgb_fold = LGBMRegressor(
                    n_estimators=400,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    random_state=self.random_state,
                    verbose=-1,
                    n_jobs=-1,
                )
                lgb_fold.fit(X_tr, y_tr)
                oof_preds['lgb'][val_idx] = lgb_fold.predict(X_val)

            # SVR (needs scaled features)
            svr_fold = SVR(kernel='rbf', C=10.0, epsilon=0.1, gamma='scale')
            svr_fold.fit(X_tr_scaled, y_tr)
            oof_preds['svr'][val_idx] = svr_fold.predict(X_val_scaled)

        # Calculate metrics for each model
        results = {}
        print("\n" + "="*70)
        print("Individual Model Performance (5-fold OOF):")
        print("="*70)

        for name, preds in oof_preds.items():
            if name == 'xgb' and self.xgb is None:
                continue
            if name == 'lgb' and self.lgb is None:
                continue
            mae = mean_absolute_error(y, preds)
            r2 = r2_score(y, preds)
            results[name] = {'mae': mae, 'r2': r2, 'preds': preds}

            model_label = {
                'rf': 'Random Forest',
                'xgb': 'XGBoost',
                'lgb': 'LightGBM',
                'svr': 'SVR',
            }[name]
            print(f"  {model_label:<20} MAE: {mae:6.2f} mGy   R²: {r2:.3f}")

        # Select best models for stacking (those with R² > 0.90)
        valid_models = [(k, v) for k, v in results.items() if v['r2'] > 0.90]
        valid_models.sort(key=lambda x: -x[1]['r2'])
        self.use_models = [k for k, _ in valid_models]

        print(f"\n  -> Stacking models: {', '.join(self.use_models)}")

        # Stack predictions from selected models
        meta_X = np.column_stack([results[m]['preds'] for m in self.use_models])
        self.meta.fit(meta_X, y)
        oof_ensemble = self.meta.predict(meta_X)

        ensemble_mae = mean_absolute_error(y, oof_ensemble)
        ensemble_r2 = r2_score(y, oof_ensemble)

        print(f"\n{'='*70}")
        print(f"  ENSEMBLE (stacked)   MAE: {ensemble_mae:6.2f} mGy   R²: {ensemble_r2:.3f}")
        print(f"{'='*70}")

        # Per-organ breakdown
        test_df = df.copy()
        test_df["oof_ensemble"] = oof_ensemble
        print("\nPer-organ ensemble performance:")
        for organ, grp in test_df.groupby("organ"):
            if len(grp) < 2:
                continue
            o_mae = mean_absolute_error(grp["pseudo_label_dose_mGy"], grp["oof_ensemble"])
            print(f"  {organ:<22} MAE: {o_mae:6.2f} mGy  (n={len(grp)})")

        # Refit all models on full data for deployment
        self.rf.fit(X, y)
        if self.xgb is not None:
            self.xgb.fit(X, y)
        if self.lgb is not None:
            self.lgb.fit(X, y)
        self.svr.fit(X_scaled, y)
        self._is_fitted = True

        # Feature importances
        print("\nRandom Forest feature importances:")
        for name, imp in sorted(zip(self.feature_cols, self.rf.feature_importances_), key=lambda x: -x[1]):
            print(f"  {name:<28} {imp:.3f}")

        print(f"\nMeta-learner weights: {dict(zip(self.use_models, self.meta.coef_))}")

        return {
            'rf_mae': results['rf']['mae'],
            'rf_r2': results['rf']['r2'],
            'xgb_mae': results.get('xgb', {}).get('mae'),
            'xgb_r2': results.get('xgb', {}).get('r2'),
            'lgb_mae': results.get('lgb', {}).get('mae'),
            'lgb_r2': results.get('lgb', {}).get('r2'),
            'svr_mae': results['svr']['mae'],
            'svr_r2': results['svr']['r2'],
            'ensemble_mae': ensemble_mae,
            'ensemble_r2': ensemble_r2,
        }

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() or load() first.")

        df = self._prepare(df)
        X = df[self.feature_cols]
        X_scaled = pd.DataFrame(self.scaler.transform(X), columns=X.columns)

        preds = []
        if 'rf' in self.use_models:
            preds.append(self.rf.predict(X))
        if 'xgb' in self.use_models and self.xgb is not None:
            preds.append(self.xgb.predict(X))
        if 'lgb' in self.use_models and self.lgb is not None:
            preds.append(self.lgb.predict(X))
        if 'svr' in self.use_models:
            preds.append(self.svr.predict(X_scaled))

        stacked = np.column_stack(preds)
        return self.meta.predict(stacked)

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        pickle.dump(self.rf, open(os.path.join(directory, "rf.pkl"), "wb"))
        pickle.dump(self.svr, open(os.path.join(directory, "svr.pkl"), "wb"))
        pickle.dump(self.meta, open(os.path.join(directory, "meta.pkl"), "wb"))
        pickle.dump(self.organ_encoder, open(os.path.join(directory, "encoder.pkl"), "wb"))
        pickle.dump(self.scaler, open(os.path.join(directory, "scaler.pkl"), "wb"))

        if self.xgb is not None:
            pickle.dump(self.xgb, open(os.path.join(directory, "xgb.pkl"), "wb"))
        if self.lgb is not None:
            pickle.dump(self.lgb, open(os.path.join(directory, "lgb.pkl"), "wb"))

        with open(os.path.join(directory, "meta.json"), "w") as f:
            json.dump({
                "use_models": self.use_models,
                "feature_cols": self.feature_cols,
                "organs": list(self.organ_encoder.classes_),
            }, f, indent=2)

        print(f"Full ensemble saved to {directory}/")

    @classmethod
    def load(cls, directory: str) -> "FullEnsemble":
        obj = cls.__new__(cls)
        obj.rf = pickle.load(open(os.path.join(directory, "rf.pkl"), "rb"))
        obj.svr = pickle.load(open(os.path.join(directory, "svr.pkl"), "rb"))
        obj.meta = pickle.load(open(os.path.join(directory, "meta.pkl"), "rb"))
        obj.organ_encoder = pickle.load(open(os.path.join(directory, "encoder.pkl"), "rb"))
        obj.scaler = pickle.load(open(os.path.join(directory, "scaler.pkl"), "rb"))

        xgb_path = os.path.join(directory, "xgb.pkl")
        obj.xgb = pickle.load(open(xgb_path, "rb")) if os.path.exists(xgb_path) else None

        lgb_path = os.path.join(directory, "lgb.pkl")
        obj.lgb = pickle.load(open(lgb_path, "rb")) if os.path.exists(lgb_path) else None

        with open(os.path.join(directory, "meta.json")) as f:
            meta = json.load(f)
            obj.use_models = meta["use_models"]
            obj.feature_cols = meta["feature_cols"]

        obj._is_fitted = True
        print(f"Loaded full ensemble from {directory}/")
        return obj
