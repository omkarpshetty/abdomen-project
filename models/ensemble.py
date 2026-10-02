"""
models/ensemble.py

Stacked ensemble: Random Forest + XGBoost -> Ridge meta-learner.

Why this combination for organ dose estimation:
  - RandomForest: robust to outliers, naturally handles missing organs,
    gives good baseline from bagged trees.
  - XGBoost: captures non-linear interactions between volume/HU/CTDIvol
    that RF misses (e.g., small-volume high-HU organs behave differently
    from large low-HU ones). Also handles imbalanced organ representation.
  - Ridge meta-learner: combines their predictions with minimal risk of
    overfitting on a small dataset (~144 patients).

Saved artifacts (via save / load):
  rf_model.pkl, xgb_model.pkl, meta_model.pkl, organ_encoder.pkl,
  feature_cols.json, model_meta.json
"""
import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBRegressor
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False


# Features used for every prediction — no CNN columns
TABULAR_FEATURES = ["organ_encoded", "volume_cm3", "mean_hu", "mean_ctdivol_mGy"]


class OrganDoseEnsemble:
    """
    Trains and persists a RF+XGB stacked ensemble for organ-specific CT dose
    estimation. Falls back to RF-only if xgboost is not installed.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        xgb_n_estimators: int = 400,
        random_state: int = 42,
        meta_alpha: float = 1.0,
    ):
        self.n_estimators = n_estimators
        self.xgb_n_estimators = xgb_n_estimators
        self.random_state = random_state
        self.meta_alpha = meta_alpha

        self.organ_encoder = LabelEncoder()
        self.feature_cols: list = list(TABULAR_FEATURES)

        self.rf = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )
        if _XGB_AVAILABLE:
            self.xgb = XGBRegressor(
                n_estimators=xgb_n_estimators,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                verbosity=0,
                n_jobs=-1,
            )
        else:
            self.xgb = None

        self.meta = Ridge(alpha=meta_alpha)
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode organ column and return df with all feature cols present."""
        df = df.copy()
        df["organ_encoded"] = self.organ_encoder.transform(df["organ"])
        return df

    def _stack_predictions(self, X: pd.DataFrame) -> np.ndarray:
        """Returns (n, 2) array of [rf_pred, xgb_pred] for meta-learning."""
        rf_pred = self.rf.predict(X[self.feature_cols])
        if self.xgb is not None:
            xgb_pred = self.xgb.predict(X[self.feature_cols])
        else:
            xgb_pred = rf_pred  # RF-only fallback
        return np.column_stack([rf_pred, xgb_pred])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> dict:
        """
        Fit the ensemble on organ-dose label rows.

        Parameters
        ----------
        df : rows from organ_dose_labels.csv — must have columns:
             organ, volume_cm3, mean_hu, mean_ctdivol_mGy,
             pseudo_label_dose_mGy, UID, PHASE

        Returns
        -------
        dict with OOF CV metrics
        """
        df = df.dropna(subset=["volume_cm3", "mean_hu", "mean_ctdivol_mGy", "pseudo_label_dose_mGy"])

        self.organ_encoder.fit(df["organ"])
        df = self._prepare(df)
        self.feature_cols = list(TABULAR_FEATURES)

        X = df[self.feature_cols]
        y = df["pseudo_label_dose_mGy"]

        print(f"Training on {len(df)} organ-rows from {df['UID'].nunique()} patients.")
        print(f"Features: {self.feature_cols}")

        # --- 5-fold CV for honest metrics ---
        cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        oof_rf = np.zeros(len(y))
        oof_xgb = np.zeros(len(y))

        for fold, (tr_idx, val_idx) in enumerate(cv.split(X), 1):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr = y.iloc[tr_idx]

            rf_fold = RandomForestRegressor(
                n_estimators=self.n_estimators,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1,
            )
            rf_fold.fit(X_tr, y_tr)
            oof_rf[val_idx] = rf_fold.predict(X_val)

            if self.xgb is not None:
                xgb_fold = XGBRegressor(
                    n_estimators=self.xgb_n_estimators,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.random_state,
                    verbosity=0,
                    n_jobs=-1,
                )
                xgb_fold.fit(X_tr, y_tr)
                oof_xgb[val_idx] = xgb_fold.predict(X_val)
            else:
                oof_xgb[val_idx] = oof_rf[val_idx]

        # Fit meta-learner on OOF predictions
        meta_X = np.column_stack([oof_rf, oof_xgb])
        self.meta.fit(meta_X, y)
        oof_ensemble = self.meta.predict(meta_X)

        cv_metrics = {
            "rf_mae":       float(mean_absolute_error(y, oof_rf)),
            "rf_r2":        float(r2_score(y, oof_rf)),
            "xgb_mae":      float(mean_absolute_error(y, oof_xgb)) if self.xgb else None,
            "xgb_r2":       float(r2_score(y, oof_xgb)) if self.xgb else None,
            "ensemble_mae": float(mean_absolute_error(y, oof_ensemble)),
            "ensemble_r2":  float(r2_score(y, oof_ensemble)),
        }

        print("\n=== 5-fold OOF performance ===")
        print(f"  RandomForest   MAE: {cv_metrics['rf_mae']:.2f} mGy   R²: {cv_metrics['rf_r2']:.3f}")
        if self.xgb:
            print(f"  XGBoost        MAE: {cv_metrics['xgb_mae']:.2f} mGy   R²: {cv_metrics['xgb_r2']:.3f}")
        print(f"  Ensemble       MAE: {cv_metrics['ensemble_mae']:.2f} mGy   R²: {cv_metrics['ensemble_r2']:.3f}")

        # --- Per-organ breakdown ---
        test_df = df.copy()
        test_df["oof_ensemble"] = oof_ensemble
        print("\nPer-organ OOF performance:")
        for organ, grp in test_df.groupby("organ"):
            if len(grp) < 2:
                continue
            o_mae = mean_absolute_error(grp["pseudo_label_dose_mGy"], grp["oof_ensemble"])
            print(f"  {organ:<22} MAE: {o_mae:6.2f} mGy  (n={len(grp)})")

        # --- Refit on full data for deployment ---
        self.rf.fit(X, y)
        if self.xgb is not None:
            self.xgb.fit(X, y)
        self._is_fitted = True

        # --- Feature importances ---
        print("\nRF feature importances:")
        for name, imp in sorted(zip(self.feature_cols, self.rf.feature_importances_), key=lambda x: -x[1]):
            print(f"  {name:<28} {imp:.3f}")
        print(f"\nMeta-learner weights: RF={self.meta.coef_[0]:.3f}  XGB={self.meta.coef_[1]:.3f}")

        return cv_metrics

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict organ dose (mGy) for each row in df.
        df must have: organ, volume_cm3, mean_hu, mean_ctdivol_mGy
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() or load() first.")
        df = self._prepare(df)
        stacked = self._stack_predictions(df)
        return self.meta.predict(stacked)

    def save(self, directory: str):
        """Persist all model artifacts to `directory`."""
        os.makedirs(directory, exist_ok=True)
        pickle.dump(self.rf,           open(os.path.join(directory, "rf_model.pkl"),       "wb"))
        pickle.dump(self.meta,         open(os.path.join(directory, "meta_model.pkl"),     "wb"))
        pickle.dump(self.organ_encoder,open(os.path.join(directory, "organ_encoder.pkl"),  "wb"))
        if self.xgb is not None:
            pickle.dump(self.xgb,      open(os.path.join(directory, "xgb_model.pkl"),      "wb"))
        with open(os.path.join(directory, "feature_cols.json"), "w") as f:
            json.dump(self.feature_cols, f)
        with open(os.path.join(directory, "model_meta.json"), "w") as f:
            json.dump({
                "xgb_available": self.xgb is not None,
                "n_estimators":  self.n_estimators,
                "xgb_n_estimators": self.xgb_n_estimators,
                "organs": list(self.organ_encoder.classes_),
            }, f, indent=2)
        print(f"Model saved to {directory}/")

    @classmethod
    def load(cls, directory: str) -> "OrganDoseEnsemble":
        """Load a previously saved ensemble from `directory`."""
        obj = cls.__new__(cls)
        obj.rf           = pickle.load(open(os.path.join(directory, "rf_model.pkl"),      "rb"))
        obj.meta         = pickle.load(open(os.path.join(directory, "meta_model.pkl"),    "rb"))
        obj.organ_encoder= pickle.load(open(os.path.join(directory, "organ_encoder.pkl"), "rb"))
        xgb_path = os.path.join(directory, "xgb_model.pkl")
        obj.xgb = pickle.load(open(xgb_path, "rb")) if os.path.exists(xgb_path) else None
        with open(os.path.join(directory, "feature_cols.json")) as f:
            obj.feature_cols = json.load(f)
        obj._is_fitted = True
        print(f"Loaded model from {directory}/")
        return obj
