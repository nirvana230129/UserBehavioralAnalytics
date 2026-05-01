import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from .adaptive_feature_processor import AdaptiveFeatureProcessor
from .sequence_encoder import SequenceEncoder


class HybridInsiderDetector:
    def __init__(self, pos_weight=100, use_lstm=True, lstm_epochs=30):
        self.pos_weight = min(pos_weight, 100)
        self.use_lstm = use_lstm
        self.lstm_epochs = lstm_epochs
        self.processor = AdaptiveFeatureProcessor()
        self.sequence_encoder = SequenceEncoder(epochs=lstm_epochs) if use_lstm else None
        self.lgbm = lgb.LGBMClassifier(
            objective='binary',
            boosting_type='gbdt',
            scale_pos_weight=self.pos_weight,
            learning_rate=0.01,
            num_leaves=15,
            max_depth=4,
            min_child_samples=10,
            reg_alpha=0.1,
            reg_lambda=0.1,
            bagging_fraction=0.8,
            bagging_freq=5,
            feature_fraction=0.8,
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        self.calibrator = None
        self.feature_names = None
        self.is_fitted = False

    def _build_feature_matrix(self, X: pd.DataFrame, sequences=None,
                               apply_dropout=False, transform_only=False) -> np.ndarray:
        if transform_only:
            X_proc = self.processor.transform(X, apply_dropout=False)
        else:
            X_proc = self.processor.fit_transform(X, apply_dropout=apply_dropout)

        mat = X_proc.values.astype(float)

        if self.use_lstm and self.sequence_encoder is not None and sequences is not None:
            lstm_feats = self.sequence_encoder.extract_features(sequences)
            mat = np.hstack([mat, lstm_feats])

        return mat

    def fit(self, X_train: pd.DataFrame, y_train, sequences_train=None,
            X_val: pd.DataFrame = None, y_val=None, sequences_val=None):
        if self.use_lstm and self.sequence_encoder is not None and sequences_train is not None:
            normal_idx = [i for i, y in enumerate(y_train) if y == 0]
            normal_seqs = [sequences_train[i] for i in normal_idx]
            if normal_seqs:
                print("  Обучение LSTM-автоэнкодера...")
                self.sequence_encoder.fit(normal_seqs)

        self.processor.fit(X_train)
        X_tr = self.processor.transform(X_train, apply_dropout=True)
        mat_train = X_tr.values.astype(float)

        if self.use_lstm and self.sequence_encoder is not None and sequences_train is not None:
            lstm_feats = self.sequence_encoder.extract_features(sequences_train)
            mat_train = np.hstack([mat_train, lstm_feats])

        base_names = list(X_tr.columns)
        if self.use_lstm and self.sequence_encoder is not None:
            base_names += self.sequence_encoder.get_feature_names()
        self.feature_names = base_names

        mat_val, y_val_arr = None, None
        if X_val is not None and y_val is not None:
            X_v = self.processor.transform(X_val, apply_dropout=False)
            mat_val = X_v.values.astype(float)
            if self.use_lstm and self.sequence_encoder is not None and sequences_val is not None:
                lstm_v = self.sequence_encoder.extract_features(sequences_val)
                mat_val = np.hstack([mat_val, lstm_v])
            y_val_arr = np.array(y_val)

        eval_set = [(mat_val, y_val_arr)] if mat_val is not None else None
        self.lgbm.fit(
            mat_train, np.array(y_train),
            eval_set=eval_set,
            callbacks=[lgb.early_stopping(50, verbose=False)] if eval_set else None,
        )

        if mat_val is not None and len(np.unique(y_val_arr)) > 1:
            try:
                self.calibrator = CalibratedClassifierCV(self.lgbm, method='isotonic', cv='prefit')
                self.calibrator.fit(mat_val, y_val_arr)
            except Exception:
                self.calibrator = None

        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame, sequences=None) -> np.ndarray:
        mat = self._build_feature_matrix(X, sequences=sequences, transform_only=True)
        if self.calibrator is not None:
            try:
                p = self.calibrator.predict_proba(mat)
                return p[:, 1]
            except Exception:
                pass
        p = self.lgbm.predict_proba(mat)
        return p[:, 1]

    def predict(self, X: pd.DataFrame, sequences=None, threshold=0.5) -> np.ndarray:
        proba = self.predict_proba(X, sequences)
        return (proba >= threshold).astype(int)

    def get_feature_matrix(self, X: pd.DataFrame, sequences=None) -> np.ndarray:
        return self._build_feature_matrix(X, sequences=sequences, transform_only=True)

    def get_feature_importance(self, top_n=20):
        importances = self.lgbm.feature_importances_
        pairs = list(zip(self.feature_names or range(len(importances)), importances))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:top_n]
