import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from .base_model import AnomalyDetector

class ResourceAccessDetector(AnomalyDetector):
    def __init__(self, contamination=0.1):
        super().__init__()
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def _prepare_features(self, X):
        """Подготовка признаков доступа к ресурсам"""
        features = np.column_stack([
            X['unique_pcs'],
            X['is_admin'],
            X['unique_domains']
        ])
        if not self.is_fitted:
            self.is_fitted = True
            return self.scaler.fit_transform(features)
        return self.scaler.transform(features)
        
    def fit(self, X, y=None):
        features = self._prepare_features(X)
        self.model.fit(features)
        return self
        
    def predict(self, X):
        features = self._prepare_features(X)
        return self.model.predict(features)
        
    def predict_proba(self, X):
        features = self._prepare_features(X)
        scores = -self.model.score_samples(features)
        return scores / np.max(scores) if len(scores) > 0 else scores 