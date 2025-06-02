import numpy as np
from sklearn.ensemble import IsolationForest
from .base_model import AnomalyDetector

class TimeActivityDetector(AnomalyDetector):
    def __init__(self, contamination=0.1):
        super().__init__()
        self.model = IsolationForest(contamination=contamination, random_state=42)
        
    def _prepare_features(self, X):
        """Подготовка признаков для анализа временных паттернов"""
        features = np.column_stack([
            X['avg_logons_per_day'],
            X['weekend_logon_ratio'],
            X['after_hours_logon_ratio'],
            X['avg_device_usage_per_day'],
            X['device_usage_ratio']
        ])
        return features
        
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