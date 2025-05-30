import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import datetime
from .base_model import AnomalyDetector

class TimeActivityDetector(AnomalyDetector):
    def __init__(self, contamination=0.1):
        super().__init__()
        self.model = IsolationForest(contamination=contamination, random_state=42)
        
    def _prepare_features(self, X):
        """Подготовка признаков времени"""
        features = []
        for timestamp in X['timestamp']:
            dt = datetime.fromtimestamp(timestamp)
            features.append([
                dt.hour,
                dt.minute,
                dt.weekday(),
                int(dt.strftime('%U')),  # номер недели
                X['connection_type']  # тип подключения (офис/удаленно)
            ])
        return np.array(features)
        
    def fit(self, X, y=None):
        features = self._prepare_features(X)
        self.model.fit(features)
        return self
        
    def predict(self, X):
        features = self._prepare_features(X)
        return self.model.predict(features)
        
    def predict_proba(self, X):
        features = self._prepare_features(X)
        # Преобразуем decision_function в вероятности
        scores = -self.model.score_samples(features)
        return scores / np.max(scores) 