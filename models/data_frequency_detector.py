import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from .base_model import AnomalyDetector

class DataFrequencyDetector(AnomalyDetector):
    def __init__(self, window_size=3600, contamination=0.1):
        super().__init__()
        self.window_size = window_size  # размер окна в секундах
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.scaler = StandardScaler()
        
    def _prepare_features(self, X):
        """Подготовка признаков частоты обращений"""
        features = []
        for user_data in X.groupby('user_id'):
            # Агрегируем события по типам ресурсов
            resource_counts = {
                'file_server_access': len(user_data[user_data['resource_type'] == 'file_server']),
                'database_access': len(user_data[user_data['resource_type'] == 'database']),
                'knowledge_base_access': len(user_data[user_data['resource_type'] == 'knowledge_base'])
            }
            features.append(list(resource_counts.values()))
        return np.array(features)
        
    def fit(self, X, y=None):
        features = self._prepare_features(X)
        features_scaled = self.scaler.fit_transform(features)
        self.model.fit(features_scaled)
        return self
        
    def predict(self, X):
        features = self._prepare_features(X)
        features_scaled = self.scaler.transform(features)
        return self.model.predict(features_scaled)
        
    def predict_proba(self, X):
        features = self._prepare_features(X)
        features_scaled = self.scaler.transform(features)
        scores = -self.model.score_samples(features_scaled)
        return scores / np.max(scores) 