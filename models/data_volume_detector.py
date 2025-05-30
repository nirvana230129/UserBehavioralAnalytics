import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from .base_model import AnomalyDetector

class DataVolumeDetector(AnomalyDetector):
    def __init__(self, window_size=3600, contamination=0.1):
        super().__init__()
        self.window_size = window_size
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.scaler = StandardScaler()
        
    def _prepare_features(self, X):
        """Подготовка признаков объемов данных"""
        features = []
        for user_data in X.groupby('user_id'):
            # Агрегируем объемы по типам ресурсов
            volume_stats = {
                'total_volume': user_data['data_volume'].sum(),
                'mean_volume': user_data['data_volume'].mean(),
                'max_volume': user_data['data_volume'].max(),
                'std_volume': user_data['data_volume'].std(),
            }
            features.append(list(volume_stats.values()))
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