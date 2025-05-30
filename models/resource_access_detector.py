import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from .base_model import AnomalyDetector

class ResourceAccessDetector(AnomalyDetector):
    def __init__(self, contamination=0.1):
        super().__init__()
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.scaler = StandardScaler()
        
    def _prepare_features(self, X):
        """Подготовка признаков доступа к ресурсам"""
        features = []
        for user_data in X.groupby('user_id'):
            # Анализируем паттерны доступа
            access_patterns = {
                'unique_resources': len(user_data['resource_id'].unique()),
                'failed_attempts': len(user_data[user_data['access_status'] == 'failed']),
                'success_ratio': len(user_data[user_data['access_status'] == 'success']) / len(user_data),
                'resource_entropy': self._calculate_resource_entropy(user_data['resource_id'])
            }
            features.append(list(access_patterns.values()))
        return np.array(features)
    
    def _calculate_resource_entropy(self, resource_series):
        """Расчет энтропии доступа к ресурсам"""
        _, counts = np.unique(resource_series, return_counts=True)
        probabilities = counts / len(resource_series)
        return -np.sum(probabilities * np.log2(probabilities))
        
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