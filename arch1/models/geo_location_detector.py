import numpy as np
from sklearn.ensemble import IsolationForest
import geoip2.database
from .base_model import AnomalyDetector

class GeoLocationDetector(AnomalyDetector):
    def __init__(self, geoip_db_path, known_good_ips=None, contamination=0.1):
        super().__init__()
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.geoip_reader = geoip2.database.Reader(geoip_db_path)
        self.known_good_ips = known_good_ips or set()
        
    def _get_location_features(self, ip):
        """Получение географических признаков по IP"""
        try:
            response = self.geoip_reader.city(ip)
            return [
                response.location.latitude,
                response.location.longitude,
                int(ip in self.known_good_ips)
            ]
        except:
            return [0, 0, 0]  # дефолтные значения при ошибке
        
    def _prepare_features(self, X):
        """Подготовка географических признаков"""
        features = []
        for ip in X['ip_address']:
            features.append(self._get_location_features(ip))
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
        scores = -self.model.score_samples(features)
        return scores / np.max(scores)
        
    def __del__(self):
        """Закрываем reader при удалении объекта"""
        if hasattr(self, 'geoip_reader'):
            self.geoip_reader.close() 