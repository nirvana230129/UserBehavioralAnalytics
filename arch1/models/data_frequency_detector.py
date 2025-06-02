import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from .base_model import AnomalyDetector

class DataFrequencyDetector(AnomalyDetector):
    def __init__(self, window_size=3600, contamination=0.1):
        super().__init__()
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def _prepare_features(self, X):
        """Подготовка признаков для анализа частоты обращений к данным"""
        features = np.column_stack([
            X['avg_http_requests_per_day'],
            X['avg_emails_sent_per_day'],
            X['avg_emails_received_per_day'],
            X['avg_file_copies_per_day']
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
        """
        Вероятности аномалий
        
        Parameters:
        -----------
        X : pandas.DataFrame или numpy.ndarray
            Данные для предсказания
            
        Returns:
        --------
        numpy.ndarray
            Массив вероятностей аномальности в диапазоне [0, 1]
        """
        features = self._prepare_features(X)
        scores = -self.model.score_samples(features)  # Получаем сырые оценки
        
        if len(scores) > 0:
            # Проверяем, есть ли разброс в оценках
            score_range = scores.max() - scores.min()
            if score_range > 0:
                # Если есть разброс, нормализуем
                scores_norm = (scores - scores.min()) / score_range
            else:
                # Если все оценки одинаковые, считаем их нормальными
                scores_norm = np.zeros_like(scores)
            return np.clip(scores_norm, 0, 1)  # Обрезаем на всякий случай
        return scores 