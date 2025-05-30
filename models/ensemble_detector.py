import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .base_model import AnomalyDetector

class EnsembleDetector(AnomalyDetector):
    def __init__(self, base_detectors, weights=None):
        """
        Parameters:
        -----------
        base_detectors : dict
            Словарь с базовыми детекторами {name: detector}
        weights : dict, optional
            Веса для каждого детектора {name: weight}
        """
        super().__init__()
        self.base_detectors = base_detectors
        self.weights = weights or {name: 1.0 for name in base_detectors.keys()}
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
    def fit(self, X, y=None):
        """
        Обучение ансамбля на основе предсказаний базовых детекторов
        """
        # Получаем предсказания от всех базовых детекторов
        predictions = self._get_base_predictions(X)
        
        if y is None:
            # Если метки не предоставлены, используем консенсус базовых детекторов
            y = np.mean(predictions, axis=1) > 0.5
            
        self.model.fit(predictions, y)
        return self
        
    def _get_base_predictions(self, X):
        """Получение предсказаний от всех базовых детекторов"""
        predictions = []
        for name, detector in self.base_detectors.items():
            pred = detector.predict_proba(X)
            predictions.append(pred * self.weights[name])
        return np.column_stack(predictions)
        
    def predict(self, X):
        """Предсказание аномальности"""
        predictions = self._get_base_predictions(X)
        return self.model.predict(predictions)
        
    def predict_proba(self, X):
        """Вероятность аномальности"""
        predictions = self._get_base_predictions(X)
        return self.model.predict_proba(predictions)[:, 1]  # Вероятность аномального класса
        
    def update_weights(self, new_weights):
        """Обновление весов базовых детекторов"""
        self.weights.update(new_weights) 