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
        # Используем параметр class_weight='balanced' для лучшей работы с несбалансированными данными
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        
    def fit(self, X, y=None):
        """
        Обучение ансамбля на основе предсказаний базовых детекторов
        """
        # Сначала обучаем каждый базовый детектор
        for detector in self.base_detectors.values():
            detector.fit(X)
            
        # Получаем предсказания от всех базовых детекторов
        predictions = self._get_base_predictions(X)
        
        if y is None:
            # Если метки не предоставлены, используем консенсус базовых детекторов
            y = np.mean(predictions, axis=1) > 0.5
        
        # Преобразуем y в целочисленный тип для классификации
        y = y.astype(int)
        
        # Убеждаемся, что у нас есть оба класса для обучения
        if len(np.unique(y)) == 1:
            # Если все примеры одного класса, добавляем искусственный пример другого класса
            additional_X = predictions[0:1]  # Берем первый пример
            additional_y = 1 - y[0:1]  # Противоположный класс
            predictions = np.vstack([predictions, additional_X])
            y = np.hstack([y, additional_y])
            
        self.model.fit(predictions, y)
        return self
        
    def _get_base_predictions(self, X):
        """Получение предсказаний от всех базовых детекторов"""
        predictions = []
        for name, detector in self.base_detectors.items():
            pred = detector.predict_proba(X)
            if isinstance(pred, np.ndarray):
                predictions.append(pred * self.weights[name])
        return np.column_stack(predictions) if predictions else np.array([])
        
    def predict(self, X):
        """Предсказание аномальности"""
        predictions = self._get_base_predictions(X)
        if len(predictions) == 0:
            return np.zeros(len(X))
        return self.model.predict(predictions)
        
    def predict_proba(self, X):
        """Вероятность аномальности"""
        predictions = self._get_base_predictions(X)
        if len(predictions) == 0:
            return np.zeros(len(X))
            
        # Получаем вероятности классов
        proba = self.model.predict_proba(predictions)
        
        # Если есть только один класс, возвращаем соответствующие вероятности
        if proba.shape[1] == 1:
            return proba.ravel()
        
        # Иначе возвращаем вероятность аномального класса (класс 1)
        return proba[:, 1]
        
    def update_weights(self, new_weights):
        """Обновление весов базовых детекторов"""
        self.weights.update(new_weights) 