import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .base_model import AnomalyDetector

class EnsembleDetector(AnomalyDetector):
    def __init__(self, detectors, weights=None):
        """
        Parameters:
        -----------
        detectors : dict
            Словарь детекторов {name: detector}
        weights : dict, optional
            Веса для каждого детектора {name: weight}
        """
        super().__init__()
        self.detectors = detectors
        self.weights = weights or {name: 1.0 for name in detectors.keys()}
        
    def fit(self, X, y=None):
        """Обучение всех базовых детекторов"""
        print("\nОбучение базовых детекторов...")
        for name, detector in self.detectors.items():
            print(f"Обучение {name}...")
            detector.fit(X)
        return self
        
    def _get_base_predictions(self, X):
        """Получение предсказаний от всех базовых детекторов"""
        predictions = {}
        for name, detector in self.detectors.items():
            pred = detector.predict_proba(X)
            predictions[name] = pred * self.weights[name]
        return predictions
        
    def predict_proba(self, X):
        """Вероятностные оценки аномальности"""
        base_predictions = self._get_base_predictions(X)
        
        # Взвешенное среднее всех предсказаний
        weighted_sum = np.zeros(len(X))
        total_weight = 0
        
        for name, pred in base_predictions.items():
            weight = self.weights[name]
            weighted_sum += pred * weight
            total_weight += weight
            
        return weighted_sum / total_weight if total_weight > 0 else weighted_sum
        
    def predict(self, X, threshold=0.8):
        """
        Предсказание аномальности
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Входные данные
        threshold : float, default=0.8
            Порог для определения аномалии
            
        Returns:
        --------
        numpy.ndarray
            Массив меток: -1 - аномалия, 1 - норма
        """
        probas = self.predict_proba(X)
        return np.where(probas > threshold, -1, 1)
        
    def update_weights(self, new_weights):
        """Обновление весов базовых детекторов"""
        self.weights.update(new_weights) 