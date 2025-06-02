import numpy as np
from .base_model import AnomalyDetector

class EnsembleDetector(AnomalyDetector):
    def __init__(self, detectors, weights=None, threshold=0.5):
        """
        Parameters:
        -----------
        detectors : dict
            Словарь детекторов {name: detector}
        weights : dict, optional
            Веса для каждого детектора {name: weight}
        threshold : float, default=0.5
            Порог для определения аномалии
        """
        super().__init__()
        self.detectors = detectors
        self.weights = weights or {name: 1.0 for name in detectors.keys()}
        self.threshold = threshold
        
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
            # Нормализуем предсказания каждого детектора в диапазон [0, 1]
            normalized_pred = np.clip(pred, 0, 1)  # Обрезаем значения до диапазона [0, 1]
            weight = self.weights[name]
            weighted_sum += normalized_pred * weight
            total_weight += weight
            
        # Нормализуем итоговые вероятности
        final_probas = weighted_sum / total_weight if total_weight > 0 else weighted_sum
        return np.clip(final_probas, 0, 1)  # Гарантируем, что итоговые вероятности в [0, 1]
        
    def predict(self, X, threshold=None):
        """
        Предсказание аномальности
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Входные данные
        threshold : float, optional
            Порог для определения аномалии. Если не указан, используется порог из конструктора
            
        Returns:
        --------
        numpy.ndarray
            Массив меток: -1 - аномалия, 1 - норма
        """
        probas = self.predict_proba(X)
        threshold = threshold if threshold is not None else self.threshold
        return np.where(probas > threshold, -1, 1)
        
    def update_weights(self, new_weights):
        """Обновление весов базовых детекторов"""
        self.weights.update(new_weights)
        
    def get_detailed_predictions(self, X):
        """
        Получение детальных предсказаний от каждого детектора
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Входные данные
            
        Returns:
        --------
        dict
            Словарь с предсказаниями каждого детектора и их взвешенными значениями
        """
        predictions = {}
        for name, detector in self.detectors.items():
            try:
                # Получаем сырые предсказания от детектора
                raw_pred = detector.predict_proba(X)
                # Нормализуем предсказания в диапазон [0, 1]
                raw_pred = np.clip(raw_pred, 0, 1)
                # Сохраняем исходную и взвешенную вероятности
                predictions[f"{name}_raw"] = raw_pred
                predictions[f"{name}_weighted"] = raw_pred * self.weights[name]
            except Exception as e:
                print(f"Ошибка при получении предсказаний от детектора {name}: {e}")
                predictions[f"{name}_raw"] = np.zeros(len(X))
                predictions[f"{name}_weighted"] = np.zeros(len(X))
        return predictions 