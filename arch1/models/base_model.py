from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator

class AnomalyDetector(BaseEstimator, ABC):
    def __init__(self):
        self.model = None
        self.threshold = None
    
    @abstractmethod
    def fit(self, X, y=None):
        """Обучение модели"""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Предсказание аномальности"""
        pass
    
    @abstractmethod
    def predict_proba(self, X):
        """Вероятность аномальности"""
        pass 