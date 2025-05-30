import numpy as np
from sklearn.ensemble import IsolationForest
from .base_model import AnomalyDetector

class FileActivityDetector(AnomalyDetector):
    def __init__(self, contamination=0.1):
        """
        Parameters:
        -----------
        contamination : float, default=0.1
            Ожидаемая доля аномалий в данных
        """
        super().__init__()
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
    def extract_features(self, X):
        """
        Извлечение признаков для анализа файловых операций
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Датафрейм с признаками пользователей
            
        Returns:
        --------
        numpy.ndarray
            Матрица признаков для анализа файловых операций
        """
        features = []
        
        # Основные признаки
        if 'avg_file_copies_per_day' in X.columns:
            features.append(X['avg_file_copies_per_day'].values.reshape(-1, 1))
            
        if 'unique_file_types' in X.columns:
            features.append(X['unique_file_types'].values.reshape(-1, 1))
            
        # Комбинированные признаки с другими активностями
        if 'after_hours_logon_ratio' in X.columns and 'avg_file_copies_per_day' in X.columns:
            features.append((X['after_hours_logon_ratio'] * X['avg_file_copies_per_day']).values.reshape(-1, 1))
            
        if 'weekend_logon_ratio' in X.columns and 'avg_file_copies_per_day' in X.columns:
            features.append((X['weekend_logon_ratio'] * X['avg_file_copies_per_day']).values.reshape(-1, 1))
            
        if 'device_usage_ratio' in X.columns and 'avg_file_copies_per_day' in X.columns:
            features.append((X['device_usage_ratio'] * X['avg_file_copies_per_day']).values.reshape(-1, 1))
            
        # Если нет признаков файловой активности, возвращаем нулевую матрицу
        if not features:
            return np.zeros((len(X), 1))
            
        return np.hstack(features)
        
    def fit(self, X, y=None):
        """
        Обучение модели
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Датафрейм с признаками пользователей
        y : array-like, default=None
            Игнорируется, добавлен для совместимости
            
        Returns:
        --------
        self : object
            Возвращает себя
        """
        features = self.extract_features(X)
        self.model.fit(features)
        return self
        
    def predict(self, X):
        """
        Предсказание аномальности
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Датафрейм с признаками пользователей
            
        Returns:
        --------
        numpy.ndarray
            Массив меток: 1 - нормальное поведение, -1 - аномальное
        """
        features = self.extract_features(X)
        return self.model.predict(features)
        
    def predict_proba(self, X):
        """
        Вероятность аномальности
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Датафрейм с признаками пользователей
            
        Returns:
        --------
        numpy.ndarray
            Массив вероятностей аномальности
        """
        features = self.extract_features(X)
        # Преобразуем decision_function в вероятности
        scores = self.model.decision_function(features)
        # Нормализуем scores в диапазон [0, 1], где 1 - наиболее аномальное
        probs = 1 - (scores - scores.min()) / (scores.max() - scores.min())
        return probs 