import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from .base_model import AnomalyDetector

class EnsembleDetector(AnomalyDetector):
    def __init__(self, detectors):
        """
        Parameters:
        -----------
        detectors : dict
            Словарь детекторов {name: detector}
        """
        super().__init__()
        self.detectors = detectors
        
        # Базовая мета-модель
        self.base_meta_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            class_weight='balanced'
        )
        
        # Калиброванная мета-модель
        self.meta_model = None
        self.scaler = StandardScaler()
        self.threshold = 0.5  # Значение по умолчанию
        
    def get_detector_weights(self):
        """
        Получение весов важности каждого детектора
        
        Returns:
        --------
        dict
            Словарь {detector_name: importance_weight}
        """
        if not hasattr(self.base_meta_model, 'feature_importances_'):
            return None
            
        weights = {}
        detector_names = list(self.detectors.keys())
        
        for name, importance in zip(detector_names, self.base_meta_model.feature_importances_):
            weights[name] = importance
            
        return weights
        
    def get_decision_threshold(self):
        """
        Получение порога принятия решения
        
        Returns:
        --------
        float
            Оптимальный порог для классификации
        """
        return self.threshold
        
    def _get_base_predictions(self, X):
        """Получение предсказаний от всех базовых детекторов"""
        predictions = {}
        for name, detector in self.detectors.items():
            pred = detector.predict_proba(X)
            # Заменяем nan на 0
            pred = np.nan_to_num(pred, nan=0.0)
            predictions[name] = pred
        return predictions
        
    def _prepare_meta_features(self, X):
        """Подготовка признаков для мета-модели"""
        base_predictions = self._get_base_predictions(X)
        
        # Собираем все предсказания в одну матрицу
        meta_features = np.column_stack([
            pred for pred in base_predictions.values()
        ])
        
        # Заменяем nan на 0
        meta_features = np.nan_to_num(meta_features, nan=0.0)
        return meta_features
        
    def fit(self, X, y=None):
        """
        Обучение всех базовых детекторов и мета-модели
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Входные данные
        y : numpy.ndarray, optional
            Метки классов (1 - нормальное поведение, -1 - аномальное)
        """
        print("\nОбучение базовых детекторов...")
        # Сначала обучаем базовые детекторы
        for name, detector in self.detectors.items():
            print(f"Обучение {name}...")
            detector.fit(X)
            
        # Получаем предсказания базовых детекторов
        meta_features = self._prepare_meta_features(X)
        
        # Если есть метки классов, обучаем мета-модель
        if y is not None:
            print("Обучение мета-модели...")
            # Нормализуем признаки
            meta_features = self.scaler.fit_transform(meta_features)
            # Преобразуем метки в бинарный формат (0 для аномалий, 1 для нормального поведения)
            y_binary = (y == 1).astype(int)  # Инвертируем логику: 1 (нормальное) -> 1, -1 (аномалия) -> 0
            
            # Проверяем количество классов и примеров в каждом классе
            n_classes = len(np.unique(y_binary))
            class_counts = np.bincount(y_binary)
            min_samples = min(class_counts) if len(class_counts) > 1 else 0
            
            if n_classes == 1 or min_samples < 2:
                print("Недостаточно данных для калибровки, используем базовую модель...")
                # Если только один класс или слишком мало примеров, используем базовую модель без калибровки
                self.base_meta_model.fit(meta_features, y_binary)
                self.meta_model = self.base_meta_model
                # Устанавливаем консервативный порог для редких событий
                self.threshold = 0.7 if min_samples > 0 else 0.9
            else:
                # Обучаем базовую мета-модель
                self.base_meta_model.fit(meta_features, y_binary)
                
                try:
                    # Пытаемся выполнить калибровку
                    print("Калибровка вероятностей...")
                    
                    # Определяем количество фолдов на основе размера минимального класса
                    n_splits = min(min_samples, 5)  # Максимум 5 фолдов
                    
                    # Используем стратифицированную кросс-валидацию
                    from sklearn.model_selection import StratifiedKFold
                    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                    
                    self.meta_model = CalibratedClassifierCV(
                        self.base_meta_model,
                        cv=cv,
                        method='sigmoid',
                        n_jobs=-1
                    )
                    self.meta_model.fit(meta_features, y_binary)
                except Exception as e:
                    print(f"Ошибка при калибровке: {e}")
                    print("Используем базовую модель без калибровки...")
                    self.meta_model = self.base_meta_model
                
                # Устанавливаем порог
                self.threshold = 0.7
        return self
        
    def predict_proba(self, X):
        """
        Вероятностные оценки аномальности
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Входные данные
            
        Returns:
        --------
        numpy.ndarray
            Массив вероятностей аномальности
        """
        # Получаем предсказания базовых детекторов
        meta_features = self._prepare_meta_features(X)
        
        # Если мета-модель обучена, используем её
        if self.meta_model is not None:
            # Нормализуем признаки
            meta_features = self.scaler.transform(meta_features)
            
            try:
                # Пробуем получить вероятности
                if hasattr(self.meta_model, 'predict_proba'):
                    probas = self.meta_model.predict_proba(meta_features)
                    if probas.shape[1] == 2:
                        # Инвертируем вероятности, чтобы высокие значения соответствовали аномалиям
                        return 1 - probas[:, 1]
                
                # Если не получилось, используем решающую функцию
                if hasattr(self.meta_model, 'decision_function'):
                    scores = self.meta_model.decision_function(meta_features)
                    # Инвертируем сигмоиду
                    return 1 - (1 / (1 + np.exp(-scores)))
                
            except Exception as e:
                print(f"Ошибка при получении вероятностей: {e}")
                # В случае ошибки используем среднее по базовым детекторам
                return np.mean(meta_features, axis=1)
        
        # Если мета-модель не обучена или возникла ошибка
        probas = np.mean(meta_features, axis=1)
        
        # Заменяем nan на 0
        probas = np.nan_to_num(probas, nan=0.0)
        return np.clip(probas, 0, 1)
        
    def predict(self, X):
        """
        Предсказание аномальности
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Входные данные
            
        Returns:
        --------
        numpy.ndarray
            Массив меток: -1 - аномалия, 1 - норма
        """
        probas = self.predict_proba(X)
        # Инвертируем логику: теперь высокая вероятность = аномальное поведение
        return np.where(probas >= self.threshold, 1, -1)
        
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
            Словарь с предсказаниями каждого детектора
        """
        predictions = {}
        
        # Получаем веса детекторов
        detector_weights = self.get_detector_weights()
        
        for name, detector in self.detectors.items():
            try:
                # Получаем сырые предсказания от детектора
                raw_pred = detector.predict_proba(X)
                # Нормализуем предсказания в диапазон [0, 1]
                raw_pred = np.clip(raw_pred, 0, 1)
                # Сохраняем вероятности
                predictions[f"{name}_raw"] = raw_pred
                # Добавляем вес детектора если есть
                if detector_weights is not None:
                    predictions[f"{name}_weight"] = detector_weights[name]
            except Exception as e:
                print(f"Ошибка при получении предсказаний от детектора {name}: {e}")
                predictions[f"{name}_raw"] = np.zeros(len(X))
                if detector_weights is not None:
                    predictions[f"{name}_weight"] = 0.0
        
        # Добавляем предсказания мета-модели
        if self.meta_model is not None:
            meta_features = self._prepare_meta_features(X)
            meta_features = self.scaler.transform(meta_features)
            probas = self.meta_model.predict_proba(meta_features)
            if probas.shape[1] == 1:
                predictions["meta_model"] = np.zeros(len(X))
            else:
                # Инвертируем вероятности для мета-модели
                predictions["meta_model"] = 1 - probas[:, 1]
            
        # Добавляем порог принятия решения
        predictions["decision_threshold"] = self.threshold
            
        return predictions 