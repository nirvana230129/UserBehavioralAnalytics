import torch
import numpy as np
from .base_autoencoder import BaseAutoencoder
from sklearn.preprocessing import StandardScaler

class ResourceAccessDetector:
    def __init__(self, input_size=24, hidden_size=12, latent_size=6, threshold=0.1):
        """
        Parameters:
        -----------
        input_size : int, default=24
            Размер входного слоя
        hidden_size : int, default=12
            Размер скрытого слоя
        latent_size : int, default=6
            Размер латентного пространства
        threshold : float, default=0.1
            Порог для определения аномалий
        """
        self.model = BaseAutoencoder(input_size, hidden_size, latent_size)
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def _prepare_features(self, X):
        """
        Подготовка признаков для анализа доступа к ресурсам
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Датафрейм с признаками
            
        Returns:
        --------
        numpy.ndarray
            Подготовленные признаки
        """
        if isinstance(X, np.ndarray):
            return X
            
        # Если есть нужные столбцы, используем их
        features = []
        
        # Основные признаки
        if 'is_admin' in X.columns:
            features.append(X['is_admin'])
        if 'unique_pcs' in X.columns:
            features.append(X['unique_pcs'])
        if 'device_usage_ratio' in X.columns:
            features.append(X['device_usage_ratio'])
        if 'avg_logons_per_day' in X.columns:
            features.append(X['avg_logons_per_day'])
        if 'after_hours_logon_ratio' in X.columns:
            features.append(X['after_hours_logon_ratio'])
            
        if features:
            features = np.column_stack(features)
        else:
            # Если нет нужных столбцов, используем все числовые признаки
            features = X.select_dtypes(include=[np.number]).values[:, :5]  # Берем первые 5 признаков
            
        # Нормализуем признаки
        if not self.is_fitted:
            self.is_fitted = True
            return self.scaler.fit_transform(features)
        return self.scaler.transform(features)
        
    def fit(self, X):
        """
        Обучение модели на данных доступа к ресурсам
        
        Parameters:
        -----------
        X : pandas.DataFrame или numpy.ndarray
            Обучающие данные
            
        Returns:
        --------
        self : object
            Возвращает себя
        """
        X_np = self._prepare_features(X)
        X_tensor = torch.FloatTensor(X_np).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters())
        
        self.model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            output = self.model(X_tensor)
            loss = torch.mean((X_tensor - output) ** 2)
            loss.backward()
            optimizer.step()
            
        return self
            
    def predict(self, X):
        """
        Предсказание аномалий
        
        Parameters:
        -----------
        X : pandas.DataFrame или numpy.ndarray
            Данные для предсказания
            
        Returns:
        --------
        numpy.ndarray
            Массив меток: 1 - нормальное поведение, -1 - аномальное
        """
        self.model.eval()
        with torch.no_grad():
            X_np = self._prepare_features(X)
            X_tensor = torch.FloatTensor(X_np).to(self.device)
            reconstruction_errors = self.model.get_reconstruction_error(X_tensor)
            predictions = (reconstruction_errors > self.threshold).cpu().numpy()
            return np.where(predictions, -1, 1)
            
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
            Массив вероятностей аномальности
        """
        self.model.eval()
        with torch.no_grad():
            X_np = self._prepare_features(X)
            X_tensor = torch.FloatTensor(X_np).to(self.device)
            reconstruction_errors = self.model.get_reconstruction_error(X_tensor)
            # Нормализуем ошибки реконструкции в диапазон [0, 1]
            errors = reconstruction_errors.cpu().numpy()
            return errors / np.max(errors) if len(errors) > 0 else errors 