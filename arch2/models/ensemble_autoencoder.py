import torch
import torch.nn as nn
import numpy as np

class EnsembleAutoencoder(nn.Module):
    def __init__(self, detectors, weights=None, threshold=0.5, latent_size=32, ensemble_hidden_size=64):
        """
        Parameters:
        -----------
        detectors : dict
            Словарь детекторов {name: detector}
        weights : dict, optional
            Веса для каждого детектора {name: weight}
        threshold : float, default=0.5
            Порог для определения аномалии
        latent_size : int, default=32
            Размер латентного пространства ансамблевого автоэнкодера
        ensemble_hidden_size : int, default=64
            Размер скрытого слоя ансамблевого автоэнкодера
        """
        super(EnsembleAutoencoder, self).__init__()
        self.detectors = detectors
        self.weights = weights or {name: 1.0 for name in detectors.keys()}
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Определяем размер входного слоя как сумму латентных представлений всех детекторов
        total_latent_size = sum(detector.model.encoder[-2].out_features 
                              for detector in detectors.values())
        
        # Создаем ансамблевый автоэнкодер
        self.encoder = nn.Sequential(
            nn.Linear(total_latent_size, ensemble_hidden_size),
            nn.ReLU(),
            nn.Linear(ensemble_hidden_size, latent_size),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, ensemble_hidden_size),
            nn.ReLU(),
            nn.Linear(ensemble_hidden_size, 1),
            nn.Sigmoid()
        )
        
        self.to(self.device)
        
    def forward(self, X_dict):
        """
        Прямой проход через ансамблевый автоэнкодер
        
        Parameters:
        -----------
        X_dict : dict
            Словарь с входными данными для каждого детектора
            
        Returns:
        --------
        torch.Tensor
            Выходные значения автоэнкодера
        """
        # Получаем латентные представления от всех детекторов
        latent_features = []
        for name, detector in self.detectors.items():
            with torch.no_grad():
                X_np = X_dict[name].values if hasattr(X_dict[name], 'values') else np.array(X_dict[name])
                X = torch.FloatTensor(X_np).to(self.device)
                latent = detector.model.encode(X)
                latent_features.append(latent)
        
        # Объединяем все латентные представления
        combined_latent = torch.cat(latent_features, dim=1)
        
        # Пропускаем через ансамблевый автоэнкодер
        encoded = self.encoder(combined_latent)
        decoded = self.decoder(encoded)
        
        return decoded
    
    def fit(self, X_dict, epochs=100, batch_size=32):
        """
        Обучение ансамблевого автоэнкодера
        
        Parameters:
        -----------
        X_dict : dict
            Словарь с обучающими данными для каждого детектора
        epochs : int, default=100
            Количество эпох обучения
        batch_size : int, default=32
            Размер батча
            
        Returns:
        --------
        self : object
            Возвращает себя
        """
        print("\nОбучение базовых детекторов...")
        for name, detector in self.detectors.items():
            print(f"Обучение {name}...")
            detector.fit(X_dict[name])
            
        print("\nОбучение ансамблевого автоэнкодера...")
        optimizer = torch.optim.Adam(self.parameters())
        criterion = nn.BCELoss()
        
        # Создаем тензор меток (1 для нормальных, 0 для аномальных)
        labels = torch.ones(len(next(iter(X_dict.values()))), 1).to(self.device)
        
        self.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.forward(X_dict)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
        return self
    
    def predict(self, X_dict):
        """
        Предсказание аномалий
        
        Parameters:
        -----------
        X_dict : dict
            Словарь с данными для каждого детектора
            
        Returns:
        --------
        numpy.ndarray
            Массив меток: 1 - нормальное поведение, -1 - аномальное
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(X_dict)
            return np.where(predictions.cpu().numpy() < self.threshold, -1, 1)
    
    def predict_proba(self, X_dict):
        """
        Вероятности аномалий
        
        Parameters:
        -----------
        X_dict : dict
            Словарь с данными для каждого детектора
            
        Returns:
        --------
        numpy.ndarray
            Массив вероятностей аномальности
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(X_dict)
            return 1 - predictions.cpu().numpy()  # Инвертируем вероятности, чтобы высокие значения соответствовали аномалиям
            
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
        X_dict = {name: X for name in self.detectors.keys()}
        
        # Получаем предсказания от каждого детектора
        for name, detector in self.detectors.items():
            try:
                # Получаем сырые предсказания от детектора
                raw_pred = 1 - detector.predict_proba(X_dict[name])
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