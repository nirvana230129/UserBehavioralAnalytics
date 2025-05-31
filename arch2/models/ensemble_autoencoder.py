import torch
import torch.nn as nn
import numpy as np

class EnsembleAutoencoder(nn.Module):
    def __init__(self, detectors, latent_size=32, ensemble_hidden_size=64):
        super(EnsembleAutoencoder, self).__init__()
        self.detectors = detectors
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
        """Обучение ансамблевого автоэнкодера"""
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
    
    def predict(self, X_dict):
        """Предсказание аномалий"""
        self.eval()
        with torch.no_grad():
            predictions = self.forward(X_dict)
            return np.where(predictions.cpu().numpy() < 0.5, -1, 1)
    
    def predict_proba(self, X_dict):
        """Вероятности аномалий"""
        self.eval()
        with torch.no_grad():
            predictions = self.forward(X_dict)
            return 1 - predictions.cpu().numpy()  # Инвертируем вероятности, чтобы высокие значения соответствовали аномалиям 