import torch
import numpy as np
from .base_autoencoder import BaseAutoencoder

class DataFrequencyDetector:
    def __init__(self, input_size=24, hidden_size=12, latent_size=6, threshold=0.1, window_size=3600):
        self.model = BaseAutoencoder(input_size, hidden_size, latent_size)
        self.threshold = threshold
        self.window_size = window_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def fit(self, X):
        """Обучение модели на данных частоты активности"""
        X_np = X.values if hasattr(X, 'values') else np.array(X)
        X_tensor = torch.FloatTensor(X_np).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters())
        
        self.model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            output = self.model(X_tensor)
            loss = torch.mean((X_tensor - output) ** 2)
            loss.backward()
            optimizer.step()
            
    def predict(self, X):
        """Предсказание аномалий"""
        self.model.eval()
        with torch.no_grad():
            X_np = X.values if hasattr(X, 'values') else np.array(X)
            X_tensor = torch.FloatTensor(X_np).to(self.device)
            reconstruction_errors = self.model.get_reconstruction_error(X_tensor)
            predictions = (reconstruction_errors > self.threshold).cpu().numpy()
            return np.where(predictions, -1, 1)
            
    def predict_proba(self, X):
        """Вероятности аномалий"""
        self.model.eval()
        with torch.no_grad():
            X_np = X.values if hasattr(X, 'values') else np.array(X)
            X_tensor = torch.FloatTensor(X_np).to(self.device)
            reconstruction_errors = self.model.get_reconstruction_error(X_tensor)
            return reconstruction_errors.cpu().numpy() 