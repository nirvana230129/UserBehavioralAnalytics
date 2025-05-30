from models.time_activity_detector import TimeActivityDetector
from models.data_frequency_detector import DataFrequencyDetector
from models.data_volume_detector import DataVolumeDetector
from models.resource_access_detector import ResourceAccessDetector
from models.geo_location_detector import GeoLocationDetector
from models.ensemble_detector import EnsembleDetector
from data_loader import DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def create_anomaly_detection_system():
    # Создаем базовые детекторы
    detectors = {
        'time_activity': TimeActivityDetector(contamination=0.1),
        'data_frequency': DataFrequencyDetector(window_size=3600, contamination=0.1),
        'data_volume': DataVolumeDetector(window_size=3600, contamination=0.1),
        'resource_access': ResourceAccessDetector(contamination=0.1)
    }
    
    # Создаем веса для каждого детектора
    weights = {
        'time_activity': 1.0,
        'data_frequency': 1.0,
        'data_volume': 1.0,
        'resource_access': 1.5  # Повышенный вес для детектора доступа к ресурсам
    }
    
    # Создаем ансамбль
    ensemble = EnsembleDetector(detectors, weights)
    return ensemble

def main():
    # Загружаем данные
    loader = DataLoader('dataset')
    loader.load_data()
    
    # Подготавливаем признаки
    features = loader.prepare_features()
    
    # Разделяем данные на обучающую и тестовую выборки
    X = features.drop(['user_id', 'is_anomaly'], axis=1)
    y = features['is_anomaly']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Создаем и обучаем систему детектирования
    system = create_anomaly_detection_system()
    system.fit(X_train, y_train)
    
    # Получаем предсказания
    predictions = system.predict(X_test)
    probabilities = system.predict_proba(X_test)
    
    # Анализируем результаты
    print("\nРезультаты анализа:")
    print(f"Всего проверено пользователей: {len(X_test)}")
    print(f"Найдено аномальных пользователей: {sum(predictions == -1)}")
    
    # Анализируем пользователей с высоким риском
    high_risk_mask = probabilities > 0.8
    high_risk_users = features.loc[X_test.index[high_risk_mask], 'user_id']
    print(f"\nПользователи с высоким риском аномального поведения (p > 0.8):")
    for user in high_risk_users:
        print(f"\nАнализ пользователя {user}:")
        # Получаем временную линию событий пользователя
        timeline = loader.get_user_timeline(user)
        print(f"Всего событий: {len(timeline)}")
        print("Типы событий:")
        print(timeline['type'].value_counts())
        
        # Если пользователь действительно инсайдер
        if any(loader.insiders_data['username'] == user):
            insider_info = loader.insiders_data[loader.insiders_data['username'] == user].iloc[0]
            print("\nПодтвержденный инсайдер!")
            print(f"Сценарий: {insider_info['scenario']}")
            print(f"Период активности: {insider_info['start']} - {insider_info['end']}")

if __name__ == '__main__':
    main() 