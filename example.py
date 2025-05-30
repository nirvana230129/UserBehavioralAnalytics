from models.time_activity_detector import TimeActivityDetector
from models.data_frequency_detector import DataFrequencyDetector
from models.data_volume_detector import DataVolumeDetector
from models.resource_access_detector import ResourceAccessDetector
from models.geo_location_detector import GeoLocationDetector
from models.ensemble_detector import EnsembleDetector

def create_anomaly_detection_system(geoip_db_path, known_good_ips=None):
    # Создаем базовые детекторы
    detectors = {
        'time_activity': TimeActivityDetector(contamination=0.1),
        'data_frequency': DataFrequencyDetector(window_size=3600, contamination=0.1),
        'data_volume': DataVolumeDetector(window_size=3600, contamination=0.1),
        'resource_access': ResourceAccessDetector(contamination=0.1),
        'geo_location': GeoLocationDetector(geoip_db_path, known_good_ips, contamination=0.1)
    }
    
    # Создаем веса для каждого детектора
    weights = {
        'time_activity': 1.0,
        'data_frequency': 1.0,
        'data_volume': 1.0,
        'resource_access': 1.0,
        'geo_location': 1.5  # Повышенный вес для географического детектора
    }
    
    # Создаем ансамбль
    ensemble = EnsembleDetector(detectors, weights)
    return ensemble

def main():
    # Пример использования
    import pandas as pd
    
    # Загружаем данные (предполагается, что у нас есть подготовленный датасет)
    data = pd.read_csv('user_activity_data.csv')
    
    # Создаем систему детектирования
    system = create_anomaly_detection_system(
        geoip_db_path='path/to/GeoLite2-City.mmdb',
        known_good_ips={'192.168.1.1', '10.0.0.1'}
    )
    
    # Обучаем систему
    system.fit(data)
    
    # Получаем предсказания для новых данных
    new_data = pd.read_csv('new_activity_data.csv')
    predictions = system.predict(new_data)
    probabilities = system.predict_proba(new_data)
    
    # Анализируем результаты
    anomalies = new_data[predictions == -1]  # -1 означает аномалию
    print(f"Найдено {len(anomalies)} аномальных событий")
    
    # Можно также получить вероятности аномальности для каждого события
    new_data['anomaly_probability'] = probabilities
    high_risk_events = new_data[probabilities > 0.8]
    print(f"Найдено {len(high_risk_events)} событий с высоким риском (p > 0.8)")

if __name__ == '__main__':
    main() 