from models.time_activity_detector import TimeActivityDetector
from models.data_frequency_detector import DataFrequencyDetector
from models.data_volume_detector import DataVolumeDetector
from models.resource_access_detector import ResourceAccessDetector
from models.file_activity_detector import FileActivityDetector
from models.ensemble_autoencoder import EnsembleAutoencoder
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_loader import DataLoader
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
import time

def create_anomaly_detection_system():
    """Создание системы детектирования аномалий на основе автоэнкодеров"""
    print("\nИнициализация детекторов...")
    
    # Создаем базовые детекторы
    detectors = {
        'time_activity': TimeActivityDetector(input_size=19, hidden_size=12, latent_size=6),
        'data_frequency': DataFrequencyDetector(input_size=19, hidden_size=12, latent_size=6),
        'data_volume': DataVolumeDetector(input_size=19, hidden_size=12, latent_size=6),
        'resource_access': ResourceAccessDetector(input_size=19, hidden_size=12, latent_size=6),
        'file_activity': FileActivityDetector(input_size=19, hidden_size=12, latent_size=6)
    }
    
    # Создаем ансамблевый автоэнкодер
    print("Создание ансамблевого автоэнкодера...")
    ensemble = EnsembleAutoencoder(detectors)
    return detectors, ensemble

def evaluate_detection(features, predictions, probabilities, loader, dataset):
    """Оценка качества обнаружения инсайдеров"""
    print("\nОценка результатов детектирования...")
    
    print("Получение списка реальных инсайдеров...")
    real_insiders = loader.insiders_data[loader.insiders_data['dataset'] == dataset]['user'].tolist()
    
    print("Подготовка массива истинных меток...")
    y_true = np.zeros(len(features))
    for idx, user in tqdm(enumerate(features['user_id']), desc="Разметка пользователей", total=len(features)):
        if user in real_insiders:
            y_true[idx] = 1
            
    print("Вычисление метрик качества...")
    y_pred = (predictions == -1).astype(int)
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print("\nМетрики качества обнаружения:")
    print(f"Precision (точность): {precision:.3f}")
    print(f"Recall (полнота): {recall:.3f}")
    print(f"F1-score: {f1:.3f}")
    
    print("\nАнализ результатов обнаружения...")
    detected_insiders = []
    missed_insiders = []
    false_positives = []
    
    for idx, (user, prob) in tqdm(enumerate(zip(features['user_id'], probabilities)), 
                                 desc="Анализ предсказаний",
                                 total=len(features)):
        if y_pred[idx] == 1:
            if user in real_insiders:
                detected_insiders.append((user, prob))
            else:
                false_positives.append((user, prob))
        elif user in real_insiders:
            missed_insiders.append((user, prob))
    
    print("\nРезультаты анализа:")
    print(f"Всего пользователей проанализировано: {len(features)}")
    print(f"Обнаружено потенциальных аномалий: {len(detected_insiders) + len(false_positives)}")
    print(f"Правильно обнаружено инсайдеров: {len(detected_insiders)}")
    print(f"Ложных срабатываний: {len(false_positives)}")
    print(f"Пропущено инсайдеров: {len(missed_insiders)}")
    
    print("\nОбнаруженные инсайдеры:")
    for user, prob in detected_insiders:
        insider_info = loader.insiders_data[
            (loader.insiders_data['dataset'] == dataset) & 
            (loader.insiders_data['user'] == user)
        ].iloc[0]
        print(f"- {user} (вероятность: {prob:.3f})")
        print(f"  Сценарий: {insider_info['scenario']}")
        print(f"  Период активности: {insider_info['start']} - {insider_info['end']}")

def main():
    start_time = time.time()
    print("\n=== Запуск системы обнаружения инсайдеров (Архитектура 2: Автоэнкодеры) ===")
    
    datasets = ['r1', 'r2', 'r3.1']
    
    for dataset in datasets:
        print(f"\n{'='*50}")
        print(f"Анализ датасета {dataset}")
        print(f"{'='*50}")
        
        print(f"\n[1/5] Загрузка данных из датасета {dataset}...")
        loader = DataLoader('dataset')
        loader.load_data(dataset)
        
        print("\n[2/5] Подготовка признаков...")
        features = loader.prepare_features()
        print(f"Извлечено {features.shape[1]} признаков для {features.shape[0]} пользователей")
        
        # Создаем систему детектирования
        print("\n[3/5] Создание системы детектирования...")
        detectors, ensemble = create_anomaly_detection_system()
        
        # Обучаем каждый детектор
        print("\n[4/5] Обучение моделей...")
        X = features.drop(['user_id'], axis=1)
        feature_dict = {name: X for name in detectors.keys()}
        
        for name, detector in detectors.items():
            print(f"Обучение детектора {name}...")
            detector.fit(feature_dict[name])
        
        # Обучаем ансамбль
        print("Обучение ансамбля...")
        ensemble.fit(feature_dict)
        
        # Получаем предсказания
        print("\n[5/5] Анализ поведения пользователей...")
        predictions = ensemble.predict(feature_dict)
        probabilities = ensemble.predict_proba(feature_dict)
        
        # Оцениваем качество обнаружения
        evaluate_detection(features, predictions, probabilities, loader, dataset)
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\n=== Анализ всех датасетов завершен за {execution_time:.2f} секунд ===")

if __name__ == '__main__':
    main() 