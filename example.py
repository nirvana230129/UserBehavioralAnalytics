from models.time_activity_detector import TimeActivityDetector
from models.data_frequency_detector import DataFrequencyDetector
from models.data_volume_detector import DataVolumeDetector
from models.resource_access_detector import ResourceAccessDetector
from models.file_activity_detector import FileActivityDetector
from models.ensemble_detector import EnsembleDetector
from data_loader import DataLoader
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
import time

def create_anomaly_detection_system():
    """Создание системы детектирования аномалий"""
    print("\n[1/4] Инициализация детекторов...")
    
    # Создаем базовые детекторы с повышенной чувствительностью
    detectors = {
        'time_activity': TimeActivityDetector(contamination=0.05),
        'data_frequency': DataFrequencyDetector(window_size=3600, contamination=0.05),
        'data_volume': DataVolumeDetector(window_size=3600, contamination=0.05),
        'resource_access': ResourceAccessDetector(contamination=0.05),
        'file_activity': FileActivityDetector(contamination=0.05)
    }
        
    print("Настройка весов детекторов...")
    # Создаем веса для каждого детектора
    weights = {
        'time_activity': 1.5,    # Повышенный вес для временных паттернов
        'data_frequency': 1.2,   # Повышенный вес для частоты обращений
        'data_volume': 1.2,      # Повышенный вес для объема данных
        'resource_access': 1.5,  # Повышенный вес для доступа к ресурсам
        'file_activity': 1.5     # Повышенный вес для файловых операций
    }
    
    # Создаем ансамбль
    print("Создание ансамбля детекторов...")
    ensemble = EnsembleDetector(detectors, weights)
    return ensemble

def evaluate_detection(features, predictions, probabilities, loader, dataset):
    """Оценка качества обнаружения инсайдеров"""
    print("\n[3/4] Оценка результатов детектирования...")
    
    print("Получение списка реальных инсайдеров...")
    # Получаем список реальных инсайдеров для текущего датасета
    real_insiders = loader.insiders_data[loader.insiders_data['dataset'] == dataset]['user'].tolist()
    
    print("Подготовка массива истинных меток...")
    # Создаем массив истинных меток
    y_true = np.zeros(len(features))
    for idx, user in tqdm(enumerate(features['user_id']), 
                         desc="Разметка пользователей", 
                         total=len(features)):
        if user in real_insiders:
            y_true[idx] = 1
            
    print("Вычисление метрик качества...")
    # Преобразуем предсказания из [-1, 1] в [0, 1]
    y_pred = (predictions == -1).astype(int)
    
    # Вычисляем метрики
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print("\nМетрики качества обнаружения:")
    print(f"Precision (точность): {precision:.3f}")
    print(f"Recall (полнота): {recall:.3f}")
    print(f"F1-score: {f1:.3f}")
    
    # Анализ результатов
    print("\nАнализ результатов обнаружения...")
    detected_insiders = []
    missed_insiders = []
    false_positives = []
    
    for idx, (user, prob) in tqdm(enumerate(zip(features['user_id'], probabilities)), 
                                 desc="Анализ предсказаний",
                                 total=len(features)):
        if y_pred[idx] == 1:  # Если обнаружена аномалия
            if user in real_insiders:
                detected_insiders.append((user, prob))
            else:
                false_positives.append((user, prob))
        elif user in real_insiders:
            missed_insiders.append((user, prob))
    
    # Вывод результатов
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
    
    if missed_insiders:
        print("\nПропущенные инсайдеры:")
        for user, prob in missed_insiders:
            print(f"- {user} (вероятность: {prob:.3f})")
    
    print("\nЛожные срабатывания (топ-5 по вероятности):")
    for user, prob in sorted(false_positives, key=lambda x: x[1], reverse=True)[:5]:
        print(f"- {user} (вероятность: {prob:.3f})")

def main():
    start_time = time.time()
    print("\n=== Запуск системы обнаружения инсайдеров ===")
    
    # Анализируем все доступные датасеты
    datasets = ['r1', 'r2', 'r3.1']
    
    for dataset in datasets:
        print(f"\n{'='*50}")
        print(f"Анализ датасета {dataset}")
        print(f"{'='*50}")
        
        # Загружаем данные
        print(f"\n[1/5] Загрузка данных из датасета {dataset}...")
        loader = DataLoader('dataset')
        loader.load_data(dataset)
        
        # Подготавливаем признаки
        print("\n[2/5] Подготовка признаков...")
        features = loader.prepare_features()
        print(f"Извлечено {features.shape[1]} признаков для {features.shape[0]} пользователей")
        
        # Используем только признаки без меток
        X = features.drop(['user_id'], axis=1)
        
        # Создаем систему детектирования
        print("\n[3/5] Создание системы детектирования...")
        system = create_anomaly_detection_system()
        
        # Обучаем систему
        print("\n[4/5] Обучение моделей...")
        system.fit(X)
        
        # Получаем предсказания для всех пользователей
        print("\n[5/5] Анализ поведения пользователей...")
        predictions = system.predict(X)
        probabilities = system.predict_proba(X)
        
        # Оцениваем качество обнаружения
        evaluate_detection(features, predictions, probabilities, loader, dataset)
        
        # Анализируем пользователей с высоким риском
        print("\nПодробный анализ пользователей высокого риска...")
        high_risk_mask = probabilities > 0.8
        high_risk_users = features.loc[high_risk_mask, 'user_id']
        
        print(f"\nНайдено {len(high_risk_users)} пользователей с высоким риском (p > 0.8)")
        
        for user in tqdm(high_risk_users, desc="Анализ пользователей высокого риска"):
            print(f"\nАнализ пользователя {user}:")
            # Получаем временную линию событий пользователя
            timeline = loader.get_user_timeline(user)
            print(f"Всего событий: {len(timeline)}")
            if not timeline.empty:
                print("Типы событий:")
                print(timeline['type'].value_counts())
                
                # Показываем примеры подозрительных действий
                print("\nПримеры последних действий:")
                print(timeline.tail().to_string())
                
                # Если это реальный инсайдер, показываем дополнительную информацию
                insider_info = loader.insiders_data[
                    (loader.insiders_data['dataset'] == dataset) & 
                    (loader.insiders_data['user'] == user)
                ]
                if not insider_info.empty:
                    info = insider_info.iloc[0]
                    print("\nПОДТВЕРЖДЕННЫЙ ИНСАЙДЕР!")
                    print(f"Сценарий: {info['scenario']}")
                    print(f"Период активности: {info['start']} - {info['end']}")
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\n=== Анализ всех датасетов завершен за {execution_time:.2f} секунд ===")

if __name__ == '__main__':
    main() 