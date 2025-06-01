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

# Глобальные константы
ANOMALY_THRESHOLD = 0.5  # Пороговое значение вероятности для определения аномального поведения

def create_anomaly_detection_system():
    """Создание системы детектирования аномалий на основе автоэнкодеров"""
    print("\nИнициализация детекторов...")
    
    # Создаем базовые детекторы с повышенной чувствительностью
    detectors = {
        'time_activity': TimeActivityDetector(input_size=5, hidden_size=4, latent_size=2),
        'data_frequency': DataFrequencyDetector(input_size=5, hidden_size=4, latent_size=2),
        'data_volume': DataVolumeDetector(input_size=5, hidden_size=4, latent_size=2),
        'resource_access': ResourceAccessDetector(input_size=5, hidden_size=4, latent_size=2),
        'file_activity': FileActivityDetector(input_size=5, hidden_size=4, latent_size=2)
    }
    
    # Создаем ансамблевый автоэнкодер
    print("Создание ансамблевого автоэнкодера...")
    ensemble = EnsembleAutoencoder(detectors)
    return detectors, ensemble

def analyze_insider_predictions(features, system, real_insiders):
    """Анализ предсказаний детекторов для инсайдера"""
    print("\nАнализ предсказаний детекторов для инсайдера:")
    print("="*80)
    
    # Получаем индексы инсайдеров
    insider_indices = features[features['user_id'].isin(real_insiders)].index
    
    if len(insider_indices) == 0:
        print("Инсайдеры не найдены в данных")
        return
        
    # Получаем данные без столбца user_id
    X = features.drop(['user_id'], axis=1)
    
    # Для каждого инсайдера
    for idx in insider_indices:
        user_id = features.iloc[idx]['user_id']
        print(f"\nИнсайдер: {user_id}")
        print("-"*40)
        
        try:
            # Получаем предсказания всех детекторов
            detailed_predictions = system.get_detailed_predictions(X.iloc[[idx]])
            
            # Выводим результаты каждого детектора
            total_weighted_prob = 0
            total_weight = 0
            
            for name in system.detectors.keys():
                try:
                    raw_pred = detailed_predictions[f"{name}_raw"][0]
                    weighted_pred = detailed_predictions[f"{name}_weighted"][0]
                    weight = system.weights[name]
                    
                    # Нормализуем вероятности
                    raw_pred = np.clip(raw_pred, 0, 1)
                    weighted_pred = raw_pred * weight
                    
                    print(f"{name}:")
                    print(f"  Исходная вероятность: {raw_pred:.3f}")
                    print(f"  Вес детектора: {weight:.1f}")
                    print(f"  Взвешенная вероятность: {weighted_pred:.3f}")
                    print("-"*40)
                    
                    total_weighted_prob += weighted_pred
                    total_weight += weight
                except Exception as e:
                    print(f"Ошибка при обработке детектора {name}: {e}")
                    print("-"*40)
            
            # Вычисляем итоговую вероятность как среднее взвешенное
            final_prob = total_weighted_prob / total_weight if total_weight > 0 else 0
            print(f"Итоговая вероятность: {final_prob:.3f}")
            print(f"Пороговое значение: {ANOMALY_THRESHOLD:.3f}")
            print("="*80)
            
        except Exception as e:
            print(f"Ошибка при анализе инсайдера {user_id}: {e}")
            print("="*80)

def evaluate_detection(features, predictions, probabilities, loader, dataset):
    """Оценка качества обнаружения инсайдеров"""
    print("\nОценка результатов детектирования...")
    
    print("Получение списка реальных инсайдеров...")
    real_insiders = loader.insiders_data[loader.insiders_data['dataset'] == dataset]['user'].tolist()
    print(f"Найдено инсайдеров в датасете: {len(real_insiders)}")
    if real_insiders:
        print("Список инсайдеров:", real_insiders)
    
    print("\nПодготовка массива истинных меток...")
    y_true = np.zeros(len(features))
    for idx, user in tqdm(enumerate(features['user_id']), 
                         desc="Разметка пользователей", 
                         total=len(features)):
        if user in real_insiders:
            y_true[idx] = 1
            
    print("\nВычисление метрик качества...")
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
    print("="*50)
    print(f"Всего пользователей проанализировано: {len(features)}")
    print(f"Обнаружено потенциальных аномалий: {len(detected_insiders) + len(false_positives)}")
    print(f"Правильно обнаружено инсайдеров: {len(detected_insiders)}")
    print(f"Ложных срабатываний: {len(false_positives)}")
    print(f"Пропущено инсайдеров: {len(missed_insiders)}")
    print("="*50)
    
    if detected_insiders:
        print("\nОбнаруженные инсайдеры:")
        print("-"*50)
        for user, prob in detected_insiders:
            insider_info = loader.insiders_data[loader.insiders_data['user'] == user].iloc[0]
            print(f"Пользователь: {user}")
            print(f"Вероятность: {prob:.3f}")
            print(f"Сценарий: {insider_info['scenario']}")
            print(f"Период активности: {insider_info['start']} - {insider_info['end']}")
            print("-"*50)
    
    if missed_insiders:
        print("\nПропущенные инсайдеры:")
        print("-"*50)
        for user, prob in missed_insiders:
            print(f"Пользователь: {user}")
            print(f"Вероятность: {prob:.3f}")
            insider_info = loader.insiders_data[loader.insiders_data['user'] == user].iloc[0]
            print(f"Сценарий: {insider_info['scenario']}")
            print(f"Период активности: {insider_info['start']} - {insider_info['end']}")
            print("-"*50)
    
    print("\nЛожные срабатывания (топ-5 по вероятности):")
    print("-"*50)
    for user, prob in sorted(false_positives, key=lambda x: x[1], reverse=True)[:5]:
        print(f"Пользователь: {user}")
        print(f"Вероятность: {prob:.3f}")
        print("-"*50)

def main():
    start_time = time.time()
    print("\n=== Запуск системы обнаружения инсайдеров (Архитектура 2: Автоэнкодеры) ===")
    
    dataset = 'r2'  # r1, r2, r3.1

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
    
    # Анализируем предсказания для инсайдеров
    real_insiders = loader.insiders_data[loader.insiders_data['dataset'] == dataset]['user'].tolist()
    analyze_insider_predictions(features, ensemble, real_insiders)
    
    # Оцениваем качество обнаружения
    evaluate_detection(features, predictions, probabilities, loader, dataset)
    
    # Анализируем пользователей с высоким риском
    print("\nПодробный анализ пользователей высокого риска...")
    high_risk_mask = probabilities > ANOMALY_THRESHOLD
    high_risk_users = features.loc[high_risk_mask, 'user_id']
    high_risk_probs = probabilities[high_risk_mask]
    
    # Сортируем пользователей по убыванию вероятности
    high_risk_sorted = sorted(zip(high_risk_users, high_risk_probs), key=lambda x: x[1], reverse=True)
    
    print(f"\nНайдено {len(high_risk_users)} пользователей с высоким риском (p > {ANOMALY_THRESHOLD})")
    print("="*80)
    
    for idx, (user, prob) in enumerate(high_risk_sorted, 1):
        if idx == 10:
            break
        print(f"\nАнализ пользователя {user} ({idx}/{len(high_risk_sorted)}):")
        print("-"*80)
        print(f"Вероятность аномального поведения: {prob:.3f}")
        
        # Если это реальный инсайдер, показываем дополнительную информацию
        insider_info = loader.insiders_data[loader.insiders_data['user'] == user]
        if not insider_info.empty:
            info = insider_info.iloc[0]
            print("\n!!! ПОДТВЕРЖДЕННЫЙ ИНСАЙДЕР !!!")
            print("-"*40)
            print(f"Сценарий: {info['scenario']}")
            print(f"Период активности: {info['start']} - {info['end']}")
        print("="*80)
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\n=== Анализ всех датасетов завершен за {execution_time:.2f} секунд ===")

if __name__ == '__main__':
    main() 