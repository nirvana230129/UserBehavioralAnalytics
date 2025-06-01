from models.time_activity_detector import TimeActivityDetector
from models.data_frequency_detector import DataFrequencyDetector
from models.data_volume_detector import DataVolumeDetector
from models.resource_access_detector import ResourceAccessDetector
from models.file_activity_detector import FileActivityDetector
from models.ensemble_detector import EnsembleDetector
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_loader import DataLoader
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, average_precision_score, auc
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Глобальные константы
ANOMALY_THRESHOLD = 0.806  # Пороговое значение вероятности для определения аномального поведения

def create_anomaly_detection_system():
    """Создание системы детектирования аномалий"""
    print("\nИнициализация детекторов...")
    
    # Создаем базовые детекторы с повышенной чувствительностью
    detectors = {
        'time_activity': TimeActivityDetector(contamination=0.01),
        'data_frequency': DataFrequencyDetector(window_size=1800, contamination=0.01),
        'data_volume': DataVolumeDetector(window_size=1800, contamination=0.01),
        'resource_access': ResourceAccessDetector(contamination=0.01),
        'file_activity': FileActivityDetector(contamination=0.01)
    }
        
    print("Настройка весов детекторов...")
    # Создаем веса для каждого детектора
    weights = {
        'time_activity': 1.5,    # Максимальный вес для временных паттернов
        'data_frequency': 1.8,   # Средний вес для частоты обращений
        'data_volume': 1.2,      # Пониженный вес для объема данных
        'resource_access': 0.4,  # Низкий вес для доступа к ресурсам
        'file_activity': 0.6     # Минимальный вес для файловых операций
    }
    
    # Создаем ансамбль
    print("Создание ансамбля детекторов...")
    ensemble = EnsembleDetector(detectors, weights, threshold=ANOMALY_THRESHOLD)
    return ensemble

def plot_probability_distribution(probabilities, y_true):
    """Построение распределения вероятностей"""
    plt.figure(figsize=(12, 6))
    
    # Разделяем вероятности на два класса
    normal_probs = probabilities[y_true == 0]
    insider_probs = probabilities[y_true == 1]
    
    # Строим гистограмму для обычных пользователей
    plt.hist(normal_probs, bins=50, alpha=0.5, 
             label=f'Обычные пользователи (min={min(normal_probs):.3f}, max={max(normal_probs):.3f})', 
             density=True)
    
    # Отмечаем инсайдеров вертикальными линиями
    if len(insider_probs) > 0:
        for prob in insider_probs:
            plt.axvline(x=prob, color='r', alpha=0.5)
        # Добавляем фиктивную линию для легенды
        plt.axvline(x=prob, color='r', alpha=0.5, 
                   label=f'Инсайдеры (min={min(insider_probs):.3f}, max={max(insider_probs):.3f})')
    
    plt.axvline(x=ANOMALY_THRESHOLD, color='black', linestyle='--', label=f'Порог ({ANOMALY_THRESHOLD})')
    
    plt.xlim(0, 1)  # Устанавливаем фиксированные границы
    plt.xlabel('Вероятность аномального поведения')
    plt.ylabel('Плотность')
    plt.title('Распределение вероятностей по классам')
    plt.legend()
    plt.grid(True)
    
    # Сохраняем график
    plt.savefig('arch1/plots/probability_distribution.png')
    plt.close()

def evaluate_detection(features, predictions, probabilities, loader, dataset):
    """Оценка качества обнаружения инсайдеров"""
    print("\nОценка результатов детектирования...")
    
    print("Получение списка реальных инсайдеров...")
    # Получаем список реальных инсайдеров для текущего датасета
    real_insiders = loader.insiders_data['user'].tolist()
    print(f"Найдено инсайдеров в датасете: {len(real_insiders)}")
    if real_insiders:
        print("Список инсайдеров:", real_insiders)
    
    print("\nПодготовка массива истинных меток...")
    # Создаем массив истинных меток
    y_true = np.zeros(len(features))
    for idx, user in tqdm(enumerate(features['user_id']), 
                         desc="Разметка пользователей", 
                         total=len(features)):
        if user in real_insiders:
            y_true[idx] = 1
            
    print("\nВычисление метрик качества...")
    
    # Для метрик на основе порога используем предсказания
    y_pred = (predictions == -1).astype(int)  # -1 это аномалия, поэтому сравниваем с -1
    
    # Строим графики и получаем метрики
    print("\nПостроение графиков...")
    pr_auc = plot_pr_curve(y_true, probabilities)
    plot_probability_distribution(probabilities, y_true)
    print("Графики сохранены в файлы 'arch1/plots/pr_curve.png' и 'arch1/plots/probability_distribution.png'")
    
    # Вычисляем Average Precision с инвертированными вероятностями
    ap_score = average_precision_score(y_true, 1 - probabilities, average='weighted')
    
    # Вычисляем weighted метрики для учета несбалансированности
    precision = precision_score(y_true, y_pred, zero_division=0, average='weighted')
    recall = recall_score(y_true, y_pred, zero_division=0, average='weighted')
    f1 = f1_score(y_true, y_pred, zero_division=0, average='weighted')
    
    print("\nМетрики качества обнаружения:")
    print("-"*50)
    print("Метрики для несбалансированных классов:")
    print(f"PR-AUC score: {pr_auc:.3f}")
    print(f"Average Precision score: {ap_score:.3f}")
    
    print(f"\nМетрики на пороге {ANOMALY_THRESHOLD}:")
    print(f"Weighted Precision: {precision:.3f}")
    print(f"Weighted Recall: {recall:.3f}")
    print(f"Weighted F1-score: {f1:.3f}")
    
    # Дополнительная информация о балансе классов
    n_total = len(y_true)
    n_insiders = sum(y_true)
    print("\nИнформация о балансе классов:")
    print(f"Всего пользователей: {n_total}")
    print(f"Инсайдеров: {n_insiders} ({(n_insiders/n_total)*100:.2f}%)")
    print(f"Обычных пользователей: {n_total - n_insiders} ({((n_total-n_insiders)/n_total)*100:.2f}%)")
    
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
    print("="*50)
    print(f"Всего пользователей проанализировано: {len(features)}")
    print(f"Обнаружено потенциальных аномалий: {len(detected_insiders) + len(false_positives)}")
    print(f"Правильно обнаружено инсайдеров: {len(detected_insiders)}")
    print(f"Ложных срабатываний: {len(false_positives)}")
    print(f"Пропущено инсайдеров: {len(missed_insiders)}")
    print("="*50)
    
    # Создаем отсортированный список всех пользователей по вероятности
    all_users_sorted = sorted(zip(features['user_id'], probabilities), key=lambda x: x[1], reverse=True)
    user_rankings = {user: idx + 1 for idx, (user, _) in enumerate(all_users_sorted)}
    
    if detected_insiders:
        print("\nОбнаруженные инсайдеры:")
        print("-"*50)
        for user, prob in detected_insiders:
            insider_info = loader.insiders_data[loader.insiders_data['user'] == user].iloc[0]
            print(f"Пользователь: {user}")
            print(f"Вероятность: {prob:.3f}")
            print(f"Позиция в рейтинге: {user_rankings[user]} из {len(all_users_sorted)}")
            print(f"Сценарий: {insider_info['scenario']}")
            print(f"Период активности: {insider_info['start']} - {insider_info['end']}")
            print("-"*50)
    
    if missed_insiders:
        print("\nПропущенные инсайдеры:")
        print("-"*50)
        for user, prob in missed_insiders:
            print(f"Пользователь: {user}")
            print(f"Вероятность: {prob:.3f}")
            print(f"Позиция в рейтинге: {user_rankings[user]} из {len(all_users_sorted)}")
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

def main():
    start_time = time.time()
    print("\n=== Запуск системы обнаружения инсайдеров ===")
    
    # Анализируемый датасет
    dataset = 'r2'  # r1, r2, r3.1
    
    # for dataset in datasets:
    print(f"\n{'='*50}")
    print(f"Анализ датасета {dataset}")
    print(f"{'='*50}")
    
    # Загружаем данные
    print(f"\n[1/4] Загрузка данных из датасета {dataset}...")
    loader = DataLoader('dataset')
    loader.load_data(dataset)
    
    # Подготавливаем признаки
    print("\n[2/4] Подготовка признаков...")
    features = loader.prepare_features()
    print(f"Извлечено {features.shape[1]} признаков для {features.shape[0]} пользователей")
    
    # Создаем и обучаем систему детектирования
    print("\n[3/4] Создание и обучение системы детектирования...")
    system = create_anomaly_detection_system()
    
    # Обучаем систему
    print("\nОбучение моделей...")
    system.fit(X := features.drop(['user_id'], axis=1))
    
    # Анализируем предсказания для инсайдера
    real_insiders = loader.insiders_data['user'].tolist()
    analyze_insider_predictions(features, system, real_insiders)
    
    # Получаем предсказания для всех пользователей
    print("\n[4/4] Анализ поведения пользователей...")
    predictions = system.predict(X)
    probabilities = system.predict_proba(X)
    
    # Оцениваем качество обнаружения
    evaluate_detection(features, predictions, probabilities, loader, dataset)
    
    # Анализируем пользователей с высоким риском
    print("\nПодробный анализ пользователей высокого риска...")
    high_risk_mask = probabilities > ANOMALY_THRESHOLD  # Порог для определения аномалий
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