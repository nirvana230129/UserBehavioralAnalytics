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

# Глобальные константы
TRAIN_DATASET = 'r1'  # Датасет для обучения (r1, r2, r3.1)
TEST_DATASET = 'r2'   # Датасет для тестирования (r1, r2, r3.1)

def separate(start='', symbol='=', end='\n', n=60):
    print(start + symbol * n, end=end)

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
    
    # Создаем ансамбль
    print("Создание ансамбля детекторов...")
    ensemble = EnsembleDetector(detectors)
    return ensemble

def plot_probability_distribution(probabilities, y_true, system, train_dataset, test_dataset):
    """Построение распределения вероятностей"""
    plt.figure(figsize=(12, 6))
    
    # Разделяем вероятности на два класса
    normal_probs = probabilities[y_true == -1]
    insider_probs = probabilities[y_true == 1]
    
    # Определяем количество бинов в зависимости от разброса данных
    n_bins = min(50, len(np.unique(normal_probs)))
    
    # Строим гистограмму для обычных пользователей с логарифмической шкалой
    plt.hist(normal_probs, bins=n_bins, alpha=0.5, 
             label=f'Обычные пользователи (min={min(normal_probs):.3f}, max={max(normal_probs):.3f})', 
             density=True)
    
    # Отмечаем инсайдеров вертикальными линиями
    if len(insider_probs) > 0:
        for prob in insider_probs:
            plt.axvline(x=prob, color='r', alpha=0.5, linewidth=2)
        # Добавляем фиктивную линию для легенды
        plt.axvline(x=prob, color='r', alpha=0.5, linewidth=2,
                   label=f'Инсайдеры (min={min(insider_probs):.3f}, max={max(insider_probs):.3f})')
    
    # Получаем порог из модели
    threshold = system.get_decision_threshold()
    plt.axvline(x=threshold, color='black', linestyle='--', linewidth=2,
               label=f'Порог ({threshold:.3f})')
    
    plt.xlim(0, 1)  # Устанавливаем фиксированные границы
    plt.xlabel('Вероятность аномального поведения')
    plt.ylabel('Плотность')
    plt.title('Распределение вероятностей')
    
    # Устанавливаем логарифмическую шкалу по y если есть большие различия в частотах
    if len(normal_probs) > 0:
        hist, _ = np.histogram(normal_probs, bins=n_bins)
        if max(hist) / (min(hist[hist > 0]) if any(hist > 0) else 1) > 10:
            plt.yscale('log')
            
    plt.legend()
    plt.grid(True)
    
    # Создаем директорию для графиков если её нет
    plot_dir = f'arch1/plots/{train_dataset}_{test_dataset}'
    os.makedirs(plot_dir, exist_ok=True)
    
    # Сохраняем график
    plt.savefig(f'{plot_dir}/probability_distribution.png')
    plt.close()

def plot_pr_curve(y_true, probabilities, train_dataset, test_dataset):
    """Построение PR-кривой"""
    plt.figure(figsize=(12, 8))
    
    # Вычисляем precision и recall для класса инсайдеров (1)
    precision, recall, _ = precision_recall_curve(y_true, probabilities, pos_label=1)
    
    # Вычисляем PR-AUC
    pr_auc = auc(recall, precision)
    
    # Строим кривую
    plt.plot(recall, precision, 
            label=f'PR-AUC = {pr_auc:.3f}',
            linestyle='-',
            linewidth=2)
    
    plt.xlim(0, 1)  # Устанавливаем фиксированные границы
    plt.ylim(0, 1)  # Устанавливаем фиксированные границы
    plt.xlabel('Полнота (Recall)')
    plt.ylabel('Точность (Precision)')
    plt.title('Кривая Precision-Recall')
    plt.legend(loc='best')
    plt.grid(True)
    
    # Создаем директорию для графиков если её нет
    plot_dir = f'arch1/plots/{train_dataset}_{test_dataset}'
    os.makedirs(plot_dir, exist_ok=True)
    
    # Сохраняем график
    plt.savefig(f'{plot_dir}/pr_curve.png')
    plt.close()    
    return pr_auc

def evaluate_detection(features, predictions, probabilities, loader, train_dataset, test_dataset, system):
    """Оценка качества обнаружения инсайдеров"""
    print("Оценка результатов детектирования...")
    separate(symbol='-')
    
    print("Получение списка реальных инсайдеров...")
    # Получаем список реальных инсайдеров для текущего датасета
    real_insiders = loader.insiders_data['user'].tolist()
    print(f"Найдено инсайдеров в датасете: {len(real_insiders)}")
    if real_insiders:
        print("Список инсайдеров:", real_insiders)
    
    print("\nПодготовка массива истинных меток...")
    # Создаем массив истинных меток
    y_true = np.full(len(features), -1)
    for idx, user in tqdm(enumerate(features['user_id']), 
                         desc="Разметка пользователей", 
                         total=len(features)):
        if user in real_insiders:
            y_true[idx] = 1
            
    # Выводим веса детекторов и порог
    print("\nПараметры модели:")
    separate(symbol='-')
    
    # Выводим порог принятия решения
    threshold = system.get_decision_threshold()
    print(f"Пороговое значение: {threshold:.3f}")
    
    print("\nВеса детекторов:")
    detector_weights = system.get_detector_weights()
    if detector_weights:
        for name, weight in detector_weights.items():
            print(f"{name}: {weight:.3f}")
    separate(symbol='-')
    
    # Строим графики и получаем метрики
    print("\nПостроение графиков...")
    pr_auc = plot_pr_curve(y_true, probabilities, train_dataset, test_dataset)
    plot_probability_distribution(probabilities, y_true, system, train_dataset, test_dataset)
    print(f"Графики сохранены в директории 'arch1/plots/{train_dataset}_{test_dataset}'")
    separate(symbol='-')
    
    # Вычисляем Average Precision с инвертированными вероятностями
    ap_score = average_precision_score(y_true, probabilities, average='weighted', pos_label=-1)
    
    # Вычисляем weighted метрики для учета несбалансированности
    precision = precision_score(y_true, predictions, zero_division=0, average='weighted')
    recall = recall_score(y_true, predictions, zero_division=0, average='weighted')
    f1 = f1_score(y_true, predictions, zero_division=0, average='weighted')
    
    print("\nМетрики качества обнаружения:")
    separate(symbol='-')
    print(f"PR-AUC score: {pr_auc:.3f}")
    print(f"Average Precision score: {ap_score:.3f}")
    print(f"Weighted Precision: {precision:.3f}")
    print(f"Weighted Recall: {recall:.3f}")
    print(f"Weighted F1-score: {f1:.3f}")
    separate(symbol='-')
    
    # Дополнительная информация о балансе классов
    n_total = len(y_true)
    n_insiders = np.count_nonzero(y_true == 1)
    print("\nИнформация о балансе классов:")
    separate(symbol='-')
    print(f"Всего пользователей: {n_total}")
    print(f"Инсайдеров: {n_insiders} ({(n_insiders/n_total)*100:.2f}%)")
    print(f"Обычных пользователей: {n_total - n_insiders} ({((n_total-n_insiders)/n_total)*100:.2f}%)")
    separate(symbol='-')

    # Анализ результатов
    print("\nАнализ результатов обнаружения...")
    detected_insiders = []
    missed_insiders = []
    false_positives = []
    
    for idx, (user, prob) in tqdm(enumerate(zip(features['user_id'], probabilities)), 
                                 desc="Анализ предсказаний",
                                 total=len(features)):
        if predictions[idx] == 1:  # Если обнаружена аномалия
            if user in real_insiders:
                detected_insiders.append((user, prob))
            else:
                false_positives.append((user, prob))
        elif user in real_insiders:
            missed_insiders.append((user, prob))
    
    # Вывод результатов
    print("\nРезультаты анализа:")
    separate(symbol='-')
    print(f"Всего пользователей проанализировано: {len(features)}")
    print(f"Обнаружено потенциальных аномалий: {len(detected_insiders) + len(false_positives)}")
    print(f"Правильно обнаружено инсайдеров: {len(detected_insiders)}")
    print(f"Ложных срабатываний: {len(false_positives)}")
    print(f"Пропущено инсайдеров: {len(missed_insiders)}")
    separate()
    
    # Создаем отсортированный список всех пользователей по вероятности
    all_users_sorted = sorted(zip(features['user_id'], probabilities), key=lambda x: x[1], reverse=True)
    user_rankings = {user: idx + 1 for idx, (user, _) in enumerate(all_users_sorted)}
    
    if detected_insiders:
        print("\nОбнаруженные инсайдеры:")
        separate(symbol='-')
        for user, prob in detected_insiders:
            insider_info = loader.insiders_data[loader.insiders_data['user'] == user].iloc[0]
            print(f"Пользователь: {user}")
            print(f"Вероятность: {prob:.3f}")
            print(f"Позиция в рейтинге: {user_rankings[user]} из {len(all_users_sorted)}")
            print(f"Сценарий: {insider_info['scenario']}")
            print(f"Период активности: {insider_info['start']} - {insider_info['end']}")
            separate(symbol='-')
    
    if missed_insiders:
        print("\nПропущенные инсайдеры:")
        separate(symbol='-')
        for user, prob in missed_insiders:
            print(f"Пользователь: {user}")
            print(f"Вероятность: {prob:.3f}")
            print(f"Позиция в рейтинге: {user_rankings[user]} из {len(all_users_sorted)}")
            insider_info = loader.insiders_data[loader.insiders_data['user'] == user].iloc[0]
            print(f"Сценарий: {insider_info['scenario']}")
            print(f"Период активности: {insider_info['start']} - {insider_info['end']}")
            separate(symbol='-')
    
    print("\nЛожные срабатывания (топ-5 по вероятности):")
    separate()
    for user, prob in sorted(false_positives, key=lambda x: x[1], reverse=True)[:5]:
        print(f"Пользователь: {user}")
        print(f"Вероятность: {prob:.3f}")
        separate(symbol='-', n=40)

def train_system(loader, system):
    """Обучение системы детектирования на тренировочном датасете"""
    print("\nПодготовка признаков для обучения...")
    features = loader.prepare_features()
    print(f"Извлечено {features.shape[1]} признаков для {features.shape[0]} пользователей")
    
    # Создаем массив меток для обучения
    print("\nПодготовка меток классов для обучения...")
    y_train = np.zeros(len(features))
    real_insiders = loader.insiders_data['user'].tolist()
    for idx, user in enumerate(features['user_id']):
        if user in real_insiders:
            y_train[idx] = -1  # -1 для аномалий (инсайдеров)
        else:
            y_train[idx] = 1   # 1 для нормального поведения
    
    # Обучаем систему с метками
    print("\nОбучение моделей...")
    system.fit(X := features.drop(['user_id'], axis=1), y=y_train)
    
    return system

def test_system(loader, system):
    """Тестирование системы детектирования на тестовом датасете"""
    print("\nПодготовка признаков для тестирования...")
    features = loader.prepare_features()
    print(f"Извлечено {features.shape[1]} признаков для {features.shape[0]} пользователей")
    
    # Получаем предсказания для всех пользователей
    print("\nАнализ поведения пользователей...")
    X = features.drop(['user_id'], axis=1)
    predictions = system.predict(X)
    probabilities = system.predict_proba(X)
    
    # Оцениваем качество обнаружения
    evaluate_detection(features, predictions, probabilities, loader, TRAIN_DATASET, TEST_DATASET, system)

def main():
    start_time = time.time()
    print("\n=== Запуск системы обнаружения инсайдеров ===")
    
    # Создаем систему детектирования
    print("\n[1/4] Создание системы детектирования...")
    system = create_anomaly_detection_system()
    
    # Загружаем тренировочные данные
    print(f"\n[2/4] Загрузка тренировочного датасета {TRAIN_DATASET}...")
    loader = DataLoader('dataset')
    loader.load_data(TRAIN_DATASET)
    
    # Обучаем систему
    print("\n[3/4] Обучение системы...")
    system = train_system(loader, system)
    
    # Загружаем тестовые данные и проводим тестирование
    print(f"\n[4/4] Тестирование на датасете {TEST_DATASET}...")
    loader = DataLoader('dataset')
    loader.load_data(TEST_DATASET)
    test_system(loader, system)
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\n=== Анализ завершен за {execution_time:.2f} секунд ===")

if __name__ == '__main__':
    main() 