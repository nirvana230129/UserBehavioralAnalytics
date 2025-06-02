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
ANOMALY_THRESHOLD = 0.500  # Пороговое значение вероятности для определения аномального поведения
DATASET = 'r2'  # Используемый датасет (r1, r2, r3.1)

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
        
    print("Настройка весов детекторов...")
    # Создаем веса для каждого детектора
    weights = {
        'time_activity': 1,    # Максимальный вес для временных паттернов
        'data_frequency': 1,   # Средний вес для частоты обращений
        'data_volume': 1,      # Пониженный вес для объема данных
        'resource_access': 1,  # Низкий вес для доступа к ресурсам
        'file_activity': 1     # Минимальный вес для файловых операций
    }
    
    # Создаем ансамбль
    print("Создание ансамбля детекторов...")
    ensemble = EnsembleDetector(detectors, weights, threshold=ANOMALY_THRESHOLD)
    return ensemble

def plot_probability_distribution(probabilities, y_true, dataset):
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
    
    # Создаем директорию для графиков если её нет
    plot_dir = f'arch1/plots/{dataset}'
    os.makedirs(plot_dir, exist_ok=True)
    
    # Сохраняем график
    plt.savefig(f'{plot_dir}/probability_distribution.png')
    plt.close()

def plot_pr_curve(y_true, probabilities, dataset):
    """Построение PR-кривой"""
    plt.figure(figsize=(12, 8))
    
    # Вычисляем precision и recall
    precision, recall, _ = precision_recall_curve(y_true, probabilities, pos_label=0)
    
    # Вычисляем PR-AUC
    pr_auc = auc(recall, precision)
    
    # Строим кривую
    plt.plot(recall, precision, 
            label=f'PR-AUC = {pr_auc:.3f}',
            linestyle='-',
            linewidth=2)
        
    plt.xlabel('Полнота (Recall)')
    plt.ylabel('Точность (Precision)')
    plt.title('Кривая Precision-Recall')
    plt.legend(loc='best')
    plt.grid(True)
    
    # Создаем директорию для графиков если её нет
    plot_dir = f'arch1/plots/{dataset}'
    os.makedirs(plot_dir, exist_ok=True)
    
    # Сохраняем график
    plt.savefig(f'{plot_dir}/pr_curve.png')
    plt.close()    
    return pr_auc

def evaluate_detection(features, predictions, probabilities, loader, dataset):
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
    pr_auc = plot_pr_curve(y_true, probabilities, dataset)
    plot_probability_distribution(probabilities, y_true, dataset)
    print(f"Графики сохранены в директории 'arch1/plots/{dataset}'")
    separate(symbol='-')
    
    # Вычисляем Average Precision с инвертированными вероятностями
    ap_score = average_precision_score(y_true, probabilities, average='weighted', pos_label=0)
    
    # Вычисляем weighted метрики для учета несбалансированности
    precision = precision_score(y_true, y_pred, zero_division=0, average='weighted')
    recall = recall_score(y_true, y_pred, zero_division=0, average='weighted')
    f1 = f1_score(y_true, y_pred, zero_division=0, average='weighted')
    
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
    n_insiders = sum(y_true)
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
        if y_pred[idx] == 1:  # Если обнаружена аномалия
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

def analyze_insider_predictions(features, system, real_insiders):
    """Анализ предсказаний детекторов для инсайдера"""
    print("\nАнализ предсказаний детекторов для инсайдера:")
    separate(symbol='-')
    
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
        separate(symbol='-', n=40)
        
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
                    separate(symbol='-', n=40)
                    
                    total_weighted_prob += weighted_pred
                    total_weight += weight
                except Exception as e:
                    print(f"Ошибка при обработке детектора {name}: {e}")
                    separate(symbol='-', n=40)
            
            # Вычисляем итоговую вероятность как среднее взвешенное
            final_prob = total_weighted_prob / total_weight if total_weight > 0 else 0
            print(f"Итоговая вероятность: {final_prob:.3f}")
            print(f"Пороговое значение: {ANOMALY_THRESHOLD:.3f}")
            separate(symbol='-')
            
        except Exception as e:
            print(f"Ошибка при анализе инсайдера {user_id}: {e}")
            separate(symbol='-')

def main():
    start_time = time.time()
    print("\n=== Запуск системы обнаружения инсайдеров ===")
    
    separate('\n')
    print(f"Анализ датасета {DATASET}")
    separate()
    
    # Загружаем данные
    print(f"\n\n[1/4] Загрузка данных из датасета {DATASET}...")
    separate(end='')
    loader = DataLoader('dataset')
    loader.load_data(DATASET)
    separate()
    
    # Подготавливаем признаки
    print("\n\n[2/4] Подготовка признаков...")
    separate()
    features = loader.prepare_features()
    print(f"Извлечено {features.shape[1]} признаков для {features.shape[0]} пользователей")
    separate()
    
    # Создаем и обучаем систему детектирования
    print("\n\n[3/4] Создание и обучение системы детектирования...")
    separate(end='')
    system = create_anomaly_detection_system()
    separate()
    
    # Обучаем систему
    print("\nОбучение моделей...")
    separate(symbol='-', end='')
    system.fit(X := features.drop(['user_id'], axis=1))
    separate(symbol='-')

    # Анализируем предсказания для инсайдера
    real_insiders = loader.insiders_data['user'].tolist()
    analyze_insider_predictions(features, system, real_insiders)
    separate()

    # Получаем предсказания для всех пользователей
    print("\n\n[4/4] Анализ поведения пользователей...")
    separate()
    predictions = system.predict(X)
    probabilities = system.predict_proba(X)

    # Оцениваем качество обнаружения
    evaluate_detection(features, predictions, probabilities, loader, DATASET)
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\n=== Анализ всех датасетов завершен за {execution_time:.2f} секунд ===")

if __name__ == '__main__':
    main() 