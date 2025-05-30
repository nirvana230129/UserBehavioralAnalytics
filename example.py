from models.time_activity_detector import TimeActivityDetector
from models.data_frequency_detector import DataFrequencyDetector
from models.data_volume_detector import DataVolumeDetector
from models.resource_access_detector import ResourceAccessDetector
from models.ensemble_detector import EnsembleDetector
from data_loader import DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve, 
    average_precision_score,
    roc_auc_score, 
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

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

def evaluate_model(y_true, y_pred, y_pred_proba):
    """Оценка модели с помощью различных метрик"""
    results = {}
    
    # Проверяем количество уникальных классов
    unique_classes = np.unique(y_true)
    
    if len(unique_classes) > 1:
        # Если есть оба класса, вычисляем все метрики
        results['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        
        # Precision-Recall AUC
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        results['pr_auc'] = auc(recall, precision)
        
        # Average Precision
        results['avg_precision'] = average_precision_score(y_true, y_pred_proba)
        
        # Метрики для порогового решения
        results['precision'] = precision_score(y_true, y_pred)
        results['recall'] = recall_score(y_true, y_pred)
        results['f1'] = f1_score(y_true, y_pred)
        
        # Матрица ошибок
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = cm[0, 0] if cm.shape == (1, 1) else 0
            fp = fn = tp = 0
            
        results['true_negatives'] = tn
        results['false_positives'] = fp
        results['false_negatives'] = fn
        results['true_positives'] = tp
        
        if (fp + tn) > 0:
            results['false_positive_rate'] = fp / (fp + tn)
        else:
            results['false_positive_rate'] = 0
            
        if (tp + fn) > 0:
            results['detection_rate'] = tp / (tp + fn)
        else:
            results['detection_rate'] = 0
    else:
        # Если только один класс
        results['warning'] = f"Только один класс в тестовой выборке: {unique_classes[0]}"
        total_samples = len(y_true)
        if unique_classes[0] == 0:  # Если все примеры нормальные
            results['true_negatives'] = sum(y_pred == 0)
            results['false_positives'] = sum(y_pred == 1)
            results['false_negatives'] = 0
            results['true_positives'] = 0
            results['precision'] = 0 if sum(y_pred == 1) == 0 else 'undefined'
            results['recall'] = 1
            results['detection_rate'] = 0
            results['false_positive_rate'] = results['false_positives'] / total_samples
        else:  # Если все примеры аномальные
            results['true_negatives'] = 0
            results['false_positives'] = 0
            results['false_negatives'] = sum(y_pred == 0)
            results['true_positives'] = sum(y_pred == 1)
            results['precision'] = 1
            results['recall'] = results['true_positives'] / total_samples
            results['detection_rate'] = results['recall']
            results['false_positive_rate'] = 0
            
    return results

def plot_precision_recall_curve(y_true, y_pred_proba):
    """Построение кривой Precision-Recall"""
    if len(np.unique(y_true)) > 1:
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, color='b', label=f'PR-AUC = {pr_auc:.3f}')
        plt.xlabel('Recall (Полнота)')
        plt.ylabel('Precision (Точность)')
        plt.title('Кривая Precision-Recall')
        plt.legend(loc='lower left')
        plt.grid(True)
        plt.savefig('precision_recall_curve.png')
        plt.close()
    else:
        print("Невозможно построить PR-кривую: в тестовой выборке только один класс")

def main():
    # Загружаем данные
    loader = DataLoader('dataset')
    loader.load_data()
    
    # Подготавливаем признаки
    features = loader.prepare_features()
    
    # Разделяем данные на обучающую и тестовую выборки
    X = features.drop(['user_id', 'is_anomaly'], axis=1)
    y = features['is_anomaly']
    
    # Стратифицированное разделение для сохранения пропорций классов
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    print("\nРаспределение классов:")
    print("Обучающая выборка:", dict(zip(*np.unique(y_train, return_counts=True))))
    print("Тестовая выборка:", dict(zip(*np.unique(y_test, return_counts=True))))
    
    # Создаем и обучаем систему детектирования
    system = create_anomaly_detection_system()
    system.fit(X_train, y_train)
    
    # Получаем предсказания
    predictions = system.predict(X_test)
    probabilities = system.predict_proba(X_test)
    
    # Оцениваем модель
    metrics = evaluate_model(y_test, predictions, probabilities)
    
    # Выводим результаты
    print("\nРезультаты оценки модели:")
    if 'warning' in metrics:
        print(f"Предупреждение: {metrics['warning']}")
        
    print("\nМатрица ошибок:")
    print(f"True Negatives (TN): {metrics['true_negatives']}")
    print(f"False Positives (FP): {metrics['false_positives']}")
    print(f"False Negatives (FN): {metrics['false_negatives']}")
    print(f"True Positives (TP): {metrics['true_positives']}")
    
    if 'roc_auc' in metrics:
        print(f"\nROC-AUC: {metrics['roc_auc']:.3f}")
        print(f"PR-AUC: {metrics['pr_auc']:.3f}")
        print(f"Average Precision: {metrics['avg_precision']:.3f}")
    
    print(f"\nМетрики для порогового решения:")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall: {metrics['recall']}")
    if 'f1' in metrics:
        print(f"F1-score: {metrics['f1']:.3f}")
    
    print(f"\nПоказатели обнаружения:")
    print(f"False Positive Rate: {metrics['false_positive_rate']:.3f}")
    print(f"Detection Rate: {metrics['detection_rate']:.3f}")
    
    # Строим кривую Precision-Recall
    plot_precision_recall_curve(y_test, probabilities)
    
    # Анализируем результаты по пользователям с высоким риском
    print("\nАнализ пользователей с высоким риском аномального поведения (p > 0.8):")
    high_risk_mask = probabilities > 0.8
    high_risk_users = features.loc[X_test.index[high_risk_mask], 'user_id']
    
    for user in high_risk_users:
        print(f"\nАнализ пользователя {user}:")
        # Получаем временную линию событий пользователя
        timeline = loader.get_user_timeline(user)
        print(f"Всего событий: {len(timeline)}")
        print("Типы событий:")
        print(timeline['type'].value_counts())
        
        # Если пользователь действительно инсайдер
        insider_info = loader.insiders_data[loader.insiders_data['user'] == user]
        if not insider_info.empty:
            print("\nПодтвержденный инсайдер!")
            for _, info in insider_info.iterrows():
                print(f"Сценарий: {info['scenario']}")
                print(f"Детали: {info['details']}")
                print(f"Период активности: {info['start']} - {info['end']}")

if __name__ == '__main__':
    main() 