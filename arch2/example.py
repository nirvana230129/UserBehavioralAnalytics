import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from arch2.utils.data_processor import InsiderDetectionDataProcessor
from arch2.models.lgbm_detector import LightGBMInsiderDetector
from arch2.utils.evaluation import InsiderDetectionEvaluator
import numpy as np
import time

TRAIN_DATASET = 'r1'
TEST_DATASET = 'r1'

def main():
    start_time = time.time()
    print("\n" + "="*60)
    print("Архитектура 2: LightGBM + Временные признаки")
    print("="*60)
    
    print(f"\n[1/4] Подготовка данных...")
    print(f"  Обучающий датасет: {TRAIN_DATASET}")
    print(f"  Тестовый датасет: {TEST_DATASET}")
    
    processor = InsiderDetectionDataProcessor()
    X_train, y_train, X_test, y_test, train_df, test_df, feature_names = processor.prepare_train_test(
        TRAIN_DATASET, TEST_DATASET, use_cache=False
    )
    
    print(f"\n  Извлечено {X_train.shape[1]} признаков")
    print(f"  Обучающая выборка: {X_train.shape[0]} пользователей ({sum(y_train)} инсайдеров)")
    print(f"  Тестовая выборка: {X_test.shape[0]} пользователей ({sum(y_test)} инсайдеров)")
    
    val_split = int(len(X_train) * 0.85)
    X_train_split = X_train.iloc[:val_split]
    y_train_split = y_train.iloc[:val_split]
    X_val = X_train.iloc[val_split:]
    y_val = y_train.iloc[val_split:]
    
    n_normal = sum(y_train_split == 0)
    n_insider = sum(y_train_split == 1)
    pos_weight = n_normal / (n_insider + 1e-6)
    
    print(f"\n[2/4] Создание модели...")
    print(f"  Вес класса инсайдеров: {pos_weight:.1f}")
    
    detector = LightGBMInsiderDetector(pos_weight=min(pos_weight, 100))
    
    print(f"\n[3/4] Обучение модели...")
    detector.fit(
        X_train_split.values, y_train_split.values,
        X_val.values if len(X_val) > 0 else None,
        y_val.values if len(y_val) > 0 else None,
        feature_names=feature_names
    )
    
    print("\nТоп-20 важных признаков:")
    for i, (feature, importance) in enumerate(detector.get_feature_importance(top_n=20), 1):
        print(f"  {i:2d}. {feature:40s} {importance:8.1f}")
    
    print(f"\n[4/4] Тестирование модели...")
    y_pred = detector.predict(X_test.values)
    y_proba = detector.predict_proba(X_test.values)
    
    evaluator = InsiderDetectionEvaluator(TRAIN_DATASET, TEST_DATASET)
    metrics = evaluator.evaluate(y_test.values, y_pred, y_proba, test_df)
    
    end_time = time.time()
    print(f"\n{'='*60}")
    print(f"Завершено за {end_time - start_time:.2f} секунд")
    print(f"{'='*60}\n")
    
    return metrics

if __name__ == '__main__':
    main()
