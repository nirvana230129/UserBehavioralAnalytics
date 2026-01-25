import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from arch2.utils.data_processor import InsiderDetectionDataProcessor
from arch2.models.lgbm_detector import LightGBMInsiderDetector
from arch2.utils.evaluation import InsiderDetectionEvaluator
import pandas as pd
import time

print("Быстрый тест архитектуры 2")
print("="*60)

dataset = 'r1'
print(f"\nПодготовка датасета {dataset} (первые 100 пользователей)...")

processor = InsiderDetectionDataProcessor()

start_time = time.time()
full_df = processor.prepare_dataset(dataset, use_cache=False)
print(f"Извлечено {len(full_df)} пользователей за {time.time() - start_time:.2f} сек")

sample_df = full_df.head(100)
print(f"Используем {len(sample_df)} пользователей для теста")

X = sample_df.drop(['user_id', 'is_insider'], axis=1)
y = sample_df['is_insider']

print(f"\nРазмерность признаков: {X.shape}")
print(f"Инсайдеров: {sum(y)}")

val_split = int(len(X) * 0.8)
X_train = X.iloc[:val_split]
y_train = y.iloc[:val_split]
X_test = X.iloc[val_split:]
y_test = y.iloc[val_split:]

print(f"\nОбучение модели...")
detector = LightGBMInsiderDetector(pos_weight=10)
detector.fit(X_train.values, y_train.values, feature_names=X.columns.tolist())

print(f"\nТоп-10 важных признаков:")
for i, (feature, importance) in enumerate(detector.get_feature_importance(top_n=10), 1):
    print(f"  {i:2d}. {feature:40s} {importance:8.1f}")

y_pred = detector.predict(X_test.values)
y_proba = detector.predict_proba(X_test.values)

print(f"\nТест завершен!")
print(f"Предсказано аномалий: {sum(y_pred)}/{len(y_pred)}")
