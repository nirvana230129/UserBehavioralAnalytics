import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_loader import DataLoader
from arch2.models.temporal_feature_extractor import TemporalFeatureExtractor
import time

print("Тест извлечения временных признаков")
print("="*60)

dataset = 'r1'
print(f"\nЗагрузка датасета {dataset}...")
loader = DataLoader('dataset')
start_time = time.time()
loader.load_data(dataset)
print(f"Загружено за {time.time() - start_time:.2f} сек")

users = loader.logon_data['user'].unique()
print(f"Найдено пользователей: {len(users)}")

extractor = TemporalFeatureExtractor()

print(f"\nИзвлечение признаков для первых 5 пользователей...")
for i, user in enumerate(users[:5]):
    print(f"  {i+1}. {user}...", end=" ")
    start_time = time.time()
    features = extractor.extract_features(user, loader)
    print(f"{len(features)} признаков за {time.time() - start_time:.2f} сек")

print("\nТест успешен!")
