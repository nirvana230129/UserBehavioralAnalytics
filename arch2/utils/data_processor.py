import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from data_loader import DataLoader
from arch2.models.temporal_feature_extractor import TemporalFeatureExtractor
from arch2.models.advanced_feature_engineering import AdvancedFeatureEngineer
from tqdm import tqdm

class InsiderDetectionDataProcessor:
    def __init__(self):
        self.temporal_extractor = TemporalFeatureExtractor()
        self.feature_engineer = AdvancedFeatureEngineer()
        
    def prepare_dataset(self, dataset_name, use_cache=True):
        cache_path = f'cache/arch2_features_{dataset_name}.pkl'
        
        if use_cache and os.path.exists(cache_path):
            print(f"Загрузка кэшированных признаков из {cache_path}")
            features_df = pd.read_pickle(cache_path)
            return features_df
        
        print(f"Подготовка датасета {dataset_name}")
        loader = DataLoader('dataset')
        loader.load_data(dataset_name)
        
        if loader.logon_data is None or len(loader.logon_data) == 0:
            print(f"Нет данных для датасета {dataset_name}")
            return pd.DataFrame()
        
        users = loader.logon_data['user'].unique()
        all_features = []
        
        print("Извлечение временных признаков...")
        for user in tqdm(users, desc="Обработка пользователей"):
            temporal_features = self.temporal_extractor.extract_features(user, loader)
            
            aggregated_features = self._get_aggregated_features(user, loader)
            
            engineered_features = self.feature_engineer.engineer_features(
                temporal_features, aggregated_features
            )
            
            combined_features = {
                'user_id': user,
                **temporal_features,
                **aggregated_features,
                **engineered_features
            }
            
            all_features.append(combined_features)
        
        features_df = pd.DataFrame(all_features)
        
        if len(features_df) > 0:
            self.feature_engineer.compute_group_statistics(features_df, loader.ldap_data)
            
            group_features = []
            for _, row in features_df.iterrows():
                dev_features = self.feature_engineer.add_group_deviation_features(
                    row.to_dict(), row['user_id'], loader.ldap_data
                )
                group_features.append(dev_features)
            
            if group_features and any(group_features):
                group_df = pd.DataFrame(group_features)
                features_df = pd.concat([features_df, group_df], axis=1)
        
        features_df = features_df.fillna(0)
        
        real_insiders = loader.insiders_data['user'].tolist() if not loader.insiders_data.empty else []
        features_df['is_insider'] = features_df['user_id'].apply(lambda x: 1 if x in real_insiders else 0)
        
        os.makedirs('cache', exist_ok=True)
        features_df.to_pickle(cache_path)
        print(f"Признаки сохранены в {cache_path}")
        
        return features_df
    
    def _get_aggregated_features(self, user_id, loader):
        user_psycho = loader.psychometric_data[loader.psychometric_data['user_id'] == user_id]
        if not user_psycho.empty:
            psycho_features = user_psycho.iloc[0][['O', 'C', 'E', 'A', 'N']].to_dict()
        else:
            psycho_features = {'O': 0, 'C': 0, 'E': 0, 'A': 0, 'N': 0}
        
        user_ldap = loader.ldap_data[loader.ldap_data['user_id'] == user_id]
        ldap_features = {
            'is_admin': 1 if not user_ldap.empty and user_ldap.iloc[-1]['role'] == 'ITAdmin' else 0
        }
        
        return {**psycho_features, **ldap_features}
    
    def prepare_train_test(self, train_dataset, test_dataset, use_cache=True):
        train_df = self.prepare_dataset(train_dataset, use_cache)
        test_df = self.prepare_dataset(test_dataset, use_cache)
        
        if len(train_df) == 0 or len(test_df) == 0:
            raise ValueError("Не удалось загрузить данные для обучения или тестирования")
        
        if 'user_id' not in train_df.columns or 'is_insider' not in train_df.columns:
            raise ValueError(f"Отсутствуют необходимые колонки в train_df. Доступные: {train_df.columns.tolist()}")
        
        X_train = train_df.drop(['user_id', 'is_insider'], axis=1)
        y_train = train_df['is_insider']
        
        X_test = test_df.drop(['user_id', 'is_insider'], axis=1)
        y_test = test_df['is_insider']
        
        common_features = list(set(X_train.columns) & set(X_test.columns))
        X_train = X_train[common_features]
        X_test = X_test[common_features]
        
        return X_train, y_train, X_test, y_test, train_df, test_df, common_features
