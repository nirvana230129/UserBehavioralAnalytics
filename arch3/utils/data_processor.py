import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data_loader import DataLoader
from arch2.models.temporal_feature_extractor import TemporalFeatureExtractor
from arch2.models.advanced_feature_engineering import AdvancedFeatureEngineer
from tqdm import tqdm
import pickle


class Arch3DataProcessor:
    def __init__(self):
        self.temporal_extractor = TemporalFeatureExtractor()
        self.feature_engineer = AdvancedFeatureEngineer()

    def _cache_path(self, dataset_name):
        return f'cache/arch3_features_{dataset_name}.pkl'

    def prepare_dataset(self, dataset_name, use_cache=True):
        cache_path = self._cache_path(dataset_name)
        if use_cache and os.path.exists(cache_path):
            print(f"  Загрузка кэша: {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        print(f"  Подготовка датасета {dataset_name}...")
        loader = DataLoader('dataset')
        loader.load_data(dataset_name)

        if loader.logon_data is None or len(loader.logon_data) == 0:
            return pd.DataFrame(), {}

        users = loader.logon_data['user'].unique()
        all_features = []
        sequences = {}

        for user in tqdm(users, desc=f"Обработка {dataset_name}"):
            temporal = self.temporal_extractor.extract_features(user, loader)
            aggregated = self._get_aggregated(user, loader)
            engineered = self.feature_engineer.engineer_features(temporal, aggregated)

            combined = {'user_id': user, **temporal, **aggregated, **engineered}
            all_features.append(combined)

            sequences[user] = self._extract_sequence(user, loader)

        features_df = pd.DataFrame(all_features).fillna(0)

        if len(features_df) > 0:
            self.feature_engineer.compute_group_statistics(features_df, loader.ldap_data)
            group_feats = []
            for _, row in features_df.iterrows():
                gf = self.feature_engineer.add_group_deviation_features(
                    row.to_dict(), row['user_id'], loader.ldap_data
                )
                group_feats.append(gf)
            if group_feats and any(group_feats):
                features_df = pd.concat([features_df, pd.DataFrame(group_feats)], axis=1)

        features_df = features_df.fillna(0)

        real_insiders = loader.insiders_data['user'].tolist() if not loader.insiders_data.empty else []
        features_df['is_insider'] = features_df['user_id'].apply(
            lambda x: 1 if x in real_insiders else 0
        )

        os.makedirs('cache', exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump((features_df, sequences), f)
        print(f"  Кэш сохранён: {cache_path}")

        return features_df, sequences

    def _get_aggregated(self, user_id, loader):
        user_psycho = loader.psychometric_data[loader.psychometric_data['user_id'] == user_id]
        psycho = user_psycho.iloc[0][['O', 'C', 'E', 'A', 'N']].to_dict() \
            if not user_psycho.empty else {'O': 0, 'C': 0, 'E': 0, 'A': 0, 'N': 0}

        user_ldap = loader.ldap_data[loader.ldap_data['user_id'] == user_id]
        is_admin = 1 if not user_ldap.empty and user_ldap.iloc[-1]['role'] == 'ITAdmin' else 0

        return {**psycho, 'is_admin': is_admin}

    def _extract_sequence(self, user_id, loader):
        logons = loader.logon_data[loader.logon_data['user'] == user_id].sort_values('date')
        sequence = []
        for _, row in logons.iterrows():
            activity = str(row.get('activity', 'Logon'))
            hour = row['date'].hour
            weekday = row['date'].weekday()
            sequence.append((activity, hour, weekday))
        return sequence

    def prepare_train_test(self, train_dataset, test_dataset, use_cache=True):
        train_result = self.prepare_dataset(train_dataset, use_cache)
        test_result = self.prepare_dataset(test_dataset, use_cache)

        train_df, train_seqs = train_result
        test_df, test_seqs = test_result

        if len(train_df) == 0 or len(test_df) == 0:
            raise ValueError("Не удалось загрузить данные")

        X_train = train_df.drop(['user_id', 'is_insider'], axis=1)
        y_train = train_df['is_insider']
        X_test = test_df.drop(['user_id', 'is_insider'], axis=1)
        y_test = test_df['is_insider']

        train_users = train_df['user_id'].tolist()
        test_users = test_df['user_id'].tolist()

        sequences_train = [train_seqs.get(u, []) for u in train_users]
        sequences_test = [test_seqs.get(u, []) for u in test_users]

        return (X_train, y_train, X_test, y_test,
                train_df, test_df,
                sequences_train, sequences_test)
