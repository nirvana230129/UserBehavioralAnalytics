import pandas as pd
import numpy as np
from collections import Counter

class TemporalFeatureExtractor:
    def __init__(self, windows=[3600, 14400, 86400, 604800]):
        self.windows = windows
        
    def extract_features(self, user_id, loader):
        user_logons = loader.logon_data[loader.logon_data['user'] == user_id].sort_values('date')
        user_devices = loader.device_data[loader.device_data['user'] == user_id].sort_values('date')
        user_http = loader.http_data[loader.http_data['user'] == user_id].sort_values('date') if 'user' in loader.http_data.columns else pd.DataFrame()
        user_emails = loader.email_data[loader.email_data['from'] == user_id].sort_values('date') if not loader.email_data.empty else pd.DataFrame()
        
        features = {}
        
        if len(user_logons) == 0:
            return self._empty_features()
        
        # Временные границы
        start_date = user_logons['date'].min()
        end_date = user_logons['date'].max()
        total_seconds = (end_date - start_date).total_seconds()
        
        if total_seconds == 0:
            total_seconds = 86400
        
        # Базовые временные метрики
        features['total_days'] = (end_date - start_date).days + 1
        features['total_logons'] = len(user_logons)
        features['total_devices'] = len(user_devices)
        features['total_http'] = len(user_http)
        features['total_emails'] = len(user_emails)
        
        # Скользящие окна
        for window_sec in self.windows:
            window_name = self._window_name(window_sec)
            
            # Логины
            logon_rates = self._rolling_rate(user_logons, window_sec)
            features[f'logon_rate_mean_{window_name}'] = np.mean(logon_rates) if len(logon_rates) > 0 else 0
            features[f'logon_rate_std_{window_name}'] = np.std(logon_rates) if len(logon_rates) > 0 else 0
            features[f'logon_rate_max_{window_name}'] = np.max(logon_rates) if len(logon_rates) > 0 else 0
            
            # Устройства
            device_rates = self._rolling_rate(user_devices, window_sec)
            features[f'device_rate_mean_{window_name}'] = np.mean(device_rates) if len(device_rates) > 0 else 0
            features[f'device_rate_max_{window_name}'] = np.max(device_rates) if len(device_rates) > 0 else 0
        
        # Поведенческий дрейф
        split_point = int(len(user_logons) * 0.7)
        if split_point > 0 and split_point < len(user_logons):
            baseline = user_logons.iloc[:split_point]
            recent = user_logons.iloc[split_point:]
            
            baseline_rate = len(baseline) / ((baseline['date'].max() - baseline['date'].min()).total_seconds() / 86400 + 1)
            recent_rate = len(recent) / ((recent['date'].max() - recent['date'].min()).total_seconds() / 86400 + 1)
            
            features['drift_logon_rate'] = recent_rate / (baseline_rate + 1e-6)
            
            baseline_pc_count = baseline['pc'].nunique()
            recent_pc_count = recent['pc'].nunique()
            features['drift_pc_diversity'] = recent_pc_count / (baseline_pc_count + 1e-6)
        else:
            features['drift_logon_rate'] = 1.0
            features['drift_pc_diversity'] = 1.0
        
        # Временные паттерны
        features['weekend_ratio'] = sum(user_logons['date'].apply(lambda x: x.weekday() >= 5)) / len(user_logons)
        features['after_hours_ratio'] = sum(user_logons['date'].apply(lambda x: x.hour < 7 or x.hour > 18)) / len(user_logons)
        features['night_ratio'] = sum(user_logons['date'].apply(lambda x: x.hour >= 22 or x.hour <= 6)) / len(user_logons)
        
        # Вычисляем типичные часы работы
        if len(user_logons) > 10:
            baseline_hours = user_logons.iloc[:int(len(user_logons)*0.7)]['date'].dt.hour
            recent_hours = user_logons.iloc[int(len(user_logons)*0.7):]['date'].dt.hour
            
            typical_hours = set(baseline_hours.value_counts().nlargest(8).index)
            atypical_count = sum(h not in typical_hours for h in recent_hours)
            features['atypical_time_ratio'] = atypical_count / (len(recent_hours) + 1e-6)
        else:
            features['atypical_time_ratio'] = 0
        
        # Последовательности действий
        if len(user_logons) >= 3:
            sequences = []
            for i in range(len(user_logons) - 2):
                seq = tuple(user_logons.iloc[i:i+3]['activity'].values)
                sequences.append(seq)
            
            if sequences:
                seq_counts = Counter(sequences)
                common_threshold = max(2, len(sequences) * 0.1)
                rare_sequences = sum(1 for count in seq_counts.values() if count < common_threshold)
                features['rare_sequence_ratio'] = rare_sequences / len(seq_counts)
            else:
                features['rare_sequence_ratio'] = 0
        else:
            features['rare_sequence_ratio'] = 0
        
        # Всплески активности
        if len(user_logons) > 1:
            time_diffs = user_logons['date'].diff().dt.total_seconds().dropna()
            features['burst_intensity'] = np.std(time_diffs) / (np.mean(time_diffs) + 1e-6) if len(time_diffs) > 0 else 0
            features['min_interval'] = np.min(time_diffs) if len(time_diffs) > 0 else 0
            features['max_interval'] = np.max(time_diffs) if len(time_diffs) > 0 else 0
        else:
            features['burst_intensity'] = 0
            features['min_interval'] = 0
            features['max_interval'] = 0
        
        # Разнообразие активности
        features['unique_pcs'] = user_logons['pc'].nunique()
        features['unique_activities'] = user_logons['activity'].nunique()
        
        if len(user_devices) > 0:
            features['unique_device_types'] = user_devices['activity'].nunique()
        else:
            features['unique_device_types'] = 0
        
        return features
    
    def _window_name(self, seconds):
        if seconds < 3600:
            return f'{seconds}s'
        elif seconds < 86400:
            return f'{seconds//3600}h'
        else:
            return f'{seconds//86400}d'
    
    def _rolling_rate(self, df, window_sec):
        if len(df) == 0:
            return []
        
        df_sorted = df.sort_values('date').copy()
        df_sorted['timestamp'] = df_sorted['date'].astype('int64') // 10**9
        
        rates = []
        for idx in range(min(10, len(df_sorted))):
            current_time = df_sorted.iloc[idx]['timestamp']
            window_start = current_time - window_sec
            count = sum((df_sorted['timestamp'] >= window_start) & (df_sorted['timestamp'] <= current_time))
            rate = count / (window_sec / 3600)
            rates.append(rate)
        
        return rates
    
    def _empty_features(self):
        features = {'total_days': 0, 'total_logons': 0, 'total_devices': 0, 'total_http': 0, 'total_emails': 0}
        
        for window_sec in self.windows:
            window_name = self._window_name(window_sec)
            features[f'logon_rate_mean_{window_name}'] = 0
            features[f'logon_rate_std_{window_name}'] = 0
            features[f'logon_rate_max_{window_name}'] = 0
            features[f'device_rate_mean_{window_name}'] = 0
            features[f'device_rate_max_{window_name}'] = 0
        
        features.update({
            'drift_logon_rate': 1.0, 'drift_pc_diversity': 1.0,
            'weekend_ratio': 0, 'after_hours_ratio': 0, 'night_ratio': 0,
            'atypical_time_ratio': 0, 'rare_sequence_ratio': 0,
            'burst_intensity': 0, 'min_interval': 0, 'max_interval': 0,
            'unique_pcs': 0, 'unique_activities': 0, 'unique_device_types': 0
        })
        
        return features
