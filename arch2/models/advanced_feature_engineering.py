import pandas as pd
import numpy as np

class AdvancedFeatureEngineer:
    def __init__(self):
        self.dept_stats = {}
        
    def engineer_features(self, temporal_features, aggregated_features):
        features = {}
        
        # Соотношения
        features['device_per_logon'] = temporal_features.get('total_devices', 0) / (temporal_features.get('total_logons', 1) + 1e-6)
        features['http_per_logon'] = temporal_features.get('total_http', 0) / (temporal_features.get('total_logons', 1) + 1e-6)
        features['email_per_day'] = temporal_features.get('total_emails', 0) / (temporal_features.get('total_days', 1) + 1e-6)
        
        # Интенсивность активности
        features['activity_intensity'] = (
            temporal_features.get('total_logons', 0) + 
            temporal_features.get('total_devices', 0) + 
            temporal_features.get('total_http', 0)
        ) / (temporal_features.get('total_days', 1) + 1e-6)
        
        # Временные аномалии
        features['temporal_anomaly_score'] = (
            temporal_features.get('weekend_ratio', 0) * 2 +
            temporal_features.get('after_hours_ratio', 0) * 3 +
            temporal_features.get('night_ratio', 0) * 4 +
            temporal_features.get('atypical_time_ratio', 0) * 3
        ) / 12
        
        # Поведенческая нестабильность
        features['behavior_instability'] = (
            abs(temporal_features.get('drift_logon_rate', 1) - 1) +
            abs(temporal_features.get('drift_pc_diversity', 1) - 1) +
            temporal_features.get('burst_intensity', 0)
        ) / 3
        
        # Разнообразие активности
        features['activity_diversity'] = (
            temporal_features.get('unique_pcs', 0) +
            temporal_features.get('unique_activities', 0) +
            temporal_features.get('unique_device_types', 0)
        ) / 3
        
        # Психометрические взаимодействия
        if 'N' in aggregated_features and 'after_hours_ratio' in temporal_features:
            features['neurotic_after_hours'] = aggregated_features['N'] * temporal_features['after_hours_ratio']
        else:
            features['neurotic_after_hours'] = 0
        
        if 'O' in aggregated_features and 'activity_diversity' in features:
            features['openness_diversity'] = aggregated_features.get('O', 0) * features['activity_diversity']
        else:
            features['openness_diversity'] = 0
        
        # Роль и активность
        if 'is_admin' in aggregated_features:
            features['admin_device_usage'] = aggregated_features['is_admin'] * features['device_per_logon']
        else:
            features['admin_device_usage'] = 0
        
        return features
    
    def compute_group_statistics(self, all_features_df, ldap_data):
        dept_stats = {}
        
        if ldap_data is None or ldap_data.empty or 'role' not in ldap_data.columns:
            return dept_stats
        
        if 'total_logons' not in all_features_df.columns:
            return dept_stats
        
        for dept in ldap_data['role'].unique():
            dept_users = ldap_data[ldap_data['role'] == dept]['user_id'].unique()
            dept_features = all_features_df[all_features_df['user_id'].isin(dept_users)]
            
            if len(dept_features) > 1:
                dept_stats[dept] = {
                    'mean_logons': dept_features['total_logons'].mean(),
                    'mean_devices': dept_features['total_devices'].mean() if 'total_devices' in dept_features.columns else 0,
                    'std_logons': dept_features['total_logons'].std()
                }
        
        self.dept_stats = dept_stats
        return dept_stats
    
    def add_group_deviation_features(self, user_features, user_id, ldap_data):
        if ldap_data is None or ldap_data.empty or not self.dept_stats or 'role' not in ldap_data.columns:
            return {}
        
        user_dept_data = ldap_data[ldap_data['user_id'] == user_id]
        if user_dept_data.empty or 'role' not in user_dept_data.columns:
            return {}
        
        dept = user_dept_data.iloc[-1]['role']
        if dept not in self.dept_stats:
            return {}
        
        stats = self.dept_stats[dept]
        
        deviation_features = {
            'logon_vs_dept_mean': user_features.get('total_logons', 0) / (stats.get('mean_logons', 0) + 1e-6),
            'device_vs_dept_mean': user_features.get('total_devices', 0) / (stats.get('mean_devices', 0) + 1e-6)
        }
        
        return deviation_features
