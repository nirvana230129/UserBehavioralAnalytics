import pandas as pd
import numpy as np
from datetime import datetime
import os
from sklearn.preprocessing import StandardScaler
import glob

class DataLoader:
    def __init__(self, base_path):
        """
        Parameters:
        -----------
        base_path : str
            Путь к директории с данными
        """
        self.base_path = base_path
        self.datasets = ['r2', 'r3.1']  # Поддерживаемые наборы данных
        self.current_dataset = None
        self.logon_data = None
        self.device_data = None
        self.http_data = None
        self.email_data = None
        self.file_data = None
        self.ldap_data = None
        self.psychometric_data = None
        self.features = None
        
    def load_data(self, dataset='r3.1'):
        """Загрузка данных из указанного набора"""
        if dataset not in self.datasets:
            raise ValueError(f"Dataset {dataset} not supported. Available datasets: {self.datasets}")
            
        self.current_dataset = dataset
        dataset_path = os.path.join(self.base_path, dataset)
        
        # Загрузка основных данных
        print(f"Loading data from {dataset}...")
        
        # Логи входа
        print("Loading logon data...")
        self.logon_data = pd.read_csv(os.path.join(dataset_path, 'logon.csv'))
        self.logon_data['date'] = pd.to_datetime(self.logon_data['date'])
        
        # Данные устройств
        print("Loading device data...")
        self.device_data = pd.read_csv(os.path.join(dataset_path, 'device.csv'))
        self.device_data['date'] = pd.to_datetime(self.device_data['date'])
        
        # HTTP данные
        print("Loading HTTP data...")
        self.http_data = pd.read_csv(os.path.join(dataset_path, 'http.csv'))
        self.http_data['date'] = pd.to_datetime(self.http_data['date'])
        
        # Email данные
        print("Loading email data...")
        self.email_data = pd.read_csv(os.path.join(dataset_path, 'email.csv'))
        self.email_data['date'] = pd.to_datetime(self.email_data['date'])
        
        # Файловые операции (только для r3.1)
        if dataset == 'r3.1':
            print("Loading file data...")
            self.file_data = pd.read_csv(os.path.join(dataset_path, 'file.csv'))
            self.file_data['date'] = pd.to_datetime(self.file_data['date'])
        
        # Психометрические данные
        print("Loading psychometric data...")
        self.psychometric_data = pd.read_csv(os.path.join(dataset_path, 'psychometric.csv'))
        
        # LDAP данные
        print("Loading LDAP data...")
        ldap_files = glob.glob(os.path.join(dataset_path, 'LDAP', '*.csv'))
        ldap_dfs = []
        for file in ldap_files:
            df = pd.read_csv(file)
            month = os.path.basename(file).split('.')[0]  # Извлекаем месяц из имени файла
            df['month'] = month
            ldap_dfs.append(df)
        self.ldap_data = pd.concat(ldap_dfs, ignore_index=True)
        
    def is_weekend(self, date):
        """Проверка является ли день выходным"""
        return date.weekday() >= 5
        
    def is_after_hours(self, date):
        """Проверка является ли время нерабочим"""
        hour = date.hour
        return hour < 7 or hour > 18
        
    def prepare_features(self):
        """Подготовка признаков для обучения"""
        features = []
        
        # Получаем список всех пользователей
        users = pd.unique(self.logon_data['user'])
        
        for user in users:
            # Фильтруем данные пользователя
            user_logons = self.logon_data[self.logon_data['user'] == user]
            user_devices = self.device_data[self.device_data['user'] == user]
            user_http = self.http_data[self.http_data['user'] == user]
            user_emails = self.email_data[
                (self.email_data['from'] == user) | 
                (self.email_data['to'].str.contains(user, na=False))
            ]
            
            # Базовые характеристики
            total_days = (user_logons['date'].max() - user_logons['date'].min()).days + 1
            
            # Характеристики входов
            logon_features = {
                'avg_logons_per_day': len(user_logons) / total_days,
                'weekend_logon_ratio': user_logons[user_logons['date'].apply(self.is_weekend)].shape[0] / len(user_logons) if len(user_logons) > 0 else 0,
                'after_hours_logon_ratio': user_logons[user_logons['date'].apply(self.is_after_hours)].shape[0] / len(user_logons) if len(user_logons) > 0 else 0,
                'unique_pcs': user_logons['pc'].nunique(),
            }
            
            # Характеристики устройств
            device_features = {
                'avg_device_usage_per_day': len(user_devices) / total_days,
                'device_usage_ratio': len(user_devices[user_devices['activity'] == 'Connect']) / total_days if total_days > 0 else 0,
            }
            
            # HTTP характеристики
            http_features = {
                'avg_http_requests_per_day': len(user_http) / total_days,
                'unique_domains': user_http['url'].apply(lambda x: x.split('/')[0]).nunique(),
            }
            
            # Email характеристики
            email_features = {
                'avg_emails_sent_per_day': len(user_emails[user_emails['from'] == user]) / total_days,
                'avg_emails_received_per_day': len(user_emails[user_emails['to'].str.contains(user, na=False)]) / total_days,
                'unique_email_contacts': pd.concat([
                    user_emails['to'].str.split(';').explode(),
                    user_emails['from']
                ]).nunique(),
            }
            
            # Дополнительные характеристики для r3.1
            if self.current_dataset == 'r3.1' and self.file_data is not None:
                user_files = self.file_data[self.file_data['user'] == user]
                file_features = {
                    'avg_file_copies_per_day': len(user_files) / total_days,
                    'unique_file_types': user_files['filename'].apply(lambda x: x.split('.')[-1] if '.' in x else '').nunique(),
                }
            else:
                file_features = {
                    'avg_file_copies_per_day': 0,
                    'unique_file_types': 0,
                }
            
            # Психометрические характеристики
            psycho_features = self.psychometric_data[self.psychometric_data['user_id'] == user].iloc[0].to_dict()
            psycho_features = {k: v for k, v in psycho_features.items() if k in ['O', 'C', 'E', 'A', 'N']}
            
            # LDAP характеристики
            latest_ldap = self.ldap_data[self.ldap_data['user_id'] == user].iloc[-1] if len(self.ldap_data[self.ldap_data['user_id'] == user]) > 0 else None
            ldap_features = {
                'is_admin': 1 if latest_ldap is not None and latest_ldap['role'] == 'ITAdmin' else 0,
            }
            
            # Объединяем все характеристики
            user_features = {
                'user_id': user,
                **logon_features,
                **device_features,
                **http_features,
                **email_features,
                **file_features,
                **psycho_features,
                **ldap_features,
            }
            
            features.append(user_features)
        
        # Создаем DataFrame
        self.features = pd.DataFrame(features)
        
        # Нормализация числовых признаков
        numeric_columns = self.features.select_dtypes(include=[np.number]).columns
        numeric_columns = numeric_columns.drop(['user_id', 'is_admin']) if 'user_id' in numeric_columns else numeric_columns
        
        scaler = StandardScaler()
        self.features[numeric_columns] = scaler.fit_transform(self.features[numeric_columns])
        
        return self.features
        
    def get_user_timeline(self, user):
        """Получение временной линии событий пользователя"""
        timeline = []
        
        # Добавляем логины
        logons = self.logon_data[self.logon_data['user'] == user]
        for _, row in logons.iterrows():
            timeline.append({
                'date': row['date'],
                'type': 'logon',
                'details': f"{row['activity']} on {row['pc']}"
            })
        
        # Добавляем использование устройств
        devices = self.device_data[self.device_data['user'] == user]
        for _, row in devices.iterrows():
            timeline.append({
                'date': row['date'],
                'type': 'device',
                'details': f"{row['activity']} on {row['pc']}"
            })
        
        # Добавляем HTTP запросы
        http = self.http_data[self.http_data['user'] == user]
        for _, row in http.iterrows():
            timeline.append({
                'date': row['date'],
                'type': 'http',
                'details': f"Visited {row['url']}"
            })
        
        # Добавляем email активность
        emails_sent = self.email_data[self.email_data['from'] == user]
        emails_received = self.email_data[self.email_data['to'].str.contains(user, na=False)]
        
        for _, row in emails_sent.iterrows():
            timeline.append({
                'date': row['date'],
                'type': 'email',
                'details': f"Sent email to {row['to']}"
            })
            
        for _, row in emails_received.iterrows():
            timeline.append({
                'date': row['date'],
                'type': 'email',
                'details': f"Received email from {row['from']}"
            })
        
        # Добавляем файловые операции для r3.1
        if self.current_dataset == 'r3.1' and self.file_data is not None:
            files = self.file_data[self.file_data['user'] == user]
            for _, row in files.iterrows():
                timeline.append({
                    'date': row['date'],
                    'type': 'file',
                    'details': f"Copied file {row['filename']} on {row['pc']}"
                })
        
        # Сортируем по времени
        timeline = pd.DataFrame(timeline)
        timeline = timeline.sort_values('date')
        
        return timeline 