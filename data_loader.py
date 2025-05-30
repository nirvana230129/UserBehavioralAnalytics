import pandas as pd
import numpy as np
from datetime import datetime
import glob
import os

class DataLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.logon_data = None
        self.device_data = None
        self.http_data = None
        self.ldap_data = None
        self.insiders_data = None
        
    def load_data(self):
        # Загрузка данных о входах/выходах
        self.logon_data = pd.read_csv(os.path.join(self.dataset_path, 'r1/logon.csv'))
        self.logon_data['date'] = pd.to_datetime(self.logon_data['date'])
        
        # Загрузка данных об устройствах
        self.device_data = pd.read_csv(os.path.join(self.dataset_path, 'r1/device.csv'))
        self.device_data['date'] = pd.to_datetime(self.device_data['date'])
        
        # Загрузка HTTP данных
        self.http_data = pd.read_csv(os.path.join(self.dataset_path, 'r1/http.csv'))
        self.http_data['date'] = pd.to_datetime(self.http_data['date'])
        
        # Загрузка LDAP данных
        ldap_files = glob.glob(os.path.join(self.dataset_path, 'r1/LDAP/*.csv'))
        ldap_dfs = []
        for file in ldap_files:
            df = pd.read_csv(file)
            month = os.path.basename(file).split('_')[1].split('.')[0]
            df['month'] = month
            ldap_dfs.append(df)
        self.ldap_data = pd.concat(ldap_dfs, ignore_index=True)
        
        # Загрузка данных об инсайдерах (истинные метки)
        self.insiders_data = pd.read_csv(os.path.join(self.dataset_path, 'answers/insiders.csv'))
        
    def prepare_features(self, start_date=None, end_date=None):
        """Подготовка признаков для обучения моделей"""
        if start_date:
            start_date = pd.to_datetime(start_date)
        if end_date:
            end_date = pd.to_datetime(end_date)
            
        features = []
        
        # Получаем список всех пользователей
        users = self.logon_data['user'].unique()
        
        for user in users:
            # Фильтруем данные по пользователю и временному окну
            user_logons = self.logon_data[
                (self.logon_data['user'] == user) &
                (self.logon_data['date'] >= start_date if start_date else True) &
                (self.logon_data['date'] <= end_date if end_date else True)
            ]
            
            user_devices = self.device_data[
                (self.device_data['user'] == user) &
                (self.device_data['date'] >= start_date if start_date else True) &
                (self.device_data['date'] <= end_date if end_date else True)
            ]
            
            user_http = self.http_data[
                (self.http_data['user'] == user) &
                (self.http_data['date'] >= start_date if start_date else True) &
                (self.http_data['date'] <= end_date if end_date else True)
            ]
            
            # Временные признаки
            logon_hours = user_logons[user_logons['activity'] == 'Logon']['date'].dt.hour
            after_hours_logons = sum((logon_hours < 8) | (logon_hours > 18))
            
            # Признаки устройств
            device_connects = len(user_devices[user_devices['activity'] == 'connect'])
            after_hours_devices = len(user_devices[
                (user_devices['activity'] == 'connect') &
                ((user_devices['date'].dt.hour < 8) | (user_devices['date'].dt.hour > 18))
            ])
            
            # HTTP признаки
            unique_domains = len(user_http['url'].apply(lambda x: x.split('/')[0]).unique())
            http_requests = len(user_http)
            
            # Признаки доступа
            unique_pcs = len(user_logons['pc'].unique())
            is_admin = any(self.ldap_data[
                (self.ldap_data['user_id'] == user) &
                (self.ldap_data['role'] == 'IT Admin')
            ])
            
            # Собираем все признаки
            user_features = {
                'user_id': user,
                'total_logons': len(user_logons[user_logons['activity'] == 'Logon']),
                'after_hours_logons': after_hours_logons,
                'device_connects': device_connects,
                'after_hours_devices': after_hours_devices,
                'unique_domains': unique_domains,
                'http_requests': http_requests,
                'unique_pcs': unique_pcs,
                'is_admin': int(is_admin),
                'logon_hour_std': logon_hours.std() if len(logon_hours) > 0 else 0,
            }
            
            # Добавляем метку аномальности (если пользователь есть в insiders_data)
            is_insider = any(self.insiders_data['username'] == user)
            user_features['is_anomaly'] = int(is_insider)
            
            features.append(user_features)
            
        return pd.DataFrame(features)

    def get_user_timeline(self, user, start_date=None, end_date=None):
        """Получение временной линии событий пользователя"""
        events = []
        
        # Добавляем события входа/выхода
        for _, row in self.logon_data[self.logon_data['user'] == user].iterrows():
            events.append({
                'date': row['date'],
                'type': 'logon',
                'activity': row['activity'],
                'pc': row['pc']
            })
            
        # Добавляем события устройств
        for _, row in self.device_data[self.device_data['user'] == user].iterrows():
            events.append({
                'date': row['date'],
                'type': 'device',
                'activity': row['activity'],
                'pc': row['pc']
            })
            
        # Добавляем HTTP события
        for _, row in self.http_data[self.http_data['user'] == user].iterrows():
            events.append({
                'date': row['date'],
                'type': 'http',
                'activity': 'visit',
                'url': row['url']
            })
            
        # Сортируем события по времени
        events = pd.DataFrame(events)
        events = events.sort_values('date')
        
        if start_date:
            events = events[events['date'] >= pd.to_datetime(start_date)]
        if end_date:
            events = events[events['date'] <= pd.to_datetime(end_date)]
            
        return events 