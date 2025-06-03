import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

class DataLoader:
    def __init__(self, base_path):
        """
        Parameters:
        -----------
        base_path : str
            Путь к директории с данными
        """
        self.base_path = base_path
        self.datasets = ['r1', 'r2', 'r3.1']
        self.current_dataset = None
        self.logon_data = None
        self.device_data = None
        self.http_data = None
        self.email_data = None
        self.file_data = None
        self.ldap_data = None
        self.psychometric_data = None
        self.features = None
        self.insiders_data = None
        self.cache_dir = 'cache'  # Директория для кэширования
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self):
        """Возвращает путь к файлу кэша для текущего датасета"""
        return os.path.join(self.cache_dir, f'features_{self.current_dataset}.pkl')

    def _load_cached_features(self):
        """Загружает признаки из кэша"""
        try:
            cache_path = self._get_cache_path()
            if os.path.exists(cache_path):
                self.features = pd.read_pickle(cache_path)
                print(f"Загружены кэшированные признаки для датасета {self.current_dataset}")
                return True
        except Exception as e:
            print(f"Не удалось загрузить кэш: {e}")
        return False

    def _save_features_cache(self):
        """Сохраняет признаки в кэш"""
        if self.features is not None:
            cache_path = self._get_cache_path()
            try:
                self.features.to_pickle(cache_path)
                print(f"Признаки сохранены в кэш: {cache_path}")
            except Exception as e:
                print(f"Ошибка при сохранении кэша: {e}")

    def _process_user(self, user):
        """Обработка одного пользователя"""
        # Фильтруем данные пользователя
        user_logons = self.logon_data[self.logon_data['user'] == user]
        user_devices = self.device_data[self.device_data['user'] == user]
        user_http = self.http_data[self.http_data['user'] == user] if 'user' in self.http_data.columns else pd.DataFrame()
        user_emails = self.email_data[
            (self.email_data['from'] == user) | 
            (self.email_data['to'].str.contains(user, na=False))
        ] if not self.email_data.empty else pd.DataFrame()
        
        # Базовые характеристики
        total_days = (user_logons['date'].max() - user_logons['date'].min()).days + 1
        if total_days == 0:  # Если все события в один день
            total_days = 1
        
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
            'avg_http_requests_per_day': len(user_http) / total_days if not user_http.empty else 0,
            'unique_domains': user_http['url'].apply(lambda x: x.split('/')[0]).nunique() if not user_http.empty and 'url' in user_http.columns else 0,
        }
        
        # Email характеристики
        email_features = {
            'avg_emails_sent_per_day': len(user_emails[user_emails['from'] == user]) / total_days if not user_emails.empty else 0,
            'avg_emails_received_per_day': len(user_emails[user_emails['to'].str.contains(user, na=False)]) / total_days if not user_emails.empty else 0,
            'unique_email_contacts': pd.concat([
                user_emails['to'].str.split(';').explode(),
                user_emails['from']
            ]).nunique() if not user_emails.empty else 0,
        }
        
        # Дополнительные характеристики для r3.1
        if self.current_dataset == 'r3.1' and not self.file_data.empty:
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
        user_psycho = self.psychometric_data[self.psychometric_data['user_id'] == user]
        if not user_psycho.empty:
            psycho_features = user_psycho.iloc[0][['O', 'C', 'E', 'A', 'N']].to_dict()
        else:
            psycho_features = {
                'O': 0, 'C': 0, 'E': 0, 'A': 0, 'N': 0
            }
        
        # LDAP характеристики
        user_ldap = self.ldap_data[self.ldap_data['user_id'] == user]
        ldap_features = {
            'is_admin': 1 if not user_ldap.empty and user_ldap.iloc[-1]['role'] == 'ITAdmin' else 0,
        }
        
        # Объединяем все характеристики
        return {
            'user_id': user,
            **logon_features,
            **device_features,
            **http_features,
            **email_features,
            **file_features,
            **psycho_features,
            **ldap_features,
        }

    def load_insiders_info(self):
        """Загрузка информации об инсайдерах"""
        try:
            # Проверяем существование файла
            insiders_path = 'dataset/answers/insiders.csv'
            if not os.path.exists(insiders_path):
                print(f"Файл с информацией об инсайдерах не найден: {insiders_path}")
                self.insiders_data = pd.DataFrame(columns=['dataset', 'scenario', 'user', 'start', 'end'])
                return self.insiders_data

            # Загружаем данные
            insiders_data = pd.read_csv(insiders_path)
            
            # Проверяем наличие необходимых столбцов
            required_columns = ['dataset', 'scenario', 'user', 'start', 'end']
            if not all(col in insiders_data.columns for col in required_columns):
                print("В файле insiders.csv отсутствуют необходимые столбцы")
                self.insiders_data = pd.DataFrame(columns=required_columns)
                return self.insiders_data
            
            # Фильтруем только по текущему датасету
            try:
                dataset_num = float(self.current_dataset.replace('r', ''))
                self.insiders_data = insiders_data[insiders_data['dataset'] == dataset_num].copy()
            except (ValueError, AttributeError) as e:
                print(f"Ошибка при фильтрации данных по датасету: {e}")
                self.insiders_data = pd.DataFrame(columns=required_columns)
                return self.insiders_data
            
            if not self.insiders_data.empty:
                print(f"\nЗагружена информация об инсайдерах для датасета {self.current_dataset}:")
                print(self.insiders_data)
            else:
                print(f"\nНет информации об инсайдерах для датасета {self.current_dataset}")
            
            return self.insiders_data
            
        except Exception as e:
            print(f"Ошибка при загрузке информации об инсайдерах: {e}")
            self.insiders_data = pd.DataFrame(columns=['dataset', 'scenario', 'user', 'start', 'end'])
            return self.insiders_data
        
    def load_data(self, dataset='r3.1'):
        """Загрузка данных из указанного набора"""
        if dataset not in self.datasets:
            raise ValueError(f"Dataset {dataset} not supported. Available datasets: {self.datasets}")
            
        self.current_dataset = dataset
        
        # Загрузка информации об инсайдерах (делаем это в любом случае)
        self.load_insiders_info()
        
        # Пробуем загрузить кэшированные признаки
        if self._load_cached_features():
            print("Загружены кэшированные признаки")
            return self.features
            
        # Если кэш не найден, загружаем данные из CSV
        print(f"Кэш не найден. Загрузка данных из {dataset}...")
        dataset_path = os.path.join(self.base_path, dataset)
        
        # Загрузка основных данных
        print(f"Loading data from {dataset}...")
        

        # Логи входа
        print("\tLoading logon data...")
        try:
            self.logon_data = pd.read_csv(os.path.join(dataset_path, 'logon.csv'))
            # Проверяем и преобразуем столбец с датой
            date_column = next((col for col in ['date', 'timestamp'] if col in self.logon_data.columns), None)
            if date_column:
                self.logon_data[date_column] = pd.to_datetime(self.logon_data[date_column])
                if date_column != 'date':
                    self.logon_data = self.logon_data.rename(columns={date_column: 'date'})
        except Exception as e:
            print(f"Error loading logon data: {e}")
            self.logon_data = pd.DataFrame(columns=['user', 'pc', 'date', 'activity'])
        

        # Данные устройств
        print("\tLoading device data...")
        try:
            self.device_data = pd.read_csv(os.path.join(dataset_path, 'device.csv'))
            date_column = next((col for col in ['date', 'timestamp'] if col in self.device_data.columns), None)
            if date_column:
                self.device_data[date_column] = pd.to_datetime(self.device_data[date_column])
                if date_column != 'date':
                    self.device_data = self.device_data.rename(columns={date_column: 'date'})
        except Exception as e:
            print(f"Error loading device data: {e}")
            self.device_data = pd.DataFrame(columns=['user', 'pc', 'date', 'activity'])
        

        # HTTP данные
        print("\tLoading HTTP data...")
        try:
            self.http_data = pd.read_csv(os.path.join(dataset_path, 'http.csv'))
            # Проверяем наличие столбца user или id
            if 'id' in self.http_data.columns and 'user' not in self.http_data.columns:
                self.http_data = self.http_data.rename(columns={'id': 'user'})
            
            # Проверяем и преобразуем столбец с датой
            date_column = next((col for col in ['date', 'timestamp'] if col in self.http_data.columns), None)
            if date_column:
                self.http_data[date_column] = pd.to_datetime(self.http_data[date_column])
                if date_column != 'date':
                    self.http_data = self.http_data.rename(columns={date_column: 'date'})
            
            # Если нет столбца url, но есть website
            if 'website' in self.http_data.columns and 'url' not in self.http_data.columns:
                self.http_data = self.http_data.rename(columns={'website': 'url'})
                
        except Exception as e:
            print(f"Error loading HTTP data: {e}")
            self.http_data = pd.DataFrame(columns=['user', 'pc', 'date', 'url'])
        
        
        # Email данные
        print("\tLoading email data...")
        try:
            self.email_data = pd.read_csv(os.path.join(dataset_path, 'email.csv'))
            date_column = next((col for col in ['date', 'timestamp'] if col in self.email_data.columns), None)
            if date_column:
                self.email_data[date_column] = pd.to_datetime(self.email_data[date_column])
                if date_column != 'date':
                    self.email_data = self.email_data.rename(columns={date_column: 'date'})
        except Exception as e:
            print(f"Error loading email data: {e}")
            self.email_data = pd.DataFrame(columns=['from', 'to', 'date'])
        
        
        # Файловые операции (только для r3.1)
        if dataset == 'r3.1':
            print("\tLoading file data...")
            try:
                self.file_data = pd.read_csv(os.path.join(dataset_path, 'file.csv'))
                # Проверяем и преобразуем столбец с датой
                date_column = 'date' if 'date' in self.file_data.columns else 'timestamp'
                self.file_data[date_column] = pd.to_datetime(self.file_data[date_column])
                if date_column != 'date':
                    self.file_data = self.file_data.rename(columns={date_column: 'date'})
            except Exception as e:
                print(f"Error loading file data: {e}")
                self.file_data = pd.DataFrame(columns=['user', 'pc', 'date', 'filename'])
        else:
            self.file_data = pd.DataFrame(columns=['user', 'pc', 'date', 'filename'])
        
        
        # Психометрические данные
        try:
            print("\tLoading psychometric data...")
            self.psychometric_data = pd.read_csv(os.path.join(dataset_path, 'psychometric.csv'))
        except Exception as e:
            print(f"Error loading psychometric data: {e}")
            self.psychometric_data = pd.DataFrame(columns=['user_id', 'O', 'C', 'E', 'A', 'N'])
        
        
        # LDAP данные
        print("\tLoading LDAP data...")
        try:
            ldap_files = glob.glob(os.path.join(dataset_path, 'LDAP', '*.csv'))
            if ldap_files:
                ldap_dfs = []
                for file in ldap_files:
                    try:
                        df = pd.read_csv(file)
                        month = os.path.basename(file).split('.')[0]
                        df['month'] = month
                        ldap_dfs.append(df)
                    except Exception as e:
                        print(f"Error loading LDAP file {file}: {e}")
                if ldap_dfs:
                    self.ldap_data = pd.concat(ldap_dfs, ignore_index=True)
                else:
                    self.ldap_data = pd.DataFrame(columns=['user_id', 'role', 'month'])
            else:
                print("No LDAP files found")
                self.ldap_data = pd.DataFrame(columns=['user_id', 'role', 'month'])
        except Exception as e:
            print(f"Error processing LDAP data: {e}")
            self.ldap_data = pd.DataFrame(columns=['user_id', 'role', 'month'])
        
    def is_weekend(self, date):
        """Проверка является ли день выходным"""
        return date.weekday() >= 5
        
    def is_after_hours(self, date):
        """Проверка является ли время нерабочим"""
        hour = date.hour
        return hour < 7 or hour > 18
        
    def prepare_features(self):
        """Подготовка признаков для обучения"""
        # Проверяем наличие кэша
        if self._load_cached_features():
            print("Загружены кэшированные признаки")
            return self.features

        print("Кэш не найден, выполняется извлечение признаков...")
        
        # Получаем список всех пользователей
        users = pd.unique(self.logon_data['user'])
        features = []
        
        print("Предварительная подготовка данных...")
        
        # Оптимизируем фильтрацию с помощью groupby
        print("\tГруппировка логов входа...")
        user_logons_dict = dict(tuple(self.logon_data.groupby('user')))
        
        print("\tГруппировка данных устройств...")
        user_devices_dict = dict(tuple(self.device_data.groupby('user')))
        
        print("\tГруппировка HTTP данных...")
        if 'user' in self.http_data.columns:
            user_http_dict = dict(tuple(self.http_data.groupby('user')))
        else:
            user_http_dict = {user: pd.DataFrame() for user in users}
        
        print("\tПодготовка email данных...")
        user_emails_dict = {}
        if not self.email_data.empty:
            for user in tqdm(users, desc="\tОбработка email для пользователей"):
                user_emails = self.email_data[
                    (self.email_data['from'] == user) | 
                    (self.email_data['to'].str.contains(user, na=False))
                ]
                user_emails_dict[user] = user_emails
        else:
            user_emails_dict = {user: pd.DataFrame() for user in users}
        
        print("\tГруппировка файловых данных...")
        if self.current_dataset == 'r3.1' and not self.file_data.empty:
            user_files_dict = dict(tuple(self.file_data.groupby('user')))
        else:
            user_files_dict = {user: pd.DataFrame() for user in users}
        
        print("\tГруппировка психометрических данных...")
        user_psycho_dict = dict(tuple(self.psychometric_data.groupby('user_id')))
        
        print("\tГруппировка LDAP данных...")
        user_ldap_dict = dict(tuple(self.ldap_data.groupby('user_id')))
        
        print("\nОбработка пользователей...")
        for user in tqdm(users, desc="Извлечение признаков"):
            # Получаем предварительно отфильтрованные данные
            user_logons = user_logons_dict.get(user, pd.DataFrame())
            user_devices = user_devices_dict.get(user, pd.DataFrame())
            user_http = user_http_dict.get(user, pd.DataFrame())
            user_emails = user_emails_dict.get(user, pd.DataFrame())
            user_files = user_files_dict.get(user, pd.DataFrame())
            user_psycho = user_psycho_dict.get(user, pd.DataFrame())
            user_ldap = user_ldap_dict.get(user, pd.DataFrame())
            
            # Базовые характеристики
            if len(user_logons) > 0:
                total_days = (user_logons['date'].max() - user_logons['date'].min()).days + 1
                if total_days == 0:  # Если все события в один день
                    total_days = 1
            else:
                total_days = 1
            
            # Характеристики входов
            logon_features = {
                'avg_logons_per_day': len(user_logons) / total_days,
                'weekend_logon_ratio': user_logons[user_logons['date'].apply(self.is_weekend)].shape[0] / len(user_logons) if len(user_logons) > 0 else 0,
                'after_hours_logon_ratio': user_logons[user_logons['date'].apply(self.is_after_hours)].shape[0] / len(user_logons) if len(user_logons) > 0 else 0,
                'unique_pcs': user_logons['pc'].nunique() if len(user_logons) > 0 else 0,
            }
            
            # Характеристики устройств
            device_features = {
                'avg_device_usage_per_day': len(user_devices) / total_days,
                'device_usage_ratio': len(user_devices[user_devices['activity'] == 'Connect']) / total_days if len(user_devices) > 0 else 0,
            }
            
            # HTTP характеристики
            http_features = {
                'avg_http_requests_per_day': len(user_http) / total_days if not user_http.empty else 0,
                'unique_domains': user_http['url'].apply(lambda x: x.split('/')[0]).nunique() if not user_http.empty and 'url' in user_http.columns else 0,
            }
            
            # Email характеристики
            email_features = {
                'avg_emails_sent_per_day': len(user_emails[user_emails['from'] == user]) / total_days if not user_emails.empty else 0,
                'avg_emails_received_per_day': len(user_emails[user_emails['to'].str.contains(user, na=False)]) / total_days if not user_emails.empty else 0,
                'unique_email_contacts': pd.concat([
                    user_emails['to'].str.split(';').explode(),
                    user_emails['from']
                ]).nunique() if not user_emails.empty else 0,
            }
            
            # Файловые характеристики
            if self.current_dataset == 'r3.1' and not user_files.empty:
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
            if not user_psycho.empty:
                psycho_features = user_psycho.iloc[0][['O', 'C', 'E', 'A', 'N']].to_dict()
            else:
                psycho_features = {
                    'O': 0, 'C': 0, 'E': 0, 'A': 0, 'N': 0
                }
            
            # LDAP характеристики
            ldap_features = {
                'is_admin': 1 if not user_ldap.empty and user_ldap.iloc[-1]['role'] == 'ITAdmin' else 0,
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
        
        print("\nСоздание и нормализация DataFrame...")
        # Создаем DataFrame
        self.features = pd.DataFrame(features)
        
        # Нормализация числовых признаков
        numeric_columns = self.features.select_dtypes(include=[np.number]).columns
        numeric_columns = numeric_columns.drop(['user_id', 'is_admin']) if 'user_id' in numeric_columns else numeric_columns
        
        if not numeric_columns.empty:
            scaler = StandardScaler()
            self.features[numeric_columns] = scaler.fit_transform(self.features[numeric_columns])
        
        # Сохраняем в кэш
        print("Сохранение результатов в кэш...")
        self._save_features_cache()
        
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
        if 'user' in self.http_data.columns:
            http = self.http_data[self.http_data['user'] == user]
        elif 'id' in self.http_data.columns:
            http = self.http_data[self.http_data['id'] == user]
        else:
            http = pd.DataFrame()  # Пустой DataFrame если нет нужных столбцов
            
        for _, row in http.iterrows():
            url = row['url'] if 'url' in row else row.get('website', 'unknown')
            timeline.append({
                'date': row['date'],
                'type': 'http',
                'details': f"Visited {url}"
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