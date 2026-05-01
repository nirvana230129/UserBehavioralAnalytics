import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer


FEATURE_GROUPS = {
    'logon': [
        'avg_logons_per_day', 'weekend_logon_ratio', 'after_hours_logon_ratio', 'unique_pcs',
        'weekend_ratio', 'after_hours_ratio', 'night_ratio', 'atypical_time_ratio',
        'drift_logon_rate', 'drift_pc_diversity', 'unique_activities',
    ],
    'device': [
        'avg_device_usage_per_day', 'device_usage_ratio', 'unique_device_types',
        'device_per_logon', 'admin_device_usage',
    ],
    'http': ['avg_http_requests_per_day', 'unique_domains', 'http_per_logon'],
    'email': [
        'avg_emails_sent_per_day', 'avg_emails_received_per_day',
        'unique_email_contacts', 'email_per_day',
    ],
    'file': ['avg_file_copies_per_day', 'unique_file_types'],
    'psychometric': ['O', 'C', 'E', 'A', 'N', 'neurotic_after_hours', 'openness_diversity'],
    'temporal_windows': [],
    'derived': [
        'activity_intensity', 'temporal_anomaly_score', 'behavior_instability',
        'activity_diversity', 'burst_intensity', 'min_interval', 'max_interval',
        'rare_sequence_ratio', 'logon_vs_dept_mean', 'device_vs_dept_mean',
    ],
    'admin': ['is_admin'],
}


def _get_group(col):
    for group, cols in FEATURE_GROUPS.items():
        if col in cols:
            return group
    if col.startswith(('logon_rate_', 'device_rate_')):
        return 'temporal_windows'
    if col.startswith('total_'):
        return 'logon'
    return 'derived'


class AdaptiveFeatureProcessor:
    def __init__(self, dropout_p=0.3, random_state=42):
        self.dropout_p = dropout_p
        self.random_state = random_state
        self.transformers = {}
        self.feature_names = None
        self.group_map = {}
        self.rng = np.random.RandomState(random_state)

    def fit(self, X: pd.DataFrame):
        self.feature_names = list(X.columns)
        self.group_map = {col: _get_group(col) for col in self.feature_names}

        groups = set(self.group_map.values())
        for group in groups:
            cols = [c for c in self.feature_names if self.group_map[c] == group]
            if not cols:
                continue
            data = X[cols].values.astype(float)
            n_q = min(len(X), 1000)
            qt = QuantileTransformer(output_distribution='normal',
                                     n_quantiles=n_q,
                                     random_state=self.random_state)
            qt.fit(data)
            self.transformers[group] = (cols, qt)
        return self

    def transform(self, X: pd.DataFrame, apply_dropout=False) -> pd.DataFrame:
        result = X.copy().astype(float)
        for group, (cols, qt) in self.transformers.items():
            present = [c for c in cols if c in result.columns]
            if not present:
                continue
            result[present] = qt.transform(result[present].values)

        if apply_dropout:
            groups = list(self.transformers.keys())
            for group in groups:
                if self.rng.rand() < self.dropout_p:
                    cols = [c for c in self.transformers[group][0] if c in result.columns]
                    result[cols] = 0.0
        return result

    def fit_transform(self, X: pd.DataFrame, apply_dropout=False) -> pd.DataFrame:
        self.fit(X)
        return self.transform(X, apply_dropout=apply_dropout)

    def align_columns(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        train_cols = set(X_train.columns)
        test_cols = set(X_test.columns)
        common = list(train_cols & test_cols)
        only_train = list(train_cols - test_cols)
        only_test = list(test_cols - train_cols)

        X_test_aligned = X_test[common].copy()
        for col in only_train:
            X_test_aligned[col] = 0.0

        X_train_aligned = X_train[common + only_train].copy()
        X_test_aligned = X_test_aligned[common + only_train]

        return X_train_aligned, X_test_aligned
