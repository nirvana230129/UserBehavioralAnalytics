import lightgbm as lgb
import numpy as np
from sklearn.calibration import CalibratedClassifierCV

class LightGBMInsiderDetector:
    def __init__(self, pos_weight=100):
        self.model = lgb.LGBMClassifier(
            objective='binary',
            boosting_type='gbdt',
            scale_pos_weight=pos_weight,
            learning_rate=0.01,
            num_leaves=15,
            max_depth=4,
            min_child_samples=20,
            min_child_weight=0.001,
            reg_alpha=0.1,
            reg_lambda=0.1,
            bagging_fraction=0.8,
            bagging_freq=5,
            feature_fraction=0.8,
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        self.calibrator = None
        self.feature_names = None
        
    def fit(self, X_train, y_train, X_val=None, y_val=None, feature_names=None):
        self.feature_names = feature_names
        
        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)] if eval_set else None
        )
        
        if X_val is not None and y_val is not None and len(np.unique(y_val)) > 1:
            try:
                self.calibrator = CalibratedClassifierCV(
                    self.model,
                    method='isotonic',
                    cv='prefit'
                )
                self.calibrator.fit(X_val, y_val)
            except:
                self.calibrator = None
        
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if self.calibrator is not None:
            try:
                probs = self.calibrator.predict_proba(X)
                return probs[:, 1] if probs.shape[1] == 2 else probs.ravel()
            except:
                pass
        
        probs = self.model.predict_proba(X)
        return probs[:, 1] if probs.shape[1] == 2 else probs.ravel()
    
    def get_feature_importance(self, top_n=20):
        importances = self.model.feature_importances_
        
        if self.feature_names:
            feature_importance = list(zip(self.feature_names, importances))
        else:
            feature_importance = list(zip(range(len(importances)), importances))
        
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        return feature_importance[:top_n]
