import numpy as np
import os

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class SHAPExplainer:
    def __init__(self, detector, feature_names=None):
        self.detector = detector
        self.feature_names = feature_names or detector.feature_names or []
        self.explainer = None
        self._init_explainer()

    def _init_explainer(self):
        if not SHAP_AVAILABLE:
            return
        try:
            self.explainer = shap.TreeExplainer(self.detector.lgbm)
        except Exception as e:
            print(f"  SHAP TreeExplainer недоступен: {e}")

    def explain_user(self, X_row: np.ndarray, top_n=10):
        if not SHAP_AVAILABLE or self.explainer is None:
            return []
        row = X_row.reshape(1, -1)
        shap_vals = self.explainer.shap_values(row)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        vals = shap_vals[0]
        pairs = list(zip(self.feature_names, vals))
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        return pairs[:top_n]

    def explain_top_suspicious(self, user_ids, X: np.ndarray, y_proba: np.ndarray,
                                top_n=5, plot_dir=None):
        ranked = sorted(zip(user_ids, y_proba, range(len(y_proba))),
                        key=lambda x: x[1], reverse=True)
        top = ranked[:top_n]

        print("\nОбъяснения топ-{} подозрительных пользователей:".format(top_n))
        print("-" * 60)
        for user_id, prob, idx in top:
            print(f"\nПользователь: {user_id}  (вероятность инсайдера: {prob:.4f})")
            contributions = self.explain_user(X[idx])
            for feat, val in contributions:
                sign = "+" if val >= 0 else ""
                print(f"  {feat:40s}  {sign}{val:.4f}")

        if SHAP_AVAILABLE and self.explainer is not None and plot_dir:
            self._save_summary_plot(X, plot_dir)

    def _save_summary_plot(self, X: np.ndarray, plot_dir: str):
        if not SHAP_AVAILABLE or self.explainer is None:
            return
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            shap_vals = self.explainer.shap_values(X)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]

            os.makedirs(plot_dir, exist_ok=True)
            shap.summary_plot(shap_vals, X,
                              feature_names=self.feature_names,
                              show=False, max_display=20)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, 'shap_summary.png'))
            plt.close()
            print(f"  SHAP summary сохранён в {plot_dir}/shap_summary.png")
        except Exception as e:
            print(f"  Ошибка при сохранении SHAP plot: {e}")

    def get_global_importance(self, X: np.ndarray, top_n=20):
        if not SHAP_AVAILABLE or self.explainer is None:
            return []
        shap_vals = self.explainer.shap_values(X)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        mean_abs = np.abs(shap_vals).mean(axis=0)
        pairs = list(zip(self.feature_names, mean_abs))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:top_n]
