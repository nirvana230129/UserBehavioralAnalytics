import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, average_precision_score, auc
import matplotlib.pyplot as plt
import os

class InsiderDetectionEvaluator:
    def __init__(self, train_dataset, test_dataset):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        
    def evaluate(self, y_true, y_pred, y_proba, test_df):
        print("\n" + "="*60)
        print(f"Оценка результатов: {self.train_dataset} -> {self.test_dataset}")
        print("="*60)
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        pr_precision, pr_recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(pr_recall, pr_precision)
        ap_score = average_precision_score(y_true, y_proba)
        
        print("\nМетрики:")
        print("-"*60)
        print(f"PR-AUC:     {pr_auc:.3f}")
        print(f"AP Score:   {ap_score:.3f}")
        print(f"F1-score:   {f1:.3f}")
        print(f"Precision:  {precision:.3f}")
        print(f"Recall:     {recall:.3f}")
        print("-"*60)
        
        n_insiders = sum(y_true)
        n_total = len(y_true)
        n_detected = sum(y_pred)
        
        print(f"\nВсего пользователей: {n_total}")
        print(f"Инсайдеров: {n_insiders} ({n_insiders/n_total*100:.2f}%)")
        print(f"Обнаружено аномалий: {n_detected}")
        print("-"*60)
        
        insiders = test_df[test_df['is_insider'] == 1]['user_id'].tolist()
        
        user_probs = list(zip(test_df['user_id'].values, y_proba))
        user_probs.sort(key=lambda x: x[1], reverse=True)
        
        top_k_values = [5, 10, 20, 50]
        print("\nTop-K Recall:")
        for k in top_k_values:
            if k <= len(user_probs):
                top_k_users = [user for user, _ in user_probs[:k]]
                detected_in_top_k = sum(1 for user in insiders if user in top_k_users)
                recall_at_k = detected_in_top_k / len(insiders) if len(insiders) > 0 else 0
                print(f"  Top-{k:2d}: {recall_at_k:.3f} ({detected_in_top_k}/{len(insiders)} найдено)")
        
        print("-"*60)
        
        if insiders:
            print("\nИнсайдеры и их ранги:")
            for insider in insiders:
                rank = next((i+1 for i, (u, _) in enumerate(user_probs) if u == insider), -1)
                prob = next((p for u, p in user_probs if u == insider), 0)
                print(f"  {insider}: ранг {rank}/{len(user_probs)}, вероятность {prob:.4f}")
        
        self._plot_pr_curve(pr_recall, pr_precision, pr_auc)
        self._plot_probability_distribution(y_true, y_proba)
        
        return {
            'pr_auc': pr_auc,
            'ap_score': ap_score,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def _plot_pr_curve(self, recall, precision, pr_auc):
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, label=f'PR-AUC = {pr_auc:.3f}', linewidth=2)
        plt.xlabel('Полнота (Recall)')
        plt.ylabel('Точность (Precision)')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        plot_dir = f'arch2/plots/{self.train_dataset}_{self.test_dataset}'
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(f'{plot_dir}/pr_curve.png')
        plt.close()
    
    def _plot_probability_distribution(self, y_true, y_proba):
        plt.figure(figsize=(12, 6))
        
        normal_probs = y_proba[y_true == 0]
        insider_probs = y_proba[y_true == 1]
        
        if len(normal_probs) > 0:
            plt.hist(normal_probs, bins=50, alpha=0.5, label='Обычные пользователи', density=True)
        
        if len(insider_probs) > 0:
            for prob in insider_probs:
                plt.axvline(x=prob, color='r', alpha=0.5, linewidth=2)
            plt.axvline(x=insider_probs[0], color='r', alpha=0.5, linewidth=2, label='Инсайдеры')
        
        plt.xlabel('Вероятность быть инсайдером')
        plt.ylabel('Плотность')
        plt.title('Распределение вероятностей')
        plt.legend()
        plt.grid(True)
        plt.xlim([0, 1])
        
        plot_dir = f'arch2/plots/{self.train_dataset}_{self.test_dataset}'
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(f'{plot_dir}/probability_distribution.png')
        plt.close()
