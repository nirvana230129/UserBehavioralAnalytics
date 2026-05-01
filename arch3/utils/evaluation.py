import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    precision_recall_curve, average_precision_score, auc,
)


class Arch3Evaluator:
    def __init__(self, train_dataset, test_dataset):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.plot_dir = f'arch3/plots/{train_dataset}_{test_dataset}'

    def evaluate(self, y_true, y_pred, y_proba, test_df, arch2_metrics=None):
        os.makedirs(self.plot_dir, exist_ok=True)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        pr_prec, pr_rec, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(pr_rec, pr_prec)
        ap_score = average_precision_score(y_true, y_proba)

        metrics = {
            'pr_auc': pr_auc,
            'ap_score': ap_score,
            'f1': f1,
            'precision': precision,
            'recall': recall,
        }

        print(f"\n{'='*60}")
        print(f"Оценка: {self.train_dataset} -> {self.test_dataset}")
        print(f"{'='*60}")
        print(f"\nМетрики arch3:")
        print(f"  PR-AUC:     {pr_auc:.3f}")
        print(f"  AP Score:   {ap_score:.3f}")
        print(f"  F1:         {f1:.3f}")
        print(f"  Precision:  {precision:.3f}")
        print(f"  Recall:     {recall:.3f}")

        if arch2_metrics:
            print(f"\nСравнение arch2 -> arch3:")
            print(f"  {'Метрика':<12} {'arch2':>8} {'arch3':>8} {'delta':>8}")
            print(f"  {'-'*40}")
            for key in ['pr_auc', 'ap_score', 'f1', 'precision', 'recall']:
                v2 = arch2_metrics.get(key, 0)
                v3 = metrics.get(key, 0)
                delta = v3 - v2
                sign = '+' if delta >= 0 else ''
                print(f"  {key:<12} {v2:>8.3f} {v3:>8.3f} {sign}{delta:>7.3f}")

        n_insiders = int(sum(y_true))
        n_total = len(y_true)
        n_detected = int(sum(y_pred))
        insiders = test_df[test_df['is_insider'] == 1]['user_id'].tolist()

        print(f"\nПользователей: {n_total}, инсайдеров: {n_insiders} "
              f"({n_insiders/n_total*100:.2f}%), обнаружено: {n_detected}")

        user_probs = sorted(
            zip(test_df['user_id'].values, y_proba),
            key=lambda x: x[1], reverse=True
        )

        print("\nTop-K Recall:")
        for k in [5, 10, 20, 50]:
            if k > len(user_probs):
                continue
            top_k = {u for u, _ in user_probs[:k]}
            found = sum(1 for u in insiders if u in top_k)
            r = found / len(insiders) if insiders else 0
            print(f"  Top-{k:2d}: {r:.3f}  ({found}/{len(insiders)})")

        if insiders:
            print("\nРанги инсайдеров:")
            for insider in insiders:
                rank = next((i + 1 for i, (u, _) in enumerate(user_probs) if u == insider), -1)
                prob = next((p for u, p in user_probs if u == insider), 0)
                print(f"  {insider}: ранг {rank}/{n_total}, p={prob:.4f}")

        self._plot_pr_curve(pr_rec, pr_prec, pr_auc, arch2_metrics)
        self._plot_prob_distribution(y_true, y_proba)

        return metrics

    def _plot_pr_curve(self, recall, precision, pr_auc, arch2_metrics=None):
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'arch3 PR-AUC={pr_auc:.3f}', linewidth=2)
        if arch2_metrics and 'pr_recall' in arch2_metrics and 'pr_precision' in arch2_metrics:
            plt.plot(arch2_metrics['pr_recall'], arch2_metrics['pr_precision'],
                     linestyle='--', label=f"arch2 PR-AUC={arch2_metrics['pr_auc']:.3f}")
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'PR Curve: {self.train_dataset} -> {self.test_dataset}')
        plt.legend()
        plt.grid(True)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.tight_layout()
        plt.savefig(f'{self.plot_dir}/pr_curve.png')
        plt.close()

    def _plot_prob_distribution(self, y_true, y_proba):
        plt.figure(figsize=(10, 5))
        normal_p = y_proba[np.array(y_true) == 0]
        insider_p = y_proba[np.array(y_true) == 1]
        if len(normal_p) > 0:
            plt.hist(normal_p, bins=50, alpha=0.5, label='Обычные', density=True)
        for p in insider_p:
            plt.axvline(x=p, color='r', alpha=0.6, linewidth=2)
        if len(insider_p) > 0:
            plt.axvline(x=insider_p[0], color='r', alpha=0.6, linewidth=2, label='Инсайдеры')
        plt.xlabel('Вероятность инсайдера')
        plt.ylabel('Плотность')
        plt.title(f'Распределение вероятностей: {self.train_dataset} -> {self.test_dataset}')
        plt.legend()
        plt.grid(True)
        plt.xlim([0, 1])
        plt.tight_layout()
        plt.savefig(f'{self.plot_dir}/probability_distribution.png')
        plt.close()
