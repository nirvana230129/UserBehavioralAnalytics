import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import numpy as np

from arch3.utils.data_processor import Arch3DataProcessor
from arch3.models.hybrid_detector import HybridInsiderDetector
from arch3.models.explainer import SHAPExplainer
from arch3.utils.evaluation import Arch3Evaluator

TRAIN_DATASET = 'r1'
TEST_DATASET = 'r1'
USE_CACHE = True
USE_LSTM = True
LSTM_EPOCHS = 30
EXPLAIN_TOP_N = 5
COMPARE_WITH_ARCH2 = True


def run_arch2_baseline(train_dataset, test_dataset, use_cache=True):
    try:
        from arch2.utils.data_processor import InsiderDetectionDataProcessor
        from arch2.models.lgbm_detector import LightGBMInsiderDetector
        from sklearn.metrics import (
            precision_score, recall_score, f1_score,
            precision_recall_curve, average_precision_score, auc,
        )

        proc = InsiderDetectionDataProcessor()
        X_tr, y_tr, X_te, y_te, _, test_df, _ = proc.prepare_train_test(
            train_dataset, test_dataset, use_cache=use_cache
        )

        val_split = int(len(X_tr) * 0.85)
        n_normal = sum(y_tr.iloc[:val_split] == 0)
        n_insider = sum(y_tr.iloc[:val_split] == 1)
        pos_w = min(n_normal / (n_insider + 1e-6), 100)

        det = LightGBMInsiderDetector(pos_weight=pos_w)
        det.fit(
            X_tr.iloc[:val_split].values, y_tr.iloc[:val_split].values,
            X_tr.iloc[val_split:].values if val_split < len(X_tr) else None,
            y_tr.iloc[val_split:].values if val_split < len(X_tr) else None,
        )

        y_pred = det.predict(X_te.values)
        y_proba = det.predict_proba(X_te.values)

        pr_prec, pr_rec, _ = precision_recall_curve(y_te.values, y_proba)

        return {
            'pr_auc': auc(pr_rec, pr_prec),
            'ap_score': average_precision_score(y_te.values, y_proba),
            'f1': f1_score(y_te.values, y_pred, zero_division=0),
            'precision': precision_score(y_te.values, y_pred, zero_division=0),
            'recall': recall_score(y_te.values, y_pred, zero_division=0),
            'pr_recall': pr_rec,
            'pr_precision': pr_prec,
        }
    except Exception as e:
        print(f"  arch2 baseline недоступен: {e}")
        return None


def run_arch3(train_dataset, test_dataset, use_cache=True,
              use_lstm=True, lstm_epochs=30, explain_top_n=5,
              arch2_metrics=None):
    print(f"\n{'='*60}")
    print(f"Архитектура 3: Гибридная модель (LightGBM + LSTM + SHAP)")
    print(f"  Train: {train_dataset}  Test: {test_dataset}")
    print(f"{'='*60}")

    print("\n[1/4] Подготовка данных...")
    proc = Arch3DataProcessor()
    (X_train, y_train, X_test, y_test,
     train_df, test_df,
     sequences_train, sequences_test) = proc.prepare_train_test(
        train_dataset, test_dataset, use_cache=use_cache
    )

    print(f"\n  Признаков: {X_train.shape[1]}")
    print(f"  Train: {X_train.shape[0]} пользователей ({sum(y_train)} инсайдеров)")
    print(f"  Test:  {X_test.shape[0]} пользователей ({sum(y_test)} инсайдеров)")

    n_normal = sum(y_train == 0)
    n_insider = sum(y_train == 1)
    pos_weight = n_normal / (n_insider + 1e-6)
    print(f"  Вес инсайдеров: {min(pos_weight, 100):.1f}")

    val_split = int(len(X_train) * 0.85)
    X_tr_s = X_train.iloc[:val_split]
    y_tr_s = y_train.iloc[:val_split]
    X_val = X_train.iloc[val_split:]
    y_val = y_train.iloc[val_split:]
    seqs_tr_s = sequences_train[:val_split]
    seqs_val = sequences_train[val_split:]

    print(f"\n[2/4] Создание модели (LSTM={'вкл' if use_lstm else 'выкл'})...")
    detector = HybridInsiderDetector(
        pos_weight=pos_weight,
        use_lstm=use_lstm,
        lstm_epochs=lstm_epochs,
    )

    print("\n[3/4] Обучение...")
    detector.fit(
        X_tr_s, y_tr_s.values, seqs_tr_s,
        X_val if len(X_val) > 0 else None,
        y_val.values if len(y_val) > 0 else None,
        seqs_val if len(seqs_val) > 0 else None,
    )

    print("\nТоп-15 важных признаков:")
    for i, (feat, imp) in enumerate(detector.get_feature_importance(15), 1):
        print(f"  {i:2d}. {feat:42s} {imp:8.1f}")

    print("\n[4/4] Тестирование и оценка...")
    y_proba = detector.predict_proba(X_test, sequences_test)
    y_pred = (y_proba >= 0.5).astype(int)

    evaluator = Arch3Evaluator(train_dataset, test_dataset)
    metrics = evaluator.evaluate(
        y_test.values, y_pred, y_proba, test_df, arch2_metrics
    )

    print(f"\n[Объяснения]")
    feat_matrix = detector.get_feature_matrix(X_test, sequences_test)
    explainer = SHAPExplainer(detector)
    test_users = test_df['user_id'].tolist()
    explainer.explain_top_suspicious(
        test_users, feat_matrix, y_proba,
        top_n=explain_top_n,
        plot_dir=evaluator.plot_dir,
    )

    return metrics, detector


def main():
    start = time.time()

    arch2_metrics = None
    if COMPARE_WITH_ARCH2:
        print(f"\n[Baseline arch2: {TRAIN_DATASET} -> {TEST_DATASET}]")
        arch2_metrics = run_arch2_baseline(TRAIN_DATASET, TEST_DATASET, USE_CACHE)
        if arch2_metrics:
            print(f"  arch2 PR-AUC: {arch2_metrics['pr_auc']:.3f}")

    run_arch3(
        TRAIN_DATASET, TEST_DATASET,
        use_cache=USE_CACHE,
        use_lstm=USE_LSTM,
        lstm_epochs=LSTM_EPOCHS,
        explain_top_n=EXPLAIN_TOP_N,
        arch2_metrics=arch2_metrics,
    )

    elapsed = time.time() - start
    print(f"\nВремя выполнения: {elapsed:.1f}с\n")


if __name__ == '__main__':
    main()
