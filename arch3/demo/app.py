import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, average_precision_score, f1_score

st.set_page_config(page_title="Детектор аномального поведения", layout="wide")


@st.cache_data(show_spinner=False)
def load_data(train_dataset, test_dataset):
    from arch3.utils.data_processor import Arch3DataProcessor
    proc = Arch3DataProcessor()
    return proc.prepare_train_test(train_dataset, test_dataset, use_cache=True)


@st.cache_resource(show_spinner=False)
def train_model(train_dataset, test_dataset, use_lstm):
    (X_train, y_train, X_test, y_test,
     train_df, test_df, seqs_train, seqs_test) = load_data(train_dataset, test_dataset)

    from arch3.models.hybrid_detector import HybridInsiderDetector

    n_normal = sum(y_train == 0)
    n_insider = sum(y_train == 1)
    pos_weight = min(n_normal / (n_insider + 1e-6), 100)

    val_split = int(len(X_train) * 0.85)
    detector = HybridInsiderDetector(
        pos_weight=pos_weight,
        use_lstm=use_lstm,
        lstm_epochs=20,
    )
    detector.fit(
        X_train.iloc[:val_split], y_train.iloc[:val_split].values,
        seqs_train[:val_split],
        X_train.iloc[val_split:] if val_split < len(X_train) else None,
        y_train.iloc[val_split:].values if val_split < len(X_train) else None,
        seqs_train[val_split:] if val_split < len(X_train) else None,
    )

    y_proba = detector.predict_proba(X_test, seqs_test)
    y_pred = (y_proba >= 0.5).astype(int)
    feat_matrix = detector.get_feature_matrix(X_test, seqs_test)

    return detector, X_test, y_test, y_pred, y_proba, test_df, feat_matrix, seqs_test


def page_overview(train_ds, test_ds, use_lstm):
    st.header("Результаты модели")
    with st.spinner("Обучение модели..."):
        detector, X_test, y_test, y_pred, y_proba, test_df, feat_matrix, _ = \
            train_model(train_ds, test_ds, use_lstm)

    y_true = y_test.values
    pr_prec, pr_rec, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(pr_rec, pr_prec)
    ap = average_precision_score(y_true, y_proba)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    c1, c2, c3 = st.columns(3)
    c1.metric("PR-AUC", f"{pr_auc:.3f}")
    c2.metric("Average Precision", f"{ap:.3f}")
    c3.metric("F1-score", f"{f1:.3f}")

    n_insiders = int(sum(y_true))
    n_total = len(y_true)
    n_detected = int(sum(y_pred))
    st.caption(f"Пользователей: {n_total} | Инсайдеров: {n_insiders} | Обнаружено: {n_detected}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(pr_rec, pr_prec, linewidth=2, label=f"PR-AUC={pr_auc:.3f}")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_title("PR Curve")
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1])

    normal_p = y_proba[y_true == 0]
    insider_p = y_proba[y_true == 1]
    if len(normal_p) > 0:
        axes[1].hist(normal_p, bins=40, alpha=0.5, label="Обычные", density=True)
    for i, p in enumerate(insider_p):
        axes[1].axvline(x=p, color='r', alpha=0.7, linewidth=2,
                        label="Инсайдеры" if i == 0 else None)
    axes[1].set_xlabel("Вероятность инсайдера")
    axes[1].set_ylabel("Плотность")
    axes[1].set_title("Распределение вероятностей")
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_xlim([0, 1])

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Топ-15 важных признаков")
    importance = detector.get_feature_importance(15)
    feat_df = pd.DataFrame(importance, columns=["Признак", "Важность"])
    st.bar_chart(feat_df.set_index("Признак"))


def page_explanations(train_ds, test_ds, use_lstm):
    st.header("Объяснения решений")

    with st.spinner("Загрузка модели..."):
        detector, X_test, y_test, y_pred, y_proba, test_df, feat_matrix, _ = \
            train_model(train_ds, test_ds, use_lstm)

    try:
        from arch3.models.explainer import SHAPExplainer, SHAP_AVAILABLE
    except ImportError:
        SHAP_AVAILABLE = False

    if not SHAP_AVAILABLE:
        st.warning("Библиотека shap не установлена. Установите: pip install shap")
        return

    user_probs = sorted(
        zip(test_df['user_id'].values, y_proba),
        key=lambda x: x[1], reverse=True
    )
    top_users = [u for u, _ in user_probs[:20]]
    selected_user = st.selectbox("Выберите пользователя (топ-20 по риску):", top_users)

    if selected_user:
        idx = test_df[test_df['user_id'] == selected_user].index
        if len(idx) == 0:
            st.error("Пользователь не найден")
            return
        row_idx = test_df.index.get_loc(idx[0])
        prob = y_proba[row_idx]
        is_ins = int(test_df.loc[idx[0], 'is_insider'])

        col1, col2 = st.columns(2)
        col1.metric("Вероятность инсайдера", f"{prob:.4f}")
        col2.metric("Метка", "Инсайдер" if is_ins else "Обычный")

        explainer = SHAPExplainer(detector)
        contributions = explainer.explain_user(feat_matrix[row_idx], top_n=15)

        if contributions:
            contrib_df = pd.DataFrame(contributions, columns=["Признак", "SHAP-значение"])
            contrib_df = contrib_df.sort_values("SHAP-значение")
            fig, ax = plt.subplots(figsize=(8, max(4, len(contrib_df) * 0.35)))
            colors = ["#d73027" if v >= 0 else "#4575b4" for v in contrib_df["SHAP-значение"]]
            ax.barh(contrib_df["Признак"], contrib_df["SHAP-значение"], color=colors)
            ax.axvline(x=0, color='black', linewidth=0.8)
            ax.set_xlabel("SHAP-значение")
            ax.set_title(f"Вклад признаков: {selected_user}")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("SHAP объяснения недоступны")


def page_comparison(train_ds, test_ds, use_lstm):
    st.header("Сравнение arch2 vs arch3")

    with st.spinner("Вычисление метрик..."):
        detector, X_test, y_test, y_pred, y_proba_3, test_df, feat_matrix, _ = \
            train_model(train_ds, test_ds, use_lstm)

        y_true = y_test.values
        pr_prec3, pr_rec3, _ = precision_recall_curve(y_true, y_proba_3)
        auc3 = auc(pr_rec3, pr_prec3)
        ap3 = average_precision_score(y_true, y_proba_3)
        f1_3 = f1_score(y_true, y_pred, zero_division=0)

        arch2_metrics = None
        pr_rec2, pr_prec2 = None, None
        try:
            from arch2.utils.data_processor import InsiderDetectionDataProcessor
            from arch2.models.lgbm_detector import LightGBMInsiderDetector
            proc2 = InsiderDetectionDataProcessor()
            X_tr2, y_tr2, X_te2, y_te2, _, _, _ = proc2.prepare_train_test(
                train_ds, test_ds, use_cache=True
            )
            vs = int(len(X_tr2) * 0.85)
            n_n = sum(y_tr2.iloc[:vs] == 0)
            n_i = sum(y_tr2.iloc[:vs] == 1)
            pw = min(n_n / (n_i + 1e-6), 100)
            det2 = LightGBMInsiderDetector(pw)
            det2.fit(
                X_tr2.iloc[:vs].values, y_tr2.iloc[:vs].values,
                X_tr2.iloc[vs:].values if vs < len(X_tr2) else None,
                y_tr2.iloc[vs:].values if vs < len(X_tr2) else None,
            )
            yp2 = det2.predict_proba(X_te2.values)
            ypred2 = det2.predict(X_te2.values)
            pr_prec2, pr_rec2, _ = precision_recall_curve(y_te2.values, yp2)
            arch2_metrics = {
                'PR-AUC': auc(pr_rec2, pr_prec2),
                'AP Score': average_precision_score(y_te2.values, yp2),
                'F1': f1_score(y_te2.values, ypred2, zero_division=0),
            }
        except Exception as e:
            st.warning(f"arch2 недоступна: {e}")

    arch3_metrics = {'PR-AUC': auc3, 'AP Score': ap3, 'F1': f1_3}

    rows = []
    for key in arch3_metrics:
        v3 = arch3_metrics[key]
        v2 = arch2_metrics.get(key, None) if arch2_metrics else None
        delta = f"+{v3 - v2:.3f}" if v2 is not None else "—"
        rows.append({
            "Метрика": key,
            "arch2": f"{v2:.3f}" if v2 is not None else "—",
            "arch3": f"{v3:.3f}",
            "Δ": delta,
        })

    st.table(pd.DataFrame(rows))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(pr_rec3, pr_prec3, label=f"arch3 ({auc3:.3f})", linewidth=2)
    if arch2_metrics and pr_rec2 is not None:
        ax.plot(pr_rec2, pr_prec2, linestyle='--',
                label=f"arch2 ({arch2_metrics['PR-AUC']:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR Curve: arch2 vs arch3")
    ax.legend()
    ax.grid(True)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def main():
    st.title("Детектор аномального поведения пользователей")
    st.caption("CERT Insider Threat Dataset — Этап 4")

    with st.sidebar:
        st.header("Параметры")
        train_ds = st.selectbox("Обучающий датасет", ["r1", "r2", "r3.1"], index=0)
        test_ds = st.selectbox("Тестовый датасет", ["r1", "r2", "r3.1"], index=0)
        use_lstm = st.checkbox("Использовать LSTM", value=True)
        st.divider()
        page = st.radio("Раздел", [
            "Обзор и метрики",
            "Объяснения (SHAP)",
            "Сравнение arch2 vs arch3",
        ])

    if page == "Обзор и метрики":
        page_overview(train_ds, test_ds, use_lstm)
    elif page == "Объяснения (SHAP)":
        page_explanations(train_ds, test_ds, use_lstm)
    elif page == "Сравнение arch2 vs arch3":
        page_comparison(train_ds, test_ds, use_lstm)


if __name__ == '__main__':
    main()
