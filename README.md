# Поведенческий анализ пользователей для детектирования аномального поведения

Система детектирования инсайдерских угроз на основе поведенческого анализа с использованием методов машинного обучения. Датасет: [CERT Insider Threat Test Dataset](https://resources.sei.cmu.edu/library/asset-view.cfm?assetid=508099) (версии r1, r2, r3.1).

---

## Структура репозитория

```
nir/
├── data_loader.py          # Универсальный загрузчик данных
├── requirements.txt        # Общие зависимости
├── dataset/                # Данные CERT (r1/, r2/, r3.1/, answers/)
├── cache/                  # Кэш вычисленных признаков (.pkl)
├── arch1/                  # Архитектура 1: ансамбль Isolation Forest
│   ├── example.py
│   ├── models/
│   └── requirements.txt
├── arch2/                  # Архитектура 2: LightGBM + временные признаки
│   ├── example.py
│   ├── models/
│   ├── utils/
│   └── requirements.txt
└── arch3/                  # Архитектура 3: Гибридная (LightGBM + LSTM + SHAP)
    ├── example.py
    ├── models/
    ├── utils/
    ├── demo/
    │   └── app.py          # Streamlit-демостенд
    └── requirements.txt
```

---

## Архитектуры

### Архитектура 1 (`arch1/`)
Ансамбль специализированных детекторов на базе **Isolation Forest**. Каждый детектор анализирует отдельный аспект поведения (активность по времени, частота событий, объём данных, доступ к ресурсам, файловые операции). Мета-модель: `RandomForestClassifier` с калибровкой через `CalibratedClassifierCV`.

### Архитектура 2 (`arch2/`)
Единая модель **LightGBM** с расширенной инженерией признаков: скользящие окна, поведенческий дрейф, редкие последовательности действий (триграммы), групповые отклонения по роли. Метрики: Precision, Recall, F1, PR-AUC, Top-K Recall.

### Архитектура 3 (`arch3/`)
Гибридная модель, объединяющая:
- **Адаптивная нормализация признаков** — `QuantileTransformer` per-group + feature-group dropout при обучении, что повышает устойчивость при переносе между датасетами (r1→r2, r2→r3.1 и т.д.)
- **LSTM-автоэнкодер** (PyTorch) — обучается на последовательностях событий нормальных пользователей; ошибка реконструкции и скрытый вектор используются как дополнительные признаки
- **SHAP-объяснения** — `shap.TreeExplainer` для per-user объяснений решений модели
- **Streamlit-демостенд** — интерактивная визуализация метрик, объяснений и сравнения arch2 vs arch3

---

## Установка зависимостей

```bash
# Общие зависимости
pip install -r requirements.txt

# Для arch1
pip install -r arch1/requirements.txt

# Для arch2
pip install -r arch2/requirements.txt

# Для arch3 (включает torch, shap, streamlit)
pip install -r arch3/requirements.txt
```

---

## Запуск

### Архитектура 1
```bash
python -m arch1.example
```

### Архитектура 2
```bash
python -m arch2.example
```

### Архитектура 3
```bash
python -m arch3.example
```

Константы `TRAIN_DATASET` и `TEST_DATASET` в `example.py` каждой архитектуры управляют выбором датасета (допустимые значения: `r1`, `r2`, `r3.1`).

### Демостенд (arch3)
```bash
streamlit run arch3/demo/app.py
```

Откроется в браузере по адресу `http://localhost:8501`. Разделы:
- **Обзор и метрики** — PR-кривая, распределение вероятностей, важность признаков
- **Объяснения (SHAP)** — вклад признаков для конкретного пользователя
- **Сравнение arch2 vs arch3** — таблица метрик и совмещённая PR-кривая

---

## Данные

Датасет размещается в директории `dataset/`:
```
dataset/
├── r1/       # logon.csv, device.csv, http.csv, LDAP/
├── r2/       # + email.csv, psychometric.csv
├── r3.1/     # + file.csv
└── answers/
    └── insiders.csv
```

Кэш признаков (`.pkl`-файлы) хранится в `cache/` и создаётся автоматически при первом запуске.
