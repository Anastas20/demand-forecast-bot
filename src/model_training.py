"""
model_training.py

Обучение лог-модели CatBoost для прогнозирования месячного спроса
и статистическая калибровка прогнозов по категориям товаров.

Вход:
    таблица ml_monthly_base в SQLite-базе demand.db
    (формируется скриптом ml_dataset_builder.py)

Выход:
    ../models/catboost_demand_log.cbm   – сохранённая модель CatBoost
    ../models/model_meta.json           – метаданные (список признаков и конфиг)
    ../models/calib_by_category.json    – k по категориям для калибровки
    таблица calibration_category        – k по категориям в SQLite
"""

import os
import json
from typing import Dict
from datetime import datetime, timezone 

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from db import get_connection  


# ========= Метрики =========

def rmse(y_true, y_pred) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true, y_pred) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def wape(y_true, y_pred) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float(np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100)


def bias(y_true, y_pred) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float(np.mean(y_pred - y_true))


# ========= Лучшие параметры модели (лог-таргет) =========

BEST_PARAMS_LOG = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "iterations": 2000,
    "random_seed": 42,
    "od_type": "Iter",
    "od_wait": 100,
    "depth": 3,
    "learning_rate": 0.05,
    "l2_leaf_reg": 3,
    "bagging_temperature": 0.5,
}


# ========= Вспомогательные функции =========

def build_feature_space(df: pd.DataFrame):
    """
    Формирует список признаков и индексы категориальных признаков.
    Из датасета исключаются:
      - таргеты и производные (qty_month, revenue_month, target_log),
      - служебные поля (month, first_date, last_date, is_stockout_period, product_name).
    """
    drop_cols = [
        "qty_month",
        "revenue_month",
        "month",
        "first_date",
        "last_date",
        "is_stockout_period",
        "product_name",
        "target_log",
    ]

    # все признаки, кроме таргета и служебных
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # Явно задаём список категориальных признаков
    candidate_cat_cols = [
        "seller_sku",
        "marketplace",
        "category",
        "subcategory",
        "family_id",
        "marketplaces_type",
    ]
    cat_cols = [c for c in candidate_cat_cols if c in feature_cols]

    # Остальные признаки считаем числовыми
    num_cols = [c for c in feature_cols if c not in cat_cols]

    # Приводим числовые признаки к float (всё странное → NaN)
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Категориальные признаки → строки без None/NaN
    for col in cat_cols:
        df[col] = df[col].astype("object")
        df[col] = df[col].where(df[col].notna(), "NA")
        df[col] = df[col].astype(str)

    cat_feature_indices = [feature_cols.index(c) for c in cat_cols]

    print("Всего признаков:", len(feature_cols))
    print("Категориальные признаки:", cat_cols)
    print("Индексы cat_features:", cat_feature_indices)

    return feature_cols, cat_feature_indices

def split_by_time(df: pd.DataFrame):
    """
    Делит данные на train / val / test по месяцам:
      - последние 4 месяца → test
      - предыдущие 4 месяца → val
      - остальные → train

    При этом:
      - из train/val исключаются сток-аут периоды (is_stockout_period == 1),
      - test берётся целиком (для последующей оценки и восстановления спроса).
    """
    df = df.copy()
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df = df[df["month"].notna()].copy()

    months = np.sort(df["month"].unique())
    print("Всего месяцев:", len(months))
    print("От", months[0], "до", months[-1])

    if len(months) < 8:
        raise ValueError("Слишком мало месяцев для разделения на train/val/test.")

    test_months = months[-4:]
    val_months = months[-8:-4]
    train_months = months[:-8]

    print("train:", train_months[0], "→", train_months[-1])
    print("val  :", val_months[0], "→", val_months[-1])
    print("test :", test_months[0], "→", test_months[-1])

    train_mask_time = df["month"].isin(train_months)
    val_mask_time = df["month"].isin(val_months)
    test_mask_time = df["month"].isin(test_months)

    train_mask = train_mask_time & (df["is_stockout_period"] == 0)
    val_mask = val_mask_time & (df["is_stockout_period"] == 0)
    test_mask = test_mask_time  # test берём как есть

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()

    print("Train size:", len(train_df))
    print("Val size  :", len(val_df))
    print("Test size :", len(test_df))
    print("Распределение is_stockout_period в test:")
    print(test_df["is_stockout_period"].value_counts())

    return train_df, val_df, test_df


def compute_category_calibration(
    df_val_log: pd.DataFrame,
    feature_cols_log,
    cat_feature_indices,
    model_cb_log: CatBoostRegressor,
    category_col: str = "category",
) -> Dict[str, float]:
    """
    Считает k по категориям на валидации (только без сток-аута).
    k_cat = Σ y_true / Σ y_pred по категории.
    """
    calib_df = df_val_log[df_val_log["is_stockout_period"] == 0].copy()
    calib_pool = Pool(
        data=calib_df[feature_cols_log],
        label=calib_df["target_log"],
        cat_features=cat_feature_indices,
    )

    calib_pred_log = model_cb_log.predict(calib_pool)
    calib_pred_qty = np.expm1(calib_pred_log)

    calib_df["y_true"] = calib_df["qty_month"].values
    calib_df["y_pred"] = calib_pred_qty

    cat_stats = (
        calib_df
        .groupby(category_col)
        .agg(
            y_true_sum=("y_true", "sum"),
            y_pred_sum=("y_pred", "sum"),
        )
        .reset_index()
    )

    cat_stats["k"] = cat_stats["y_true_sum"] / cat_stats["y_pred_sum"]
    cat_stats.loc[~np.isfinite(cat_stats["k"]), "k"] = 1.0

    print("\nКалибровочные коэффициенты по категориям:")
    print(cat_stats[[category_col, "k"]])

    calib_by_category = cat_stats.set_index(category_col)["k"].to_dict()
    return calib_by_category


def evaluate_before_after_calibration(
    df_log: pd.DataFrame,
    feature_cols_log,
    cat_feature_indices,
    model_cb_log: CatBoostRegressor,
    calib_by_category: Dict[str, float],
    category_col: str = "category",
):
    """
    Строит прогнозы на тесте и выводит метрики:
      - без калибровки,
      - с калибровкой по категориям.
    """

    test_pool_log = Pool(
        data=df_log[feature_cols_log],
        label=df_log["target_log"],
        cat_features=cat_feature_indices,
    )
    test_pred_log = model_cb_log.predict(test_pool_log)
    test_pred_qty = np.expm1(test_pred_log)

    df_pred = df_log.copy()
    df_pred["y_true"] = df_pred["qty_month"].values
    df_pred["y_pred"] = test_pred_qty

    # применяем k по category
    k_series = df_pred[category_col].map(calib_by_category).fillna(1.0)
    df_pred["y_pred_corr"] = df_pred["y_pred"] * k_series

    def eval_mask(mask, name: str):
        y_true = df_pred.loc[mask, "y_true"].values
        y_pred = df_pred.loc[mask, "y_pred"].values
        y_pred_corr = df_pred.loc[mask, "y_pred_corr"].values

        print(f"\n=== {name} (без калибровки) ===")
        print("RMSE:", rmse(y_true, y_pred))
        print("MAE :", mae(y_true, y_pred))
        print("MAPE:", mape(y_true, y_pred))
        print("WAPE:", wape(y_true, y_pred))
        print("Bias:", bias(y_true, y_pred))

        print(f"\n=== {name} (с калибровкой) ===")
        print("RMSE:", rmse(y_true, y_pred_corr))
        print("MAE :", mae(y_true, y_pred_corr))
        print("MAPE:", mape(y_true, y_pred_corr))
        print("WAPE:", wape(y_true, y_pred_corr))
        print("Bias:", bias(y_true, y_pred_corr))

    mask_clean = df_pred["is_stockout_period"] == 0
    mask_all = np.ones(len(df_pred), dtype=bool)

    eval_mask(mask_clean, "Test без сток-аута")
    eval_mask(mask_all, "Test полный")


# ========= Основная функция обучения =========

def train_model(
    model_dir: str = "../models",
    category_col: str = "category",
):
    # 1. Загружаем датасет из таблицы ml_monthly_base
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM ml_monthly_base", conn)
    conn.close()

    if "qty_month" not in df.columns:
        raise ValueError(
            "В таблице ml_monthly_base нет столбца 'qty_month'. "
            "Убедитесь, что перед этим был запущен ml_dataset_builder.py"
        )

    # 2. Делим на train / val / test по времени
    train_df, val_df, test_df = split_by_time(df)

    # 3. Лог-таргет и копии датафреймов
    df_log = df.copy()
    df_log["target_log"] = np.log1p(df_log["qty_month"])

    # 4. Формируем признаки и cat_features
    feature_cols_log, cat_feature_indices = build_feature_space(df_log)

    train_df_log = df_log.loc[train_df.index].copy()
    val_df_log = df_log.loc[val_df.index].copy()
    test_df_log = df_log.loc[test_df.index].copy()

    train_pool_log = Pool(
        data=train_df_log[feature_cols_log],
        label=train_df_log["target_log"],
        cat_features=cat_feature_indices,
    )
    val_pool_log = Pool(
        data=val_df_log[feature_cols_log],
        label=val_df_log["target_log"],
        cat_features=cat_feature_indices,
    )

    # 5. Обучаем финальную лог-модель
    # --- диагностика таргета на train ---
    print("\nДиагностика train-таргета:")
    print("train qty_month nunique:", train_df_log["qty_month"].nunique())
    print(train_df_log["qty_month"].value_counts().head())

    print("train target_log nunique:", train_df_log["target_log"].nunique())
    print(train_df_log["target_log"].value_counts().head())
    
    print("\n=== Обучение финальной лог-модели CatBoost ===")
    model_cb_log = CatBoostRegressor(**BEST_PARAMS_LOG)
    model_cb_log.fit(train_pool_log, eval_set=val_pool_log, verbose=100)

    # 6. Метрики на валидации (в штуках)
    val_pred_log = model_cb_log.predict(val_pool_log)
    val_pred_qty = np.expm1(val_pred_log)
    y_val = val_df_log["qty_month"].values

    print("\nLog-CatBoost (validation, в штуках):")
    print("RMSE:", rmse(y_val, val_pred_qty))
    print("MAE :", mae(y_val, val_pred_qty))
    print("MAPE:", mape(y_val, val_pred_qty))
    print("WAPE:", wape(y_val, val_pred_qty))
    print("Bias:", bias(y_val, val_pred_qty))

    # 7. Калибровка по категориям на валидации
    calib_by_category = compute_category_calibration(
        df_val_log=val_df_log,
        feature_cols_log=feature_cols_log,
        cat_feature_indices=cat_feature_indices,
        model_cb_log=model_cb_log,
        category_col=category_col,
    )

    # 8. Оценка на тесте до/после калибровки (для отчётности)
    print("\n=== Оценка на тесте до и после калибровки ===")
    evaluate_before_after_calibration(
        df_log=test_df_log,
        feature_cols_log=feature_cols_log,
        cat_feature_indices=cat_feature_indices,
        model_cb_log=model_cb_log,
        calib_by_category=calib_by_category,
        category_col=category_col,
    )

    # 9. Сохраняем модель и метаданные
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "catboost_demand_log.cbm")
    meta_path = os.path.join(model_dir, "model_meta.json")
    calib_path = os.path.join(model_dir, "calib_by_category.json")

    model_cb_log.save_model(model_path, format="cbm")

    meta = {
        "feature_cols_log": feature_cols_log,
        "cat_feature_indices": list(cat_feature_indices),
        "target_col_log": "target_log",
        "category_col": category_col,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    with open(calib_path, "w", encoding="utf-8") as f:
        json.dump(calib_by_category, f, ensure_ascii=False, indent=2)

    print(f"\nМодель сохранена в {model_path}")
    print(f"Метаданные сохранены в {meta_path}")
    print(f"Калибровка по категориям сохранена в {calib_path}")

    # 10. Дополнительно сохраняем калибровочные коэффициенты в таблицу calibration_category (SQLite)
    conn = get_connection()
    cur = conn.cursor()

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    rows = [
        (category, float(k), now)
        for category, k in calib_by_category.items()
    ]

    cur.executemany(
        """
        INSERT INTO calibration_category (category, k, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(category) DO UPDATE SET
            k = excluded.k,
            updated_at = excluded.updated_at;
        """,
        rows,
    )

    conn.commit()
    conn.close()

    print("Калибровка по категориям также сохранена в таблице calibration_category БД.")


if __name__ == "__main__":
    train_model()
