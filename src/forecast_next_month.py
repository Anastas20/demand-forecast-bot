"""
forecast_next_month.py

Скрипт для построения прогноза спроса на последний месяц
с использованием сохранённой лог-модели CatBoost и
калибровочных коэффициентов по категориям.

Прогноз считается для выбранного месяца
(по умолчанию – для последнего месяца, который есть
в таблице ml_monthly_base в SQLite-базе demand.db).
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from db import get_connection  # читаем данные и калибровку из SQLite


# ---------- Базовые пути (корень проекта / data / models / forecasts) ----------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
FORECASTS_DIR = DATA_DIR / "forecasts"


# ---------- Функции ----------

def load_model_and_meta():
    """
    Загружает CatBoost-модель, метаданные и калибровку по категориям.

    Калибровка берётся из таблицы calibration_category, при ошибке – из calib_by_category.json.
    """
    model_path = MODELS_DIR / "catboost_demand_log.cbm"
    meta_path = MODELS_DIR / "model_meta.json"
    calib_json_path = MODELS_DIR / "calib_by_category.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Не найдена модель {model_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Не найден файл метаданных {meta_path}")

    model = CatBoostRegressor()
    model.load_model(model_path.as_posix())

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    feature_cols_log = meta["feature_cols_log"]
    cat_feature_indices = meta["cat_feature_indices"]
    category_col = meta.get("category_col", "category")

    # --- калибровка по категориям: сначала пробуем взять из БД ---
    calib_by_category = None
    try:
        conn = get_connection()
        calib_df = pd.read_sql_query(
            "SELECT category, k FROM calibration_category",
            conn,
        )
        conn.close()
        if not calib_df.empty:
            calib_by_category = dict(
                zip(calib_df["category"], calib_df["k"])
            )
            print(
                f"Загружено {len(calib_by_category)} k по категориям "
                f"из таблицы calibration_category."
            )
    except Exception as e:
        print(f"Не удалось прочитать calibration_category из БД: {e}")

    # если из БД ничего не получилось – читаем JSON, как раньше
    if calib_by_category is None:
        if not calib_json_path.exists():
            raise FileNotFoundError(
                f"Не найдено ни таблицы calibration_category с данными, "
                f"ни файла калибровки {calib_json_path}"
            )
        with open(calib_json_path, "r", encoding="utf-8") as f:
            calib_by_category = json.load(f)
        print(
            f"Калибровка по категориям загружена из {calib_json_path} "
            f"({len(calib_by_category)} категорий)."
        )

    return model, feature_cols_log, cat_feature_indices, category_col, calib_by_category


def choose_forecast_month(df: pd.DataFrame, month_str: str | None = None) -> pd.Timestamp:
    """
    Определяет месяц, на который считаем прогноз.

    Если month_str не задан, берём последний месяц из датасета.
    month_str можно передать в формате '2025-10-31' или '2025-10-01'.
    """
    df = df.copy()
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df = df[df["month"].notna()]

    if df.empty:
        raise ValueError("В ml_monthly_base нет корректных значений столбца 'month'.")

    if month_str is None:
        forecast_month = df["month"].max()
    else:
        forecast_month = pd.to_datetime(month_str)
        # нормализуем к концу месяца (как в ml_dataset_builder)
        forecast_month = forecast_month + pd.offsets.MonthEnd(0)

    if forecast_month not in df["month"].unique():
        raise ValueError(
            f"В ml_monthly_base нет данных для месяца {forecast_month.date()}."
        )

    return forecast_month


def forecast_for_month(month_str: str | None = None):
    """
    Основная функция: строит прогноз для заданного месяца
    (или для последнего месяца, если month_str=None)
    и сохраняет результат в Excel.
    """

    # 1. Загружаем модель, метаданные и калибровку
    model, feature_cols_log, cat_feature_indices, category_col, calib_by_category = (
        load_model_and_meta()
    )

    # 2. Загружаем датасет из таблицы ml_monthly_base
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM ml_monthly_base", conn)
    conn.close()

    if df.empty:
        raise ValueError(
            "Таблица ml_monthly_base пуста. "
            "Сначала нужно собрать датасет через ml_dataset_builder.py."
        )

    # 3. Выбираем месяц для прогноза
    forecast_month = choose_forecast_month(df, month_str=month_str)
    print(f"Строим прогноз для месяца: {forecast_month.date()}")

    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df_month = df[df["month"] == forecast_month].copy()

    if df_month.empty:
        raise ValueError("После фильтрации по month датафрейм пустой.")

    # 4. Проверяем, что все признаки присутствуют
    missing = set(feature_cols_log) - set(df_month.columns)
    if missing:
        raise ValueError(
            f"В данных для прогноза отсутствуют столбцы признаков: {missing}"
        )

    # 5. Делаем прогноз (лог-пространство -> штуки)
    pool = Pool(
        data=df_month[feature_cols_log],
        cat_features=cat_feature_indices,
    )

    pred_log = model.predict(pool)
    y_pred = np.expm1(pred_log)
    df_month["y_pred"] = y_pred

    # 6. Применяем калибровку по категориям
    k_series = df_month[category_col].map(calib_by_category).fillna(1.0)
    df_month["k_category"] = k_series
    df_month["y_pred_corr"] = df_month["y_pred"] * df_month["k_category"]

    # 7. Если есть фактический спрос, можно посмотреть, есть ли что сравнивать
    if "qty_month" in df_month.columns and df_month["qty_month"].notna().any():
        print(
            f"В ml_monthly_base есть фактические продажи за {forecast_month.date()} – "
            f"можно затем сравнить прогноз и факт."
        )
    else:
        print(
            "Столбец 'qty_month' отсутствует или пустой — считаем, что это прогноз на будущий месяц."
        )

    # 8. Сохраняем результат
    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
    out_name = f"forecast_{forecast_month.strftime('%Y_%m')}.xlsx"
    out_path = FORECASTS_DIR / out_name

    # Оставим в файле ключевые столбцы + прогноз
    cols_for_output = []
    for col in [
        "month",
        "seller_sku",
        "marketplace",
        category_col,
        "qty_month",      # фактические продажи, если есть
        "y_pred",         # прогноз до калибровки
        "k_category",     # коэффициент калибровки
        "y_pred_corr",    # откалиброванный прогноз
    ]:
        if col in df_month.columns:
            cols_for_output.append(col)

    df_month[cols_for_output].to_excel(out_path, index=False)
    print(f"Файл с прогнозом сохранён: {out_path}")
    
    return out_path


if __name__ == "__main__":
    # вариант 1: прогноз для последнего месяца в ml_monthly_base
    forecast_for_month()

    # вариант 2: можно явно указать месяц, раскомментировав строку ниже:
    # forecast_for_month("2025-11-01")
