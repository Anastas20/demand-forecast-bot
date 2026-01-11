"""
forecast_next_month.py

Скрипт для построения прогноза спроса на следующий месяц
с использованием сохранённой лог-модели CatBoost и
калибровочных коэффициентов по категориям.

По умолчанию прогноз считается на месяц, следующий
за последним месяцем, который есть в таблице ml_monthly_base
в SQLite-базе demand.db.

При явной передаче month_str можно построить прогноз/оценку
для конкретного месяца, который уже присутствует в ml_monthly_base.
"""

from pathlib import Path
import json
import os

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from db import get_connection  # читаем данные и калибровку из SQLite


# ---------- Базовые пути (корень проекта / data / models / forecasts) ----------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
FORECASTS_DIR = DATA_DIR / "forecasts"


# ---------- Вспомогательные функции ----------

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
    Определяет месяц, для которого строим ПРОГНОЗ/ОЦЕНКУ по уже имеющимся строкам.

    Используется только при явной передаче month_str.
    """
    df = df.copy()
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df = df[df["month"].notna()]

    if df.empty:
        raise ValueError("В ml_monthly_base нет корректных значений столбца 'month'.")

    if month_str is None:
        raise ValueError("month_str должен быть задан для choose_forecast_month().")

    forecast_month = pd.to_datetime(month_str)
    forecast_month = forecast_month + pd.offsets.MonthEnd(0)

    if forecast_month not in df["month"].unique():
        raise ValueError(
            f"В ml_monthly_base нет данных для месяца {forecast_month.date()}."
        )

    return forecast_month


def build_next_month_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """
    Строит датафрейм признаков для МЕСЯЦА, СЛЕДУЮЩЕГО за последним в ml_monthly_base.

    Логика:
    - находим последний месяц last_month в df['month'];
    - берём все строки этого месяца (df_last);
    - на их основе формируем df_next для next_month = last_month + 1 месяц,
      обновляя календарные признаки и частично сдвигая лаги.

    Возвращает:
    - df_next: датафрейм признаков для следующего месяца;
    - next_month: Timestamp конца следующего месяца;
    - last_month: Timestamp конца последнего месяца с фактами.
    """
    df = df.copy()
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df = df[df["month"].notna()]

    if df.empty:
        raise ValueError("В ml_monthly_base нет корректных значений столбца 'month'.")

    last_month = df["month"].max()
    next_month = last_month + pd.offsets.MonthEnd(1)

    df_last = df[df["month"] == last_month].copy()
    if df_last.empty:
        raise ValueError(f"Не удалось выделить строки за последний месяц {last_month.date()}.")

    df_next = df_last.copy()
    df_next["month"] = next_month

    # Календарные признаки
    year = next_month.year
    month_num = next_month.month

    if "year" in df_next.columns:
        df_next["year"] = year
    if "month_num" in df_next.columns:
        df_next["month_num"] = month_num
    if "is_q4" in df_next.columns:
        df_next["is_q4"] = (month_num in (10, 11, 12)).astype(int) if hasattr(month_num, "astype") else int(month_num in (10, 11, 12))
        # но проще:
        df_next["is_q4"] = int(month_num in (10, 11, 12))
    if "is_summer" in df_next.columns:
        df_next["is_summer"] = int(month_num in (6, 7, 8))

    # Лаги спроса: сдвигаем на один шаг вперёд там, где это логично и просто
    if "qty_month" in df_last.columns:
        if "qty_lag_1" in df_next.columns:
            df_next["qty_lag_1"] = df_last["qty_month"].values
        if "qty_lag_2" in df_next.columns and "qty_lag_1" in df_last.columns:
            df_next["qty_lag_2"] = df_last["qty_lag_1"].values
        if "qty_lag_3" in df_next.columns and "qty_lag_2" in df_last.columns:
            df_next["qty_lag_3"] = df_last["qty_lag_2"].values

    # Остальные лаги и скользящие/тренды переносим как приближение
    for col in [
        "qty_lag_6",
        "qty_lag_12",
        "qty_roll_mean_3",
        "qty_roll_mean_6",
        "qty_trend_3",
        "price_lag_1",
        "price_roll_mean_3",
    ]:
        if col in df_next.columns and col in df_last.columns:
            df_next[col] = df_last[col].values

    # Цена следующего месяца — приближённо на уровне последнего
    if "price_month" in df_next.columns and "price_month" in df_last.columns:
        df_next["price_month"] = df_last["price_month"].values

    # Сток-аут в будущем месяце заранее считаем отсутствующим (консервативно)
    if "is_stockout_period" in df_next.columns:
        df_next["is_stockout_period"] = 0

    # Факт спроса будущего месяца неизвестен
    if "qty_month" in df_next.columns:
        df_next["qty_month"] = np.nan

    return df_next, next_month, last_month


# ---------- Основная функция ----------

def forecast_for_month(month_str: str | None = None):
    """
    Строит прогноз:

    - если month_str is None:
        прогноз на МЕСЯЦ, СЛЕДУЮЩИЙ за последним в ml_monthly_base;
    - если month_str задан:
        прогноз/оценка для указанного месяца, который уже есть в ml_monthly_base.
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

    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df = df[df["month"].notna()]

    if df.empty:
        raise ValueError("В ml_monthly_base нет корректных значений столбца 'month'.")

    # 3. Определяем режим: следующий месяц или явный month_str
    if month_str is None:
        # Прогноз на следующий месяц после последнего месяца с фактами
        df_month, forecast_month, last_month = build_next_month_frame(df)
        print(
            f"Последний месяц с фактами в ml_monthly_base: {last_month.date()}\n"
            f"Строим прогноз на следующий месяц: {forecast_month.date()}"
        )
    else:
        # Прогноз/оценка для конкретного месяца, уже присутствующего в датасете
        forecast_month = choose_forecast_month(df, month_str=month_str)
        print(f"Строим прогноз/оценку для месяца: {forecast_month.date()}")
        df_month = df[df["month"] == forecast_month].copy()

    if df_month.empty:
        raise ValueError("После подготовки данных для прогноза получен пустой датафрейм.")

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

    # # 7. Информационное сообщение про наличие факта
    # if "qty_month" in df_month.columns and df_month["qty_month"].notna().any():
    #     print(
    #         f"В данных за {forecast_month.date()} есть фактические продажи – "
    #         f"можно сравнить прогноз и факт."
    #     )
    # else:
    #     print(
    #         f"Фактический спрос за {forecast_month.date()} отсутствует или неизвестен – "
    #         f"это прогноз на будущий месяц."
    #     )

    # 8. Сохраняем результат
    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
    out_name = f"forecast_{forecast_month.strftime('%Y_%m')}.xlsx"
    out_path = FORECASTS_DIR / out_name

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
    # вариант 1: прогноз на следующий месяц после последнего месяца с фактами
    forecast_for_month()

    # вариант 2: явный месяц (если нужно сравнить прогноз и факт для конкретного месяца)
    # forecast_for_month("2025-10-01")
    # forecast_for_month("2025-11-01")
