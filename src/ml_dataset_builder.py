"""
ml_dataset_builder.py

Строит месячный ML-датасет ml_monthly_base из таблиц SQLite:
- sales_daily
- products_stats
- stockout_periods

Учитывает, что часть числовых полей в БД сохранена как байты (BLOB),
и переводит их обратно в целые числа.

Дополнительно:
- отфильтровывает SKU с очень маленькой историей продаж, чтобы
  не засорять модель длинным «хвостом».
"""

from pathlib import Path
import numpy as np
import pandas as pd

from db import get_connection


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# --- Пороговые значения для отбора SKU в ML-датасет ---
# Можно менять по необходимости:
MIN_TOTAL_QTY_FOR_MODEL = 0   # минимальная суммарная продажа SKU за весь период
MIN_DAYS_WITH_SALES_FOR_MODEL = 0 # минимальное число дней с продажами


def bytes_to_int(val):
    """
    Преобразование значения, сохранённого как байты b'..', в целое число.
    Если это уже число или строка — возвращаем как есть.
    """
    if isinstance(val, (bytes, bytearray)):
        return int.from_bytes(val, byteorder="little", signed=False)
    return val


def build_ml_monthly_base():
    conn = get_connection()

    # ------------------------------------------------------------------
    # 1. Читаем посуточные продажи и приводим quantity к числу
    # ------------------------------------------------------------------
    sales = pd.read_sql_query(
        "SELECT * FROM sales_daily",
        conn,
        parse_dates=["date"],
    )

    # quantity в БД сейчас хранится как байты → переводим в int
    sales["quantity"] = sales["quantity"].apply(bytes_to_int)
    sales["quantity"] = pd.to_numeric(sales["quantity"], errors="coerce").fillna(0)

    # месяц как конец месяца
    sales["month"] = sales["date"].dt.to_period("M").dt.to_timestamp("M")

    # агрегируем в помесячные продажи
    monthly = (
        sales.groupby(["seller_sku", "marketplace", "month"], as_index=False)
        .agg(
            qty_month=("quantity", "sum"),
            revenue_month=("revenue", "sum"),
        )
    )

    # ------------------------------------------------------------------
    # 2. Достраиваем "полную сетку" (seller_sku, marketplace, month)
    #    чтобы были строки и в те месяцы, когда продаж не было (0)
    # ------------------------------------------------------------------
    all_months = np.sort(monthly["month"].unique())
    pairs = monthly[["seller_sku", "marketplace"]].drop_duplicates()

    months_df = pd.DataFrame({"month": all_months})
    pairs["key"] = 1
    months_df["key"] = 1

    full = pairs.merge(months_df, on="key").drop(columns="key")

    df = full.merge(
        monthly,
        on=["seller_sku", "marketplace", "month"],
        how="left",
    )

    df["qty_month"] = df["qty_month"].fillna(0)
    df["revenue_month"] = df["revenue_month"].fillna(0)

    df["price_month"] = np.where(
        df["qty_month"] > 0,
        df["revenue_month"] / df["qty_month"],
        np.nan,
    )

    # ------------------------------------------------------------------
    # 3. Подтягиваем product-статистику из products_stats
    #    и переводим байты → числа
    # ------------------------------------------------------------------
    prod = pd.read_sql_query("SELECT * FROM products_stats", conn)

    for col in ["total_quantity", "days_with_sales", "is_new"]:
        if col in prod.columns:
            prod[col] = prod[col].apply(bytes_to_int)
            prod[col] = pd.to_numeric(prod[col], errors="coerce").fillna(0)

    if "is_new" in prod.columns:
        prod["is_new"] = prod["is_new"].astype(int)
    if "days_with_sales" in prod.columns:
        prod["days_with_sales"] = prod["days_with_sales"].astype(int)

    # тип маркетплейсов (Ozon / WB / Ozon+WB / UNKNOWN)
    if "marketplaces" in prod.columns:
        def map_marketplaces(m):
            s = (m or "").strip()
            if not s:
                return "UNKNOWN"
            s_low = s.lower()
            has_ozon = "ozon" in s_low
            has_wb = ("wb" in s_low) or ("wildberries" in s_low)
            if has_ozon and has_wb:
                return "Ozon+WB"
            if has_ozon:
                return "Ozon"
            if has_wb:
                return "WB"
            return "UNKNOWN"

        prod["marketplaces_type"] = prod["marketplaces"].apply(map_marketplaces)
    else:
        prod["marketplaces_type"] = "UNKNOWN"

    prod_cols = [
        "seller_sku",
        "product_name",
        "category",
        "subcategory",
        "family_id",
        "is_new",
        "total_quantity",
        "total_revenue",
        "first_date",
        "last_date",
        "days_with_sales",
        "marketplaces_type",
    ]
    prod_use = prod[[c for c in prod_cols if c in prod.columns]].copy()

    df = df.merge(prod_use, on="seller_sku", how="left")

    # ------------------------------------------------------------------
    # 4. Календарные признаки
    # ------------------------------------------------------------------
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df = df[df["month"].notna()].copy()

    df["year"] = df["month"].dt.year.astype(int)
    df["month_num"] = df["month"].dt.month.astype(int)
    df["is_q4"] = df["month_num"].isin([10, 11, 12]).astype(int)
    df["is_summer"] = df["month_num"].isin([6, 7, 8]).astype(int)

    # ------------------------------------------------------------------
    # 5. Сортировка и лаги / скользящие средние
    # ------------------------------------------------------------------
    df = df.sort_values(["seller_sku", "marketplace", "month"]).reset_index(drop=True)

    group = df.groupby(["seller_sku", "marketplace"], group_keys=False)

    for lag in [1, 2, 3, 6, 12]:
        df[f"qty_lag_{lag}"] = group["qty_month"].shift(lag)

    df["qty_roll_mean_3"] = (
        group["qty_month"]
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )
    df["qty_roll_mean_6"] = (
        group["qty_month"]
        .rolling(window=6, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    # Простейший тренд за 3 месяца: текущее значение минус среднее за последние 3
    def compute_trend(s: pd.Series) -> pd.Series:
        vals = s.values.astype(float)
        out = np.full_like(vals, np.nan, dtype=float)
        for i in range(len(vals)):
            start = max(0, i - 2)
            window = vals[start : i + 1]
            if len(window) == 3:
                out[i] = vals[i] - window.mean()
        return pd.Series(out, index=s.index)

    df["qty_trend_3"] = group["qty_month"].apply(compute_trend)

    df["price_lag_1"] = group["price_month"].shift(1)
    df["price_roll_mean_3"] = (
        group["price_month"]
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    # ------------------------------------------------------------------
    # 6. Флаг сток-аута:
    #    - эвристика по данным
    #    - плюс явные периоды из stockout_periods
    # ------------------------------------------------------------------
    df["is_stockout_period"] = 0

    # эвристика: не новый товар, продаж в месяц нет, а среднее за 3 месяца ≥ 3
    df["is_stockout_period"] = (
        (df["is_new"].fillna(0) == 0)
        & (df["qty_month"] == 0)
        & (df["qty_roll_mean_3"].fillna(0) >= 3)
    ).astype(int)

    # переопределяем по таблице stockout_periods (если есть)
    try:
        stock = pd.read_sql_query(
            "SELECT seller_sku, marketplace, month, is_stockout_period FROM stockout_periods",
            conn,
        )
        stock["month"] = pd.to_datetime(stock["month"], errors="coerce")
        stock = stock[stock["month"].notna()]

        stock["is_stockout_period"] = stock["is_stockout_period"].apply(bytes_to_int)
        stock["is_stockout_period"] = (
            pd.to_numeric(stock["is_stockout_period"], errors="coerce")
            .fillna(0)
            .astype(int)
        )

        df = df.merge(
            stock,
            on=["seller_sku", "marketplace", "month"],
            how="left",
            suffixes=("", "_from_table"),
        )

        df["is_stockout_period"] = np.where(
            df["is_stockout_period_from_table"].notna(),
            df["is_stockout_period_from_table"],
            df["is_stockout_period"],
        ).astype(int)

        df = df.drop(columns=["is_stockout_period_from_table"])
    except Exception as e:
        print("Не удалось учесть stockout_periods:", e)

    # ------------------------------------------------------------------
    # 7. Финальный порядок колонок и приведение типов
    # ------------------------------------------------------------------
    col_order = [
        "seller_sku",
        "marketplace",
        "month",
        "qty_month",
        "revenue_month",
        "price_month",
        "product_name",
        "category",
        "subcategory",
        "family_id",
        "is_new",
        "total_quantity",
        "total_revenue",
        "first_date",
        "last_date",
        "days_with_sales",
        "marketplaces_type",
        "year",
        "month_num",
        "is_q4",
        "is_summer",
        "qty_lag_1",
        "qty_lag_2",
        "qty_lag_3",
        "qty_lag_6",
        "qty_lag_12",
        "qty_roll_mean_3",
        "qty_roll_mean_6",
        "qty_trend_3",
        "price_lag_1",
        "price_roll_mean_3",
        "is_stockout_period",
    ]

    int_defaults = [
        "is_new",
        "days_with_sales",
        "year",
        "month_num",
        "is_q4",
        "is_summer",
        "is_stockout_period",
    ]

    for col in col_order:
        if col not in df.columns:
            if col in int_defaults:
                df[col] = 0
            else:
                df[col] = np.nan

    df_out = df[col_order].copy()

    for col in int_defaults:
        df_out[col] = pd.to_numeric(df_out[col], errors="coerce").fillna(0).astype(int)

    # ------------------------------------------------------------------
    # 8. Фильтрация SKU с очень малой историей продаж
    # ------------------------------------------------------------------
    # Оставляем только те SKU, у которых:
    # - суммарные продажи за весь период >= MIN_TOTAL_QTY_FOR_MODEL
    # - число дней с продажами >= MIN_DAYS_WITH_SALES_FOR_MODEL
    mask_keep = (
        (df_out["total_quantity"].fillna(0) >= MIN_TOTAL_QTY_FOR_MODEL)
        & (df_out["days_with_sales"].fillna(0) >= MIN_DAYS_WITH_SALES_FOR_MODEL)
    )

    df_model = df_out[mask_keep].copy()

    print(
        f"Всего строк в ml_monthly_base до фильтрации: {len(df_out)}; "
        f"после фильтрации по активности SKU: {len(df_model)}"
    )

    # ------------------------------------------------------------------
    # 9. Сохраняем в таблицу ml_monthly_base
    # ------------------------------------------------------------------
    with conn:
        conn.execute("DELETE FROM ml_monthly_base")
        df_model.to_sql("ml_monthly_base", conn, if_exists="append", index=False)

    conn.close()

    print(f"Собран ml_monthly_base: {df_model.shape[0]} строк, {df_model.shape[1]} колонок.")
    print("Пример распределения qty_month (после фильтрации):")
    print(df_model["qty_month"].describe())


if __name__ == "__main__":
    build_ml_monthly_base()
