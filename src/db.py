from pathlib import Path
import sqlite3
from typing import Optional

import pandas as pd

# путь на корень проекта
BASE_DIR = Path(__file__).resolve().parent.parent 
# путь к BASE_DIR/data
DATA_DIR = BASE_DIR / "data"
# путь к файлу базы данных
DB_PATH = DATA_DIR / "demand.db"

def get_connection() -> sqlite3.Connection:
    """
    Возвращает соединение с базой данных.
    Если каталога data/ нет - создаёт его.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH.as_posix())
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """
    Создает необходимые таблицы, если их нет.
    Таблицы:
      - sales_daily       -ежедневные продажи
      - ml_monthly_base    - месячный ML-датаcет (все признаки, которые ты перечислила)
      - products_stats     - агрегированная статистика по товарам (аналог products_with_stats_updated.xlsx)
      - calibration_category -калибровочные коэффициенты k по категориям
    """
    conn = get_connection()
    cur = conn.cursor()

    # Ежедневные продажи
    cur.execute("""
            CREATE TABLE IF NOT EXISTS sales_daily (
            date TEXT NOT NULL,
            seller_sku TEXT NOT NULL,
            marketplace TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            price REAL,
            revenue REAL,
            wb_agreed_discount_pct REAL,
            wb_final_discount_pct REAL,
            wb_price_with_disc REAL,
            wb_spp_pct REAL,
            PRIMARY KEY (date, seller_sku, marketplace)
        );
        """
    )
        # Месячный датасет
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ml_monthly_base (
            seller_sku TEXT NOT NULL,
            marketplace TEXT NOT NULL,
            month TEXT NOT NULL,           -- например '2025-10-31'
            qty_month REAL,
            revenue_month REAL,
            price_month REAL,
            product_name TEXT,
            category TEXT,
            subcategory TEXT,
            family_id TEXT,
            is_new INTEGER,                -- 0/1
            total_quantity REAL,
            total_revenue REAL,
            first_date TEXT,               -- 'YYYY-MM-DD'
            last_date TEXT,                -- 'YYYY-MM-DD'
            days_with_sales INTEGER,
            marketplaces_type TEXT,
            year INTEGER,
            month_num INTEGER,
            is_q4 INTEGER,                 -- 0/1
            is_summer INTEGER,             -- 0/1
            qty_lag_1 REAL,
            qty_lag_2 REAL,
            qty_lag_3 REAL,
            qty_lag_6 REAL,
            qty_lag_12 REAL,
            qty_roll_mean_3 REAL,
            qty_roll_mean_6 REAL,
            qty_trend_3 REAL,
            price_lag_1 REAL,
            price_roll_mean_3 REAL,
            is_stockout_period INTEGER,    -- 0/1
            PRIMARY KEY (seller_sku, marketplace, month)
        );
        """
    )
    # seller_sku	marketplace	month	qty_month	
    # revenue_month	price_month	product_name	category	
    # subcategory	family_id	is_new	total_quantity	
    # total_revenue	first_date	last_date	days_with_sales	
    # marketplaces_type	year	month_num	is_q4	
    # is_summer	qty_lag_1	qty_lag_2	qty_lag_3	
    # qty_lag_6	qty_lag_12	qty_roll_mean_3	qty_roll_mean_6	
    # qty_trend_3	price_lag_1	price_roll_mean_3	
    # is_stockout_period

     # Статистика по товарам (аналог products_with_stats_updated)

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS products_stats (
            seller_sku TEXT PRIMARY KEY,
            product_name TEXT,
            category TEXT,
            subcategory TEXT,
            total_quantity REAL,
            total_revenue REAL,
            first_date TEXT,           -- 'YYYY-MM-DD'
            last_date TEXT,            -- 'YYYY-MM-DD'
            days_with_sales INTEGER,
            marketplaces TEXT,
            is_new INTEGER,  
            family_id TEXT
        );
        """
    )
    # seller_sku	product_name	category	subcategory	total_quantity	
    # total_revenue	first_date	last_date	days_with_sales	marketplaces	
    # is_new	family_id


    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS calibration_category (
        category TEXT PRIMARY KEY,
        k REAL NOT NULL,
        updated_at TEXT NOT NULL
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS stockout_periods (
            seller_sku TEXT NOT NULL,
            marketplace TEXT NOT NULL,
            month TEXT NOT NULL,            -- 'YYYY-MM-DD', конец месяца
            is_stockout_period INTEGER NOT NULL,  -- 0/1
            PRIMARY KEY (seller_sku, marketplace, month)
        );
        """
    )


    conn.commit()
    conn.close()
    print(f"Инициализирована база данных: {DB_PATH}")

def insert_sales_daily(df: pd.DataFrame):
    """
    Вставляет/обновляет записи в sales_daily из DataFrame.
    """
    if df is None or df.empty:
        print("insert_sales_daily: пустой DataFrame, нечего записывать.")
        return

    df = df.copy()   

    # Дата → строка YYYY-MM-DD
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df[df['date'].notna()].copy()
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')

    required_cols = ['date', 'seller_sku', 'marketplace', 'quantity', 'revenue']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"insert_sales_daily: в DataFrame отсутствует колонка: {col}")

    if "price" not in df.columns:
        df["price"] = df.apply(
            lambda row: (row["revenue"] / row["quantity"] if row["quantity"] > 0 else None),
            axis=1,
        )

    extra_cols = [
        "wb_agreed_discount_pct",
        "wb_final_discount_pct",
        "wb_price_with_disc",
        "wb_spp_pct",
    ]
    for col in extra_cols:
        if col not in df.columns:
            df[col] = None

    records = df[
        [
            "date",
            "seller_sku",
            "marketplace",
            "quantity",
            "price",
            "revenue",
            "wb_agreed_discount_pct",
            "wb_final_discount_pct",
            "wb_price_with_disc",
            "wb_spp_pct",
        ]
    ].to_records(index=False)

    conn = get_connection()
    cur = conn.cursor()

    cur.executemany(
        """
        INSERT INTO sales_daily (
            date, seller_sku, marketplace,
            quantity, price, revenue,
            wb_agreed_discount_pct, wb_final_discount_pct,
            wb_price_with_disc, wb_spp_pct
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(date, seller_sku, marketplace)
        DO UPDATE SET
            quantity = excluded.quantity,
            price = excluded.price,
            revenue = excluded.revenue,
            wb_agreed_discount_pct = excluded.wb_agreed_discount_pct,
            wb_final_discount_pct = excluded.wb_final_discount_pct,
            wb_price_with_disc = excluded.wb_price_with_disc,
            wb_spp_pct = excluded.wb_spp_pct;
        """,
        list(records), # список кортежей, каждый кортеж = один набор параметров
    )
    conn.commit()
    conn.close()
    print(f"Вставлено/обновлено {len(df)} строк в sales_daily")

def upsert_products_stats(df: pd.DataFrame):
    """
    Запись/обновление агрегированной статистики по товарам в таблицу products_stats.

    Ожидаемые колонки в df (из products_with_stats_updated.xlsx):
      seller_sku, product_name, category, subcategory,
      total_quantity, total_revenue, first_date, last_date,
      days_with_sales, marketplaces,
      is_new, family_id
    """
    if df is None or df.empty:
        print("upsert_products_stats: пустой DataFrame, нечего записывать.")
        return

    df = df.copy()

    required_cols = [
        "seller_sku",
        "product_name",
        "category",
        "subcategory",
        "total_quantity",
        "total_revenue",
        "first_date",
        "last_date",
        "days_with_sales",
        "marketplaces",
        "is_new",
        "family_id",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"upsert_products_stats: отсутствуют колонки: {missing}")

    # Даты → строки 'YYYY-MM-DD'
    for col in ["first_date", "last_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        df[col] = df[col].dt.strftime("%Y-%m-%d")

    # is_new → 0/1
    df["is_new"] = df["is_new"].fillna(0).astype(int)

    records = df[
        [
            "seller_sku",
            "product_name",
            "category",
            "subcategory",
            "total_quantity",
            "total_revenue",
            "first_date",
            "last_date",
            "days_with_sales",
            "marketplaces",
            "is_new",
            "family_id",
        ]
    ].to_records(index=False)

    conn = get_connection()
    cur = conn.cursor()

    cur.executemany(
        """
        INSERT INTO products_stats (
            seller_sku,
            product_name,
            category,
            subcategory,
            total_quantity,
            total_revenue,
            first_date,
            last_date,
            days_with_sales,
            marketplaces,
            is_new,
            family_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(seller_sku) DO UPDATE SET
            product_name     = excluded.product_name,
            category         = excluded.category,
            subcategory      = excluded.subcategory,
            total_quantity   = excluded.total_quantity,
            total_revenue    = excluded.total_revenue,
            first_date       = excluded.first_date,
            last_date        = excluded.last_date,
            days_with_sales  = excluded.days_with_sales,
            marketplaces     = excluded.marketplaces,
            is_new           = excluded.is_new,
            family_id        = excluded.family_id;
        """,
        list(records),
    )

    conn.commit()
    conn.close()
    print(f"upsert_products_stats: вставлено/обновлено строк: {len(df)}")

def upsert_stockout_periods(df: pd.DataFrame):
    """
    Записывает/обновляет разметку сток-аутов в таблицу stockout_periods.

    Ожидаемые колонки:
      - seller_sku
      - marketplace
      - month  (строка 'YYYY-MM-DD' или datetime)
      - is_stockout_period (0/1)
    """
    if df is None or df.empty:
        print("upsert_stockout_periods: пустой DataFrame, нечего записывать.")
        return

    df = df.copy()

    # sku и marketplace — в едином формате (нижний регистр, без пробелов)
    df["seller_sku"] = df["seller_sku"].astype(str).str.strip().str.lower()
    df["marketplace"] = df["marketplace"].astype(str).str.strip().str.upper()

    # month → строка 'YYYY-MM-DD'
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df = df[df["month"].notna()].copy()
    df["month"] = df["month"].dt.strftime("%Y-%m-%d")

    df["is_stockout_period"] = df["is_stockout_period"].fillna(0).astype(int)

    records = df[["seller_sku", "marketplace", "month", "is_stockout_period"]].to_records(index=False)

    conn = get_connection()
    cur = conn.cursor()

    cur.executemany(
        """
        INSERT INTO stockout_periods (
            seller_sku, marketplace, month, is_stockout_period
        )
        VALUES (?, ?, ?, ?)
        ON CONFLICT(seller_sku, marketplace, month) DO UPDATE SET
            is_stockout_period = excluded.is_stockout_period;
        """,
        list(records),
    )

    conn.commit()
    conn.close()
    print(f"upsert_stockout_periods: вставлено/обновлено строк: {len(df)}")



if __name__ == "__main__":
    init_db()