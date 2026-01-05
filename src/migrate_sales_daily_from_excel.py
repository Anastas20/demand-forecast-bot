from pathlib import Path
import pandas as pd

from db import DATA_DIR, init_db, insert_sales_daily

def migrate():
    excel_path = DATA_DIR / "SALES_ALL_DAILY.xlsx"
    if not excel_path.exists():
        raise FileNotFoundError(f"Файл {excel_path} не найден.")

    print(f"Читаем {excel_path}...")
    df = pd.read_excel(excel_path, parse_dates=["date"])
    
    expected = [
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

    print("Columns in the file:", list(df.columns))

    df = df[[c for c in df.columns if not str(c).startswith("Unnamed")]]
    
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"В sales_all_daily отсутствуют колонки {missing}")

    insert_sales_daily(df)


if __name__ == "__main__":
    init_db()
    migrate()
