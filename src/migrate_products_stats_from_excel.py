# src/migrate_products_stats_from_excel.py

from pathlib import Path

import pandas as pd

from db import DATA_DIR, init_db, upsert_products_stats


def migrate():
    excel_path = DATA_DIR / "products_with_stats_updated.xlsx"
    if not excel_path.exists():
        raise FileNotFoundError(f"Файл {excel_path} не найден.")

    print(f"Читаем {excel_path}...")
    df = pd.read_excel(excel_path)

    print("Колонки в products_with_stats_updated:", list(df.columns))

    # Убираем возможные технические колонки типа 'Unnamed: 0'
    df = df[[c for c in df.columns if not str(c).startswith("Unnamed")]]

    upsert_products_stats(df)


if __name__ == "__main__":
    init_db()
    migrate()
