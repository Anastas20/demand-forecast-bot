# src/migrate_stockout_periods_from_old_excel.py

from pathlib import Path

import pandas as pd

from db import DATA_DIR, init_db, upsert_stockout_periods


def migrate():
    excel_path = DATA_DIR / "ML_MONTHLY_BASE.xlsx"
    if not excel_path.exists():
        raise FileNotFoundError(f"Файл {excel_path} не найден.")

    print(f"Читаем {excel_path}...")
    df = pd.read_excel(excel_path)

    # Проверяем наличие нужных колонок
    required = ["seller_sku", "marketplace", "month", "is_stockout_period"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"В ML_MONTHLY_BASE.xlsx отсутствуют колонки: {missing}")

    # Берём только строки, где is_stockout_period == 1
    so = df[df["is_stockout_period"] == 1].copy()
    print(f"Строк со сток-аутами в старом датасете: {len(so)}")

    if so.empty:
        print("В старом ML_MONTHLY_BASE.xlsx нет строк с is_stockout_period == 1.")
        return

    # Оставляем только нужные поля
    so = so[["seller_sku", "marketplace", "month", "is_stockout_period"]]

    upsert_stockout_periods(so)


if __name__ == "__main__":
    init_db()
    migrate()
