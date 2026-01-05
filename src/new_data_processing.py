"""
new_data_processing.py

Обработка новых недельных отчётов Wildberries и Ozon:

- чтение сырых Excel-файлов;
- очистка и выбор нужных столбцов;
- приведение к единому формату (дневные продажи по SKU и маркетплейсу);
- фильтрация по релевантным товарам из products_with_stats_updated.xlsx;
- запись результатов в таблицу sales_daily в SQLite-базе demand.db.
"""

from __future__ import annotations

from pathlib import Path
import argparse
from typing import Set, Optional

import pandas as pd
import numpy as np

from db import BASE_DIR, DATA_DIR, init_db, insert_sales_daily

PRODUCTS_PATH = DATA_DIR / "products_with_stats_updated.xlsx"


# ---------- Вспомогательные функции ----------

def detect_sku_column(df: pd.DataFrame) -> str:
    """
    Определяет, какой столбец в products_with_stats_updated хранит артикул продавца.

    Допустимые варианты: 'seller_sku', 'sku', 'Артикул продавца', 'Артикул'.
    При необходимости можно расширить список.
    """
    candidates = ["seller_sku", "sku", "Артикул продавца", "Артикул"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        f"Не удалось определить колонку с артикулом в products_with_stats_updated. "
        f"Ожидались столбцы из {candidates}, фактически: {list(df.columns)}"
    )


def load_relevant_skus(products_path: Path = PRODUCTS_PATH) -> Set[str]:
    """
    Загружает список релевантных SKU из products_with_stats_updated.xlsx.
    Используется только для фильтрации; сами статистики в этом скрипте не пересчитываются.
    """
    if not products_path.exists():
        raise FileNotFoundError(f"Не найден файл с товарами: {products_path}")

    prod_df = pd.read_excel(products_path)
    sku_col = detect_sku_column(prod_df)

    skus = (
        prod_df[sku_col]
        .astype(str)
        .str.strip()
        .dropna()
        .unique()
        .tolist()
    )
    relevant_skus: Set[str] = set(skus)

    print(f"Загружено {len(relevant_skus)} релевантных SKU из {products_path.name}")
    return relevant_skus


# ---------- Обработка отчёта Wildberries ----------

def process_wb_report(
    report_path: str | Path,
    relevant_skus: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """
    Читает недельный отчёт Wildberries и возвращает DataFrame в формате:

        date, seller_sku, marketplace, qty_day, revenue_day

    Использует:
      - лист, название которого начинается с "Отчет"
      - header=1 (так как первая строка – заголовок отчёта)
      - колонки:
            'Артикул продавца'
            'Дата'
            'Выкупили, шт'
            'Выкупили на сумму, ₽'
    """
    report_path = Path(report_path)
    if not report_path.exists():
        raise FileNotFoundError(f"Файл отчёта WB не найден: {report_path}")

    xls = pd.ExcelFile(report_path)

    sheet_name = None
    for s in xls.sheet_names:
        if str(s).startswith("Отчет"):
            sheet_name = s
            break
    if sheet_name is None:
        sheet_name = xls.sheet_names[0]

    print(f"[WB] Читаем лист '{sheet_name}' из файла {report_path.name}")
    df = pd.read_excel(xls, sheet_name=sheet_name, header=1)

    required_cols = ["Артикул продавца", "Дата", "Выкупили, шт", "Выкупили на сумму, ₽"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[WB] В отчёте отсутствуют столбцы: {missing}")

    # Оставляем только строки, где есть выкупленные штуки
    df = df[df["Выкупили, шт"] > 0].copy()

    # Базовое приведение типов
    df["seller_sku"] = df["Артикул продавца"].astype(str).str.strip().str.lower()
    df["date"] = pd.to_datetime(df["Дата"], errors="coerce")
    df = df[df["date"].notna()].copy()

    df["qty"] = df["Выкупили, шт"].fillna(0).astype(int)
    df["revenue"] = df["Выкупили на сумму, ₽"].fillna(0).astype(float)
    df["marketplace"] = "WB"

    if relevant_skus is not None:
        before = len(df)
        df = df[df["seller_sku"].isin(relevant_skus)].copy()
        after = len(df)
        print(f"[WB] Фильтрация по SKU: {before} → {after} строк")

    # Агрегируем по дню, SKU и маркетплейсу
    agg = (
        df.groupby(["date", "seller_sku", "marketplace"], as_index=False)
          .agg(
              quantity=("qty", "sum"),
              revenue=("revenue", "sum"),
          )
    )

    # средняя цена за день
    agg["price"] = np.where(
        agg["quantity"] > 0,
        agg["revenue"] / agg["quantity"],
        np.nan,
    )

    print(f"[WB] После агрегации строк: {len(agg)}")
    return agg


# ---------- Обработка отчёта Ozon ----------

def process_ozon_report(
    report_path: str | Path,
    relevant_skus: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """
    Читает недельный отчёт Ozon и возвращает DataFrame в формате:

        date, seller_sku, marketplace, qty_day, revenue_day

    Логика:
      - выбираем лист, название которого начинается с "Отчет";
      - фильтруем строки:
            'Группа услуг' == 'Продажи'
            'Тип начисления' == 'Выручка'
      - используем колонки:
            'Артикул'
            'Дата начисления'
            'Количество'
            'Сумма итого, руб.'
    """
    report_path = Path(report_path)
    if not report_path.exists():
        raise FileNotFoundError(f"Файл отчёта Ozon не найден: {report_path}")

    xls = pd.ExcelFile(report_path)

    sheet_name = None
    for s in xls.sheet_names:
        if str(s).startswith("Отчет"):
            sheet_name = s
            break
    if sheet_name is None:
        sheet_name = xls.sheet_names[0]

    print(f"[OZON] Читаем лист '{sheet_name}' из файла {report_path.name}")
    df = pd.read_excel(xls, sheet_name=sheet_name)

    required_cols = ["Артикул", "Дата начисления", "Количество", "Сумма итого, руб."]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[OZON] В отчёте отсутствуют столбцы: {missing}")

    # Оставляем только строки с продажами (выручка)
    mask_sales = (df["Группа услуг"] == "Продажи") & (df["Тип начисления"] == "Выручка")
    df = df[mask_sales].copy()

    # Приведение типов
    df["seller_sku"] = df["Артикул"].astype(str).str.strip().str.lower()
    df["date"] = pd.to_datetime(df["Дата начисления"], errors="coerce")
    df = df[df["date"].notna()].copy()

    df["qty"] = df["Количество"].fillna(0).astype(int)
    df["revenue"] = df["Сумма итого, руб."].fillna(0).astype(float)
    df["marketplace"] = "OZON"

    if relevant_skus is not None:
        before = len(df)
        df = df[df["seller_sku"].isin(relevant_skus)].copy()
        after = len(df)
        print(f"[OZON] Фильтрация по SKU: {before} → {after} строк")

    agg = (
        df.groupby(["date", "seller_sku", "marketplace"], as_index=False)
          .agg(
              quantity=("qty", "sum"),
              revenue=("revenue", "sum"),
          )
    )

    agg["price"] = np.where(
        agg["quantity"] > 0,
        agg["revenue"] / agg["quantity"],
        np.nan,
    )

    print(f"[OZON] После агрегации строк: {len(agg)}")
    return agg


# ---------- CLI-интерфейс для ручного запуска ----------

def main():
    parser = argparse.ArgumentParser(
        description="Обработка новых недельных отчётов WB и Ozon и обновление SALES_ALL_DAILY."
    )
    parser.add_argument(
        "--wb",
        nargs="*",
        help="Пути к отчётам Wildberries (.xlsx)",
    )
    parser.add_argument(
        "--ozon",
        nargs="*",
        help="Пути к отчётам Ozon (.xlsx)",
    )

    args = parser.parse_args()

    wb_files = [Path(p) for p in (args.wb or [])]
    ozon_files = [Path(p) for p in (args.ozon or [])]

    if not wb_files and not ozon_files:
        print("Не указаны файлы отчётов (--wb / --ozon). Нечего обрабатывать.")
        return

    # Инициализируем БД и загружаем список SKU
    init_db()
    relevant_skus = load_relevant_skus(PRODUCTS_PATH)

    parts = []

    for path in wb_files:
        parts.append(process_wb_report(path, relevant_skus=relevant_skus))

    for path in ozon_files:
        parts.append(process_ozon_report(path, relevant_skus=relevant_skus))

    if not parts:
        print("После обработки отчётов не осталось ни одной строки.")
        return

    new_sales = pd.concat(parts, ignore_index=True)
    print(f"Всего новых строк (WB + Ozon): {len(new_sales)}")

    # 2. Записываем в таблицу sales_daily в БД
    insert_sales_daily(new_sales)


if __name__ == "__main__":
    main()
