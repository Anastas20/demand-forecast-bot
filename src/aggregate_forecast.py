"""
aggregate_forecast.py

Агрегация прогноза по всем маркетплейсам:
из файла вида forecast_YYYY_MM.xlsx строится
таблица с суммарным прогнозом по каждому SKU
(объединяя Ozon и WB).

По умолчанию берётся «последний» forecast_*.xlsx
по времени модификации в папке data/forecasts.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

# Базовые директории
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FORECASTS_DIR = DATA_DIR / "forecasts"


def find_latest_forecast() -> Path:
    """
    Ищет последний файл forecast_*.xlsx в папке forecasts
    по времени модификации.
    """
    if not FORECASTS_DIR.exists():
        raise FileNotFoundError(f"Папка с прогнозами не найдена: {FORECASTS_DIR}")

    candidates = list(FORECASTS_DIR.glob("forecast_*.xlsx"))
    if not candidates:
        raise FileNotFoundError(f"В папке {FORECASTS_DIR} нет файлов forecast_*.xlsx")

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest


def aggregate_forecast(forecast_path: Optional[str | Path] = None) -> Path:
    """
    Агрегирует прогноз по всем маркетплейсам для одного месяца.

    Если forecast_path не задан, берётся последний forecast_*.xlsx из data/forecasts.

    Возвращает путь к агрегированному файлу.
    """
    # 1. Определяем входной файл
    if forecast_path is None:
        forecast_file = find_latest_forecast()
    else:
        forecast_file = Path(forecast_path)

    if not forecast_file.exists():
        raise FileNotFoundError(f"Не найден файл прогноза: {forecast_file}")

    print(f"Читаем прогноз из файла: {forecast_file}")
    df = pd.read_excel(forecast_file)

    # 2. Проверяем необходимые столбцы
    required_cols = [
        "month",
        "seller_sku",
        "category",
        "qty_month",
        "y_pred",
        "y_pred_corr",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"В файле нет необходимых столбцов: {missing}")

    # 3. Группировка: суммируем по SKU и категории, игнорируя marketplace
    group_cols = ["month", "seller_sku", "category"]

    agg_dict = {
        "qty_month": "sum",
        "y_pred": "sum",
        "y_pred_corr": "sum",
    }

    # k_category одинаковый внутри категории, берём первый (если есть)
    if "k_category" in df.columns:
        agg_dict["k_category"] = "first"

    df_agg = (
        df.groupby(group_cols, as_index=False)
          .agg(agg_dict)
    )

    # 4. Формируем имя выходного файла
    # Предполагаем, что это прогноз за один месяц
    if df_agg["month"].nunique() == 1:
        month_val = pd.to_datetime(df_agg["month"].iloc[0], errors="coerce")
        suffix = month_val.strftime("%Y_%m")
    else:
        # на всякий случай, если в файле несколько месяцев
        suffix = "multi"

    out_name = f"forecast_agg_{suffix}.xlsx"
    out_path = FORECASTS_DIR / out_name

    # 5. Сохраняем агрегированный прогноз
    df_agg.to_excel(out_path, index=False)
    print(f"Агрегированный прогноз сохранён в: {out_path}")

    return out_path


if __name__ == "__main__":
    # вариант 1: агрегировать последний forecast_*.xlsx
    aggregate_forecast()

    # вариант 2: можно явно задать файл:
    # aggregate_forecast(FORECASTS_DIR / "forecast_2025_10.xlsx")
