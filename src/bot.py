#!/usr/bin/env python3
"""
bot.py

Телеграм-бот для:
- загрузки еженедельных отчётов WB/Ozon и записи их в БД (sales_daily) через new_data_processing.py;
- пересборки ML-датасета и переобучения модели;
- запуска прогноза на следующий месяц и выдачи Excel-файла с агрегированным прогнозом;
- просмотра статуса (дата последнего обучения и прогноза).

Команды:
/start, /help
/upload_wb
/upload_ozon
/train_model
/forecast
/status
"""

from __future__ import annotations

import logging
import os
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# Импорты из проекта
from ml_dataset_builder import build_ml_monthly_base
from model_training import train_model
from forecast_next_month import forecast_for_month
from aggregate_forecast import aggregate_forecast

# ---------- Базовые пути ----------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_WB_DIR = RAW_DIR / "wb"
RAW_OZON_DIR = RAW_DIR / "ozon"
MODELS_DIR = BASE_DIR / "models"
FORECASTS_DIR = DATA_DIR / "forecasts"
SRC_DIR = BASE_DIR / "src"

RAW_WB_DIR.mkdir(parents=True, exist_ok=True)
RAW_OZON_DIR.mkdir(parents=True, exist_ok=True)
FORECASTS_DIR.mkdir(parents=True, exist_ok=True)

# После строки 53 (SRC_DIR = BASE_DIR / "src")
SRC_DIR = BASE_DIR / "src"

# Добавляем src/ в Python path для импортов модулей
sys.path.insert(0, str(SRC_DIR))

# ---------- Логирование ----------

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ---------- Вспомогательные функции ----------

def run_new_data_processing(wb_path: Path | None = None, ozon_path: Path | None = None) -> str:
    """
    Запускает new_data_processing.py как отдельный процесс.
    wb_path/ozon_path — пути к файлам отчётов (могут быть None).

    Возвращает stdout скрипта (для отображения пользователю или в логах).
    """
    script_path = SRC_DIR / "new_data_processing.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Не найден скрипт new_data_processing.py по пути: {script_path}")

    cmd = [sys.executable, str(script_path)]
    if wb_path is not None:
        cmd += ["--wb", str(wb_path)]
    if ozon_path is not None:
        cmd += ["--ozon", str(ozon_path)]

    logger.info("Запускаем new_data_processing.py: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("Ошибка при выполнении new_data_processing.py: %s", result.stderr)
        raise RuntimeError(f"new_data_processing.py завершился с ошибкой:\n{result.stderr}")

    logger.info("new_data_processing.py stdout:\n%s", result.stdout)
    return result.stdout


def format_ts(ts: float | None) -> str:
    """Форматирование timestamp в строку, если ts задан, иначе '-'."""
    if ts is None:
        return "-"
    dt = datetime.fromtimestamp(ts)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def find_latest_file(pattern: str) -> Path | None:
    """
    Ищет последний (по времени модификации) файл в FORECASTS_DIR,
    соответствующий маске pattern (например, 'forecast_agg_*.xlsx').
    """
    candidates = list(FORECASTS_DIR.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


# ---------- Хендлеры команд ----------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Приветствие и краткая справка."""
    text = (
        "Здравствуйте! Это бот системы прогнозирования спроса.\n\n"
        "Доступные команды:\n"
        "/upload_wb – загрузить еженедельный отчёт Wildberries (.xlsx)\n"
        "/upload_ozon – загрузить еженедельный отчёт Ozon (.xlsx)\n"
        "/train_model – пересобрать ML-датасет и переобучить модель\n"
        "/forecast – сформировать прогноз на следующий месяц и получить Excel\n"
        "/status – посмотреть дату последнего обучения и прогноза\n"
    )
    await update.message.reply_text(text)


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Дублируем информацию /start."""
    await start(update, context)


async def upload_wb_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Начало сценария загрузки отчёта WB.
    После этой команды пользователь должен отправить Excel-файл.
    """
    context.user_data["expect_upload"] = "wb"
    await update.message.reply_text(
        "Пожалуйста, отправьте Excel-файл отчёта Wildberries за нужную неделю (формат .xlsx)."
    )


async def upload_ozon_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Начало сценария загрузки отчёта Ozon.
    После этой команды пользователь должен отправить Excel-файл.
    """
    context.user_data["expect_upload"] = "ozon"
    await update.message.reply_text(
        "Пожалуйста, отправьте Excel-файл отчёта Ozon за нужную неделю (формат .xlsx)."
    )


async def train_model_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Пересборка ml_monthly_base и переобучение модели.
    """
    await update.message.reply_text(
        "Запускаю пересборку ML-датасета и обучение модели. "
        "Процесс может занять некоторое время…"
    )
    try:
        # 1) Пересобрать ml_monthly_base на основе БД
        build_ml_monthly_base()
        # 2) Обучить лог-модель CatBoost и сохранить её + калибровку
        train_model()
    except Exception as e:
        logger.exception("Ошибка в /train_model")
        await update.message.reply_text(f"Произошла ошибка при обучении модели: {e}")
        return

    await update.message.reply_text("Обновление датасета и обучение модели успешно завершены.")


async def forecast_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Запуск прогноза на следующий месяц и выдача агрегированного Excel-файла.
    """
    await update.message.reply_text(
        "Запускаю формирование прогноза на следующий месяц. "
        "Пожалуйста, подождите…"
    )

    try:
        # 1) Сформировать прогноз по всем SKU/маркетплейсам
        # (forecast_for_month сам возьмёт последний месяц из ML-датасета, если параметр не задан)
        forecast_for_month()

        # 2) Агрегировать прогноз по площадкам (до уровня SKU)
        agg_path = aggregate_forecast()

    except Exception as e:
        logger.exception("Ошибка в /forecast")
        await update.message.reply_text(f"Произошла ошибка при формировании прогноза: {e}")
        return

    # 3) Отправляем файл пользователю
    if not agg_path.exists():
        await update.message.reply_text(
            "Прогноз был выполнен, но агрегированный файл не найден."
        )
        return

    await update.message.reply_text("Прогноз успешно сформирован, отправляю файл.")
    with agg_path.open("rb") as f:
        await context.bot.send_document(
            chat_id=update.effective_chat.id,
            document=f,
            filename=agg_path.name,
            caption="Агрегированный прогноз по SKU (сумма по маркетплейсам).",
        )

def fmt_metric(value) -> str:
    """Красиво форматирует число метрики или возвращает '-'."""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "-"


async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Статус:
    - когда в последний раз обучалась модель (из model_meta.json, если есть),
    - последний агрегированный прогноз,
    - ключевые метрики (валидация + тест после калибровки, если сохранены).
    """
    # 1. Читаем model_meta.json
    meta_file = MODELS_DIR / "model_meta.json"
    last_train_str = "-"
    val_metrics = {}
    test_full_calib_metrics = {}

    if meta_file.exists():
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)

            last_train_str = meta.get("last_train_time", "-")

            metrics = meta.get("metrics", {})
            val_metrics = metrics.get("val", {}) or {}
            test_full_calib_metrics = metrics.get("test_full_calib", {}) or {}
        except Exception:
            last_train_str = "ошибка чтения model_meta.json"
    else:
        last_train_str = "model_meta.json отсутствует"

    # 2. Последний агрегированный прогноз
    latest_agg = find_latest_file("forecast_agg_*.xlsx")
    if latest_agg is not None:
        last_forecast_ts = latest_agg.stat().st_mtime
        month_info = latest_agg.stem.replace("forecast_agg_", "")
        last_forecast_str = format_ts(last_forecast_ts)
    else:
        last_forecast_str = "-"
        month_info = "-"

    # 3. Формируем текст с метриками
    text_lines = [
        "Статус системы:",
        "",
        f"Последнее обучение модели: {last_train_str}",
        f"Файл модели: {MODELS_DIR / 'catboost_demand_log.cbm'}",
        "",
        f"Последний агрегированный прогноз: {last_forecast_str}",
        f"Месяц прогноза (из имени файла): {month_info}",
        "",
        "Метрики качества (если сохранены в model_meta.json):",
        "",
        "Валидация (Log-CatBoost, в штуках):",
        f"  RMSE = {fmt_metric(val_metrics.get('rmse'))}",
        f"  MAE  = {fmt_metric(val_metrics.get('mae'))}",
        f"  MAPE = {fmt_metric(val_metrics.get('mape'))}",
        f"  WAPE = {fmt_metric(val_metrics.get('wape'))}",
        f"  Bias = {fmt_metric(val_metrics.get('bias'))}",
        "",
        "Тест, полный период (после калибровки):",
        f"  RMSE = {fmt_metric(test_full_calib_metrics.get('rmse'))}",
        f"  MAE  = {fmt_metric(test_full_calib_metrics.get('mae'))}",
        f"  MAPE = {fmt_metric(test_full_calib_metrics.get('mape'))}",
        f"  WAPE = {fmt_metric(test_full_calib_metrics.get('wape'))}",
        f"  Bias = {fmt_metric(test_full_calib_metrics.get('bias'))}",
    ]

    await update.message.reply_text("\n".join(text_lines))



# ---------- Обработка загруженных файлов ----------

async def document_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Обработка входящих файлов (документов).
    Ожидаем, что перед этим пользователь вызвал /upload_wb или /upload_ozon,
    и в context.user_data['expect_upload'] записан тип ('wb' или 'ozon').
    """
    message = update.message
    if not message or not message.document:
        return

    expect = context.user_data.get("expect_upload")
    if expect not in ("wb", "ozon"):
        await message.reply_text(
            "Неясно, к какому отчёту отнести файл. "
            "Сначала вызовите /upload_wb или /upload_ozon, затем отправьте файл."
        )
        return

    document = message.document
    file_name = document.file_name or "report.xlsx"

    # Простая проверка расширения
    if not file_name.lower().endswith(".xlsx"):
        await message.reply_text("Пожалуйста, отправьте файл в формате .xlsx.")
        return

    # Определяем папку назначения
    if expect == "wb":
        dest_dir = RAW_WB_DIR
    else:
        dest_dir = RAW_OZON_DIR

    dest_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_path = dest_dir / f"{ts}_{file_name}"

    # Скачиваем файл
    file_obj = await document.get_file()
    await file_obj.download_to_drive(str(dest_path))

    logger.info("Получен файл %s, сохранён как %s", file_name, dest_path)
    await message.reply_text(f"Файл '{file_name}' сохранён как '{dest_path.name}'.")

    # Сбрасываем ожидание, чтобы не путаться
    context.user_data["expect_upload"] = None

    # Запускаем обработку данных через new_data_processing.py
    await message.reply_text("Запускаю обработку отчёта и добавление данных в БД…")

    try:
        if expect == "wb":
            stdout_text = run_new_data_processing(wb_path=dest_path, ozon_path=None)
        else:
            stdout_text = run_new_data_processing(wb_path=None, ozon_path=dest_path)
    except Exception as e:
        logger.exception("Ошибка при обработке нового отчёта")
        await message.reply_text(f"Произошла ошибка при обработке отчёта: {e}")
        return

    # Можно отправить пользователю только часть лога, чтобы не заспамить чат
    short_log = "\n".join(stdout_text.strip().splitlines()[-10:])
    await message.reply_text(
        "Обработка отчёта завершена.\n"
        "Фрагмент служебного лога:\n"
        f"{short_log}"
    )


# ---------- main() ----------

def main() -> None:
    """
    Точка входа: создаём Application и запускаем long polling.
    Токен берём из переменной окружения TELEGRAM_BOT_TOKEN.
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError(
            "Не задан TELEGRAM_BOT_TOKEN. "
            "Установите переменную окружения TELEGRAM_BOT_TOKEN с токеном бота."
        )

    application = Application.builder().token(token).build()

    # Команды
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_cmd))
    application.add_handler(CommandHandler("upload_wb", upload_wb_cmd))
    application.add_handler(CommandHandler("upload_ozon", upload_ozon_cmd))
    application.add_handler(CommandHandler("train_model", train_model_cmd))
    application.add_handler(CommandHandler("forecast", forecast_cmd))
    application.add_handler(CommandHandler("status", status_cmd))

    # Обработка документов (Excel-файлов)
    application.add_handler(MessageHandler(filters.Document.ALL, document_handler))

    logger.info("Бот запущен. Ожидаем сообщения…")
    application.run_polling(allowed_updates=None)


if __name__ == "__main__":
    main()
