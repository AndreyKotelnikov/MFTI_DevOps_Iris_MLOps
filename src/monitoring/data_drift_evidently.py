from pathlib import Path

import pandas as pd
from sklearn import datasets

from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "iris_processed.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_iris_dataframe() -> pd.DataFrame:
    """Загрузить данные iris либо из сохранённого CSV, либо из sklearn."""
    if DATA_PATH.exists():
        print(f"[Evidently] Использую подготовленный датасет: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
        return df

    print("[Evidently] Файл iris_processed.csv не найден, загружаю iris из sklearn")
    iris = datasets.load_iris(as_frame=True)
    df = iris.frame.copy()
    # В свежем sklearn target уже есть в frame, но на всякий случай
    if "target" not in df.columns:
        df["target"] = iris.target
    return df


def split_reference_current(df: pd.DataFrame, reference_frac: float = 0.7, random_state: int = 42):
    """Разделить на reference и current, чтобы эмулировать временной сдвиг."""
    df_shuffled = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    split_idx = int(len(df_shuffled) * reference_frac)
    reference = df_shuffled.iloc[:split_idx].copy()
    current = df_shuffled.iloc[split_idx:].copy()
    return reference, current


def run_evidently_reports():
    df = load_iris_dataframe()
    reference, current = split_reference_current(df)

    # Конфиг репорта: дрейф + сводка по качеству данных
    report = Report(
        metrics=[
            DataDriftPreset(),    # общий дрейф фич
            DataSummaryPreset(),  # сводные метрики качества данных
        ],
        include_tests=True,       # автоматически сгенерированные тесты
    )

    # В Evidently 0.7.x run возвращает объект результата
    result = report.run(
        reference_data=reference,
        current_data=current,
    )

    html_path = REPORTS_DIR / "evidently_data_drift.html"
    json_path = REPORTS_DIR / "evidently_data_drift.json"

    # save_html / json вызываются у result
    result.save_html(str(html_path))
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(result.json())

    print(f"[Evidently] HTML отчёт сохранён в: {html_path}")
    print(f"[Evidently] JSON отчёт сохранён в: {json_path}")


if __name__ == "__main__":
    run_evidently_reports()
