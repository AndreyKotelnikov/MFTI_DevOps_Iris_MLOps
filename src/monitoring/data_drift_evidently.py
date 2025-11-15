from pathlib import Path

from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.report import Report

from src.data.load_data import get_iris_data

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def run_data_drift_report() -> Path:
    """Строит отчёт Evidently по качеству данных и дрейфу."""
    X_train, X_test, y_train, y_test = get_iris_data()

    ref = X_train.copy()
    ref["target"] = y_train.values

    cur = X_test.copy()
    cur["target"] = y_test.values

    report = Report(
        metrics=[
            DataQualityPreset(),
            DataDriftPreset(),
        ]
    )

    report.run(reference_data=ref, current_data=cur)

    output_path = REPORTS_DIR / "evidently_data_drift_report.html"
    report.save_html(str(output_path))
    print(f"Evidently data drift report saved to {output_path}")
    return output_path


if __name__ == "__main__":
    run_data_drift_report()
