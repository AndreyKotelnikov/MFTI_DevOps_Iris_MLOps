from pathlib import Path

from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity, train_test_validation

from src.data.load_data import get_iris_data

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def run_data_quality_checks() -> Path:
    """Запускает проверки качества данных и сохраняет HTML отчёты."""
    X_train, X_test, y_train, y_test = get_iris_data()

    train_df = X_train.copy()
    train_df["target"] = y_train.values

    test_df = X_test.copy()
    test_df["target"] = y_test.values

    ds_train = Dataset(train_df, label="target")
    ds_test = Dataset(test_df, label="target")

    integrity_suite = data_integrity()
    integrity_result = integrity_suite.run(ds_train)
    integrity_path = REPORTS_DIR / "deepchecks_data_integrity.html"
    integrity_result.save_as_html(
        file = str(integrity_path),
        as_widget = False,
        requirejs = False,
        connected = False
    )
    print(f"Deepchecks data integrity report saved to {integrity_path}")

    tt_suite = train_test_validation()
    tt_result = tt_suite.run(ds_train, ds_test)
    tt_path = REPORTS_DIR / "deepchecks_train_test_validation.html"
    tt_result.save_as_html(
        file = str(tt_path),
        as_widget = False,
        requirejs = False,
        connected = False
    )
    print(f"Deepchecks train/test validation report saved to {tt_path}")

    return tt_path


if __name__ == "__main__":
    run_data_quality_checks()
