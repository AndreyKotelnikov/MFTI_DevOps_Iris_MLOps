import os
from pathlib import Path
from typing import Dict

import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from src.data.load_data import get_iris_data


def get_model_path() -> Path:
    """Возвращает путь к файлу модели на основе переменных окружения."""
    model_dir = Path(os.getenv("MODEL_DIR", "models"))
    model_dir.mkdir(parents=True, exist_ok=True)
    filename = os.getenv("MODEL_FILENAME", "iris_rf_model.pkl")
    return model_dir / filename


def _configure_mlflow() -> None:
    """Настраивает MLflow: tracking URI и имя эксперимента."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "iris-classification")
    mlflow.set_experiment(experiment_name)


def train_and_save_model() -> Dict[str, float]:
    """Обучает модель, логирует её в MLflow и сохраняет локально."""
    _configure_mlflow()

    X_train, X_test, y_train, y_test = get_iris_data()

    params = {
        "n_estimators": int(os.getenv("RF_N_ESTIMATORS", 100)),
        "max_depth": int(os.getenv("RF_MAX_DEPTH", 5)),
        "random_state": 42,
    }

    model_path = get_model_path()

    with mlflow.start_run(run_name="random_forest_iris"):
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        mlflow.log_params(params)
        mlflow.log_metrics({"accuracy": acc, "f1_weighted": f1})

        registered_name = os.getenv(
            "MLFLOW_REGISTERED_MODEL_NAME",
            "IrisClassifierModel"
        )
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=registered_name,
        )

        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="artifacts")

    metrics = {"accuracy": float(acc), "f1_weighted": float(f1)}
    print(f"Training done. Metrics: {metrics}, model_path={model_path}")
    return {**metrics, "model_path": str(model_path)}


if __name__ == "__main__":
    train_and_save_model()
