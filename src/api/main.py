import os

from fastapi import FastAPI
import mlflow

from src.api.schemas import IrisBatchRequest, IrisBatchPrediction, IrisPrediction
from src.api.inference import predict as predict_batch


def configure_mlflow_for_traces() -> None:
    """
    Конфигурация MLflow для трассировки FastAPI.

    1. Берём tracking URI из MLFLOW_TRACKING_URI.
    2. Привязываем активную модель:
       - сначала по MLFLOW_ACTIVE_MODEL_ID (конкретный logged model),
       - если его нет, то по MLFLOW_ACTIVE_MODEL_NAME (по имени).
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    active_model_id = os.getenv("MLFLOW_ACTIVE_MODEL_ID")
    active_model_name = os.getenv("MLFLOW_ACTIVE_MODEL_NAME")

    # Если явно передан model_id из UI — используем его.
    if active_model_id:
        print(f"[MLflow] set_active_model(model_id={active_model_id})")
        mlflow.set_active_model(model_id=active_model_id)
    elif active_model_name:
        # fallback: привязка по имени LoggedModel
        print(f"[MLflow] set_active_model(name={active_model_name})")
        mlflow.set_active_model(name=active_model_name)
    else:
        print(
            "[MLflow] WARNING: ни MLFLOW_ACTIVE_MODEL_ID, ни MLFLOW_ACTIVE_MODEL_NAME "
            "не заданы, трассировки будут без привязки к LoggedModel."
        )


app = FastAPI(
    title="Iris Classification API",
    version="0.1.0",
    description="Пример MLOps-проекта: классификация ирисов",
)


@app.on_event("startup")
def on_startup() -> None:
    """Инициализация MLflow для трассировки при старте FastAPI."""
    configure_mlflow_for_traces()


@app.get("/health")
def health():
    """Простой health-check."""
    return {"status": "ok"}


@app.post("/predict", response_model=IrisBatchPrediction)
@mlflow.trace(
    name="api_predict",
    attributes={
        "component": "fastapi",
        "endpoint": "/predict",
        "service": "iris-api",
    },
)
def predict(request: IrisBatchRequest):
    """
    Делает предсказания класса ириса по входным признакам.

    Благодаря декоратору @mlflow.trace каждый вызов будет логироваться
    как span в MLflow Tracing и (если настроен active_model) привязываться
    к конкретному Logged Model.
    """
    samples = [
        [
            s.sepal_length,
            s.sepal_width,
            s.petal_length,
            s.petal_width,
        ]
        for s in request.samples
    ]
    raw_predictions = predict_batch(samples)
    predictions = [IrisPrediction(**p) for p in raw_predictions]
    return IrisBatchPrediction(predictions=predictions)
