from fastapi import FastAPI

from src.api.schemas import IrisBatchRequest, IrisBatchPrediction, IrisPrediction
from src.api.inference import predict as predict_batch


app = FastAPI(
    title="Iris Classification API",
    version="0.1.0",
    description="Пример MLOps-проекта: классификация ирисов",
)


@app.get("/health")
def health():
    """Простой health-check."""
    return {"status": "ok"}


@app.post("/predict", response_model=IrisBatchPrediction)
def predict(request: IrisBatchRequest):
    """Делает предсказания класса ириса по входным признакам."""
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
