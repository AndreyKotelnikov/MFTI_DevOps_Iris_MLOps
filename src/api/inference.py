import os
from pathlib import Path
from typing import List, Dict

import joblib
from sklearn.base import ClassifierMixin

from src.models.train import train_and_save_model

CLASS_NAMES = ["setosa", "versicolor", "virginica"]


def get_model_path() -> Path:
    model_dir = Path(os.getenv("MODEL_DIR", "models"))
    model_dir.mkdir(parents=True, exist_ok=True)
    filename = os.getenv("MODEL_FILENAME", "iris_rf_model.pkl")
    return model_dir / filename


_model: ClassifierMixin | None = None


def load_model() -> ClassifierMixin:
    """Лениво загружает модель, при отсутствии обучает её."""
    global _model
    if _model is not None:
        return _model

    model_path = get_model_path()
    if not model_path.exists():
        print("Model file not found, training a new model...")
        train_and_save_model()

    _model = joblib.load(model_path)
    return _model


def predict(samples: List[List[float]]) -> List[Dict]:
    """Делает предсказание для списка векторов признаков."""
    model = load_model()
    class_ids = model.predict(samples)
    probas = model.predict_proba(samples)

    results = []
    for idx, class_id in enumerate(class_ids):
        best_prob = float(max(probas[idx]))
        results.append(
            {
                "class_id": int(class_id),
                "class_name": CLASS_NAMES[int(class_id)],
                "probability": best_prob,
            }
        )
    return results
