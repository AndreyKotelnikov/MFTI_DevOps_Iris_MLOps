from pathlib import Path

from src.models.train import train_and_save_model


def test_training_pipeline_produces_reasonable_metrics(tmp_path, monkeypatch):
    """Проверяем, что обучение даёт адекватные метрики и файл модели."""
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    metrics = train_and_save_model()

    assert metrics["accuracy"] > 0.8
    assert metrics["f1_weighted"] > 0.8

    model_path = Path(metrics["model_path"])
    assert model_path.exists()
