# Iris MLOps Project

Полноценный пример MLOps-проекта для задачи **классификации ирисов** с использованием:

- **Python / scikit-learn** — модель RandomForestClassifier
- **MLflow** — логирование экспериментов, метрик и артефактов
- **MinIO (S3)** — хранение артефактов модели (через MLflow)
- **Deepchecks** — проверка качества данных и валидация модели
- **EvidentlyAI** — анализ дрейфа данных
- **FastAPI + Gunicorn + UvicornWorker** — REST API для инференса
- **Docker / docker-compose** — контейнеризация и локальная оркестрация
- **GitHub Actions / GitLab CI** — CI-пайплайны
- **Git LFS** — хранение больших файлов моделей

## 1. Предварительные требования

- Python 3.12+ (локально можно и 3.13, если библиотеки поддерживают)
- Docker + Docker Compose
- Git + Git LFS
- Аккаунты GitHub и GitLab (для CI/CD)

## 2. Установка и запуск локально (без Docker)

```bash
git clone <ваш-репозиторий> iris-mlops
cd iris-mlops

# создайте и активируйте виртуальное окружение (пример для conda)
conda create -n iris-mlops python=3.12 -y
conda activate iris-mlops

pip install -r requirements.txt
cp .env.example .env
```


## 1. Структура проекта

```text
iris-mlops/
├─ README.md
├─ requirements.txt
├─ .env.example
├─ .gitignore
├─ .gitattributes
├─ docker-compose.yml
├─ Dockerfile
├─ data/
│  └─ .gitkeep
├─ models/
│  └─ .gitkeep          # большие файлы тут → Git LFS
├─ reports/
│  └─ .gitkeep          # HTML отчёты Deepchecks / Evidently
├─ src/
│  ├─ __init__.py
│  ├─ data/
│  │  ├─ __init__.py
│  │  └─ load_data.py
│  ├─ models/
│  │  ├─ __init__.py
│  │  └─ train.py
│  ├─ monitoring/
│  │  ├─ __init__.py
│  │  ├─ data_quality_deepchecks.py
│  │  └─ data_drift_evidently.py
│  └─ api/
│     ├─ __init__.py
│     ├─ schemas.py
│     ├─ inference.py
│     └─ main.py
├─ tests/
│  ├─ __init__.py
│  ├─ test_model.py
│  └─ test_api.py
├─ .github/
│  └─ workflows/
│     └─ ci.yml
└─ .gitlab-ci.yml
```

---

> После запуска `docker compose up -d`:
>
> * MinIO UI: [http://localhost:9001](http://localhost:9001) (создать bucket `mlflow`).
> * MLflow UI: [http://localhost:5000](http://localhost:5000)
> * API: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 2. Локальный запуск без Docker

### 2.1 Обучение модели и логирование в MLflow

```bash
python -m src.models.train
```

Скрипт:

* загружает датасет Iris;
* делит на train/test;
* обучает RandomForestClassifier;
* логирует параметры и метрики в MLflow;
* сохраняет модель в директорию `models/` и логирует артефакт.

### 2.2 Проверка данных Deepchecks

```bash
python -m src.monitoring.data_quality_deepchecks
```

Скрипт:

* формирует `Dataset` для train и test;
* запускает `data_integrity` и `train_test_validation` suites;
* сохраняет HTML отчёты в `reports/deepchecks_*.html`.

Откройте HTML-файлы в браузере и проанализируйте качество данных и корректность разбиения train/test.

### 2.3 Анализ дрейфа с Evidently

```bash
python -m src.monitoring.data_drift_evidently
```

Скрипт:

* сравнивает распределения признаков и таргета между train и test;
* строит HTML отчёт `reports/evidently_data_drift_report.html`.

В отчёте видно, есть ли признаки с дрейфом и насколько значительны изменения.

### 2.4 Локальный запуск API

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

API эндпоинты:

* `GET /health` — health-check
* `POST /predict` — предсказание класса ириса

Пример запроса:

```json
{
  "samples": [
    {
      "sepal_length": 5.1,
      "sepal_width": 3.5,
      "petal_length": 1.4,
      "petal_width": 0.2
    }
  ]
}
```

Документация OpenAPI доступна по адресу `http://localhost:8000/docs`.

## 3. Запуск в Docker / Docker Compose

### 3.1 Подготовка

```bash
cp .env.example .env
# при необходимости поменяйте пароли MinIO и имя бакета для MLflow
```

Убедитесь, что Docker Desktop запущен.

### 3.2 Старт всех сервисов

```bash
docker compose up -d --build
```

Запустятся:

* `app` — FastAPI API (порт 8000)
* `mlflow` — MLflow Tracking Server (порт 5000)
* `minio` — S3-совместимое хранилище (9000 — API, 9001 — web-console)

Зайдите в MinIO UI: `http://localhost:9001` и создайте bucket `mlflow`
(или тот, что указан в `MLFLOW_S3_BUCKET`).

После этого:

* MLflow UI: `http://localhost:5000`
* API: `http://localhost:8000/docs`

### 3.3 Предсказание через API

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
        "samples": [
          {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
          }
        ]
      }'
```

Ответ содержит `class_id`, `class_name` и `probability`.

## 4. CI/CD

### 4.1 GitHub Actions

Файл: `.github/workflows/ci.yml`.

Пайплайн:

* устанавливает Python и зависимости;
* запускает `pytest`;
* гарантирует, что проект собирается "с нуля" и тесты проходят.

При каждом `push` или `pull_request` в ветки `main/master` CI запускается автоматически.

### 4.2 GitLab CI

Файл: `.gitlab-ci.yml`.

Пайплайн:

* использует образ `python:3.12-slim`;
* устанавливает зависимости;
* запускает `pytest`.

CI гарантирует воспроизводимость окружения и корректность кода в GitLab.

## 5. Git LFS для моделей

В файле `.gitattributes` настроено отслеживание `models/**` через Git LFS.

Шаги один раз на машине:

```bash
git lfs install
git lfs track "models/*"
git add .gitattributes
git commit -m "Enable Git LFS for models"
```

Так большие файлы моделей не будут раздувать историю репозитория.

## 6. Тесты

### 6.1 Тесты модели

`tests/test_model.py`:

* вызывает `train_and_save_model()`;
* проверяет, что accuracy и F1 выше 0.8;
* проверяет наличие файла модели.

### 6.2 Тесты API

`tests/test_api.py`:

* обучает модель во временной директории;
* инициализирует FastAPI-приложение;
* отправляет запрос к `/predict`;
* валидирует структуру и значения ответа.

Запуск тестов локально:

```bash
pytest -v
```

В CI эти же команды выполняются автоматически.

## 7 Запуск обучения модели и отчётов в Docker

Запуск обучения модели и логирования в MLflow:
```bash
docker compose run --rm app python -m src.models.train
```

Запуск проверки качества данных Deepchecks:
```bash
docker compose run --rm app python -m src.monitoring.data_quality_deepchecks
```

Запуск анализа дрейфа данных Evidently:
```bash
docker compose run --rm app python -m src.monitoring.data_drift_evidently
```

