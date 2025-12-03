# Iris MLOps Project

MLOps-проект: обучение модели классификации ирисов, REST API для инференса, трекинг экспериментов в MLflow, отчёты Deepchecks и Evidently, плюс фиксация архитектурных решений через ADR.

## Технологический стек

- **Язык / ML**
  - Python 3.12
  - scikit-learn (RandomForestClassifier)
  - pandas, numpy, scipy

- **API**
  - FastAPI
  - Gunicorn + UvicornWorker

- **MLOps**
  - MLflow (tracking + модельный реестр)
  - MinIO (локальное S3-совместимое хранилище артефактов)
  - Deepchecks (качество данных)
  - Evidently (дрейф и сводка по данным)

- **Инфраструктура**
  - Docker, Docker Compose
  - GitLab CI (build + test)
  - Yandex Cloud Serverless Containers (деплой standalone API)
  - adr-tools (Architecture Decision Records)

---

## Структура проекта

```text
.
├── .adr-dir                  # Конфигурация adr-tools (указывает на docs/adr)
├── .gitlab-ci.yml            # CI: docker build + pytest
├── Dockerfile                # Dev/compose образ (API + обучение + мониторинг)
├── Dockerfile.mlflow         # Образ MLflow + boto3
├── docker-compose.yml        # Стенд: app + mlflow + minio
├── deploy/
│   ├── Dockerfile.api        # Облегчённый образ только с API и моделью
│   └── deploy_to_yc.ps1      # Скрипт деплоя в Yandex Cloud
├── docs/
│   ├── adr/                  # ADR (Architecture Decision Records)
│   │   ├── 0001-record-architecture-decisions.md
│   │   ├── 0002-iris-api.md
│   │   ├── 0003-fastapi-gunicorn.md
│   │   ├── 0004-mlflow-minio.md
│   │   ├── 0005-deepchecks-evidently.md
│   │   └── 0006-iris-api-serverless-yandex-cloud.md
│   └── locust/               # (зарезервировано под нагрузочное тестирование)
├── models/                   # Локальные файлы модели (pickle)
├── reports/                  # HTML-отчёты Deepchecks/Evidently
├── scripts/
│   └── all.ps1               # Полный пайплайн: train + Deepchecks + Evidently
├── src/
│   ├── api/                  # FastAPI-приложение
│   │   ├── main.py           # эндпоинты /health и /predict + MLflow trace
│   │   ├── inference.py      # ленивое чтение/обучение модели
│   │   └── schemas.py        # Pydantic-схемы запросов/ответов
│   ├── data/
│   │   └── load_data.py      # загрузка iris + train/test split
│   ├── models/
│   │   └── train.py          # обучение, логирование в MLflow, сохранение модели
│   └── monitoring/
│       ├── data_quality_deepchecks.py
│       └── data_drift_evidently.py
├── tests/
│   ├── test_model.py         # проверка метрик и наличия файла модели
│   └── test_api.py           # проверка /predict
└── tools/
    └── adr-tools/            # исходники adr-tools (git subdir)
````

---

## Быстрый старт (Docker Compose)

### 1. Требования

* Docker + Docker Compose
* Порт `8000` (API), `5000` (MLflow), `9000/9001` (MinIO) свободны

### 2. Создать файл `.env`

Пример минимальной конфигурации:

```env
# Доступ в MinIO
MINIO_ROOT_USER=mlflow
MINIO_ROOT_PASSWORD=mlflowsecret

# Имя бакета для артефактов MLflow
MLFLOW_S3_BUCKET=mlflow-iris

# (опционально) настройки обучения и MLflow
MLFLOW_EXPERIMENT_NAME=iris-classification
MLFLOW_REGISTERED_MODEL_NAME=IrisClassifierModel
RF_N_ESTIMATORS=100
RF_MAX_DEPTH=5
```

### 3. Поднять стенд

Из корня репозитория:

```bash
docker compose up -d
```

Поднимутся сервисы:

* `app` — FastAPI API (`http://localhost:8000`)
* `mlflow` — MLflow UI (`http://localhost:5000`)
* `minio` — MinIO (`http://localhost:9000`, консоль `http://localhost:9001`)

---

## Обучение модели и генерация отчётов

Есть удобный PowerShell-скрипт, который выполняет весь пайплайн внутри docker-контейнера `app`.

Из корня проекта (Windows PowerShell):

```powershell
.\scripts\all.ps1
```

Скрипт делает:

1. `docker compose run --rm app python -m src.models.train`

   * обучает RandomForestClassifier на iris
   * логирует параметры и метрики в MLflow
   * сохраняет модель в `models/iris_rf_model.pkl`

2. `docker compose run --rm app python -m src.monitoring.data_quality_deepchecks`

   * создаёт отчёт качества данных:

     * `reports/deepchecks_data_integrity.html`
     * `reports/deepchecks_train_test_validation.html`

3. `docker compose run --rm app python -m src.monitoring.data_drift_evidently`

   * создаёт отчёты Evidently:

     * `reports/evidently_data_drift.html`
     * `reports/evidently_data_drift.json`

После выполнения:

* Модель лежит в `models/`
* Отчёты — в `reports/`
* В MLflow UI (`http://localhost:5000`) видно эксперимент `iris-classification` и зарегистрированную модель.

---

## REST API

### Эндпоинты

* `GET /health`
  Простой health-check:

  ```json
  { "status": "ok" }
  ```

* `POST /predict`
  Делает предсказание класса ириса по списку измерений.
  Схемы описаны в `src/api/schemas.py`.

Пример запроса:

```bash
curl -X POST "http://localhost:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"samples\":[{\"sepal_length\":5.1,\"sepal_width\":3.5,\"petal_length\":1.4,\"petal_width\":0.2}]}"
```

Пример ответа:

```json
{
  "predictions": [
    {
      "class_id": 0,
      "class_name": "setosa",
      "probability": 0.98
    }
  ]
}
```

При каждом вызове `/predict`:

* модель лениво загружается (или обучается с нуля, если файла нет);
* вызов трассируется в MLflow через декоратор `@mlflow.trace`.

---

## Локальный запуск без Docker (опционально)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# Обучить модель
python -m src.models.train

# Запустить API (dev-режим)
uvicorn src.api.main:app --reload --port 8000
```

---

## Тесты

Тесты проверяют:

* качество пайплайна обучения (`tests/test_model.py`);
* корректность работы `/predict` (`tests/test_api.py`).

Локальный запуск:

```bash
pytest -v
```

Или как в GitLab CI:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest -v
```

---

## CI (GitLab)

Файл `.gitlab-ci.yml` содержит два джоба:

* `build`

  * запускает `docker compose build` (сборка образов)
* `test`

  * устанавливает зависимости
  * запускает `pytest -v`

Конфигурация рассчитана на использование `docker:dind` для сборки образов.

---

## ADR: фиксация архитектурных решений

Для документирования архитектуры используются **Architecture Decision Records** (ADR) и библиотека **adr-tools**.

### Где лежат решения

* Конфигурация adr-tools: `.adr-dir` (указывает на `docs/adr`)
* Текущие ADR:

  * `0001-record-architecture-decisions.md` — введение ADR
  * `0002-iris-api.md` — архитектура проекта
  * `0003-fastapi-gunicorn.md` — выбор FastAPI + Gunicorn/Uvicorn
  * `0004-mlflow-minio.md` — выбор MLflow + MinIO
  * `0005-deepchecks-evidently.md` — выбор Deepchecks + Evidently
  * `0006-iris-api-serverless-yandex-cloud.md` — деплой в Yandex Cloud serverless

### Использование adr-tools

Исходники adr-tools находятся в `tools/adr-tools`. Для удобного запуска:

1. Установить Git Bash (если ещё не установлен).

2. Добавить в `~/.bashrc` (путь подправить под свой):

   ```bash
   export PATH="$PATH:/e/DS/3_семестр/DevOps/ДЗ/iris-mlops-project/tools/adr-tools/src"
   ```

3. В терминале (Git Bash или терминал PyCharm с Shell path = Git Bash):

   ```bash
   cd /e/DS/3_семестр/DevOps/ДЗ/iris-mlops-project

   # Инициализация (уже сделано в этом репо, но на всякий случай)
   adr init docs/adr

   # Создать новый ADR
   adr new "Краткое описание решения"
   ```

Новый Markdown-файл появится в `docs/adr`. Дальше его заполняют по шаблону (Context / Decision / Consequences).

---

## Деплой iris-api в Yandex Cloud (serverless)

Для продакшн-/демо-деплоя используется отдельный образ и PowerShell-скрипт.

### 1. Собрать standalone-образ локально

```bash
docker build -f deploy/Dockerfile.api -t iris-api-standalone:0.0.1 .
docker run --rm -p 8000:8000 iris-api-standalone:0.0.1
```

### 2. Деплой через `deploy_to_yc.ps1`

Требуется:

* установленный `yc` (Yandex Cloud CLI);
* настроенные:

  * `YC_SA_ID` (ID сервисного аккаунта),
  * `YC_FOLDER_ID` (ID каталога),
  * `RegistryId` — ID контейнерного реестра (по умолчанию уже указан).

Пример вызова:

```powershell
cd .\deploy
.\deploy_to_yc.ps1 `
  -ImageName iris-api `
  -ImageTag 0.0.1 `
  -RegistryId crpdc99upo405hm8s93v `
  -ContainerName iris-api `
  -ServiceAccountId $env:YC_SA_ID `
  -FolderId $env:YC_FOLDER_ID
```

Скрипт:

1. собирает образ по `deploy/Dockerfile.api`;
2. пушит его в Yandex Container Registry;
3. создаёт/обновляет serverless-контейнер;
4. включает вызов без авторизации.

