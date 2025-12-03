# 3. Почему используем FastAPI и Gunicorn

Date: 2025-12-03

## Status

Accepted

## Context

Необходимо поднять HTTP API для инференса модели классификации ирисов:

- минимум один endpoint `/predict` с валидацией входа и схемой ответа;
- endpoint `/health` для health-check;
- возможность запускать сервис как локально, так и в Docker / serverless-контейнере.

Рассматривались варианты:
- Flask + ручная валидация входа;
- Django REST Framework;
- FastAPI с Pydantic-схемами и ASGI-стеком.

## Decision

Используем **FastAPI** как web-фреймворк и **Gunicorn с UvicornWorker** как процесс-менеджер в Docker-образах.

### Обоснование

1. **Типизация и валидация.**  
   Pydantic-схемы в `src/api/schemas.py` дают строгую валидацию и автогенерацию OpenAPI/Swagger.

2. **Производительность и асинхронность.**  
   FastAPI работает поверх ASGI и хорошо масштабируется, а UvicornWorker даёт нам production-ready запуск.

3. **Простая интеграция с MLflow.**  
   Декоратор `@mlflow.trace` удобно вешается на обработчик `POST /predict`.

4. **Простота контейнеризации.**  
   В `Dockerfile` используется стандартная команда:
   `gunicorn -k uvicorn.workers.UvicornWorker src.api.main:app ...`.

## Consequences

- Команда должна понимать основы асинхронного стека FastAPI/ASGI.
- В случае более тяжёлых моделей может понадобиться настройка количества воркеров и таймаутов Gunicorn.
- При миграции на другой фреймворк (например, gRPC) потребуется новый ADR и изменение API-слоя.
