# Dockerfile

# ===== Стадия сборки (builder) =====
FROM python:3.12-slim AS builder
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .

# Кешируем колёса для ускорения сборок
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip wheel --no-cache-dir -r requirements.txt -w /wheels

COPY . .

# ===== Стадия рантайма =====
FROM python:3.12-slim AS runtime
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Устанавливаем зависимости из ранее собранных колёс
COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-index --find-links=/wheels -r requirements.txt && \
    useradd -m appuser && \
    chown -R appuser /app

USER appuser

# Копируем исходники
COPY --from=builder /app /app

EXPOSE 8000

# Продовый запуск: Gunicorn + UvicornWorker
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "src.api.main:app", "--bind", "0.0.0.0:8000", "--workers", "2"]
