FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN groupadd --system bot && useradd --system --gid bot --home-dir /app bot

COPY . /app
RUN pip install --no-cache-dir . && \
    mkdir -p /app/data && \
    chown -R bot:bot /app

USER bot

CMD ["python", "-m", "bot"]
