FROM python:3.12-slim

ENV PYTHONPATH="/app"
ENV PATH="/root/.local/bin:$PATH"

RUN apt-get update && apt-get install -y curl \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false && poetry install --no-root

COPY . /app

COPY .env /app/.env

CMD ["python", "main.py"]
