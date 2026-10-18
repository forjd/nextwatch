# One image for every service: the command decides what it runs (plan 09).
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.12.2 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable
# Bake the embedding model in, so jobs and mood search never download it at runtime.
RUN .venv/bin/python -c "from fastembed import TextEmbedding; \
TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/opt/models')"

FROM python:3.12-slim
# libgomp: OpenMP runtime for LightGBM and implicit.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 1000 app \
 && mkdir /data && chown app /data
COPY --from=build /app/.venv /app/.venv
COPY --from=build --chown=app /opt/models /opt/models
ENV PATH=/app/.venv/bin:$PATH DATA_DIR=/data MODELS_DIR=/opt/models PYTHONUNBUFFERED=1
WORKDIR /data
USER app
EXPOSE 8000
CMD ["nextwatch", "web", "--host", "0.0.0.0"]
