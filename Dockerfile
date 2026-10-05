# syntax=docker/dockerfile:1
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.10 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# libgomp: OpenMP runtime used by compiled ML dependencies
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first (cached unless pyproject.toml / uv.lock change)
COPY pyproject.toml uv.lock ./
# The uv cache is a build cache mount, so it never ends up in the image
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Then the project itself (API and Streamlit demo share this image)
COPY README.md config.json ./
COPY src ./src
COPY app ./app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

EXPOSE 8000 8501
# Default: the API. The model is not in the image: mount models/ (see docker-compose.yml).
CMD ["uvicorn", "apex.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
