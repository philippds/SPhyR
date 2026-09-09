# The environment serves tasks built by the sphyr library and scores them with
# its physics engine, so the image is built from the repository root:
#
#   docker build -f envs/sphyr_env/server/Dockerfile -t sphyr-env:latest .
#
FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.3 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first, so a code change does not re-resolve the environment.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# The bundled dataset files travel with the library, so a container serves
# episodes without reaching for the hub.
COPY src ./src
COPY envs ./envs
RUN uv sync --frozen --no-dev --no-editable

ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app/envs/sphyr_env

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
