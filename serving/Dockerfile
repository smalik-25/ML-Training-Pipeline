# Serving image for the FastAPI price-premium API (App Runner target).
#
# Distinct from the root Dockerfile, which is the batch stage-runner. This image
# runs the online endpoint: uvicorn serving the model loaded from MODEL_URI (an
# s3:// path to model.pt) at startup, on port 8080 (App Runner's default).
#
# We install CPU-only torch from the PyTorch CPU wheel index, which keeps the
# image around ~1GB instead of the ~5GB the default CUDA build would add. v1
# serves via MODEL_URI, so MLflow is intentionally not installed here; the
# registry-driven variant would add it.
#
# Build + run locally:
#   docker build -f serving/Dockerfile -t sneaker-serving .
#   docker run --rm -p 8080:8080 \
#     -e MODEL_URI=s3://my-bucket/models/2026-06-28/<run_id>/model.pt \
#     -e AWS_ACCESS_KEY_ID=... -e AWS_SECRET_ACCESS_KEY=... -e AWS_DEFAULT_REGION=us-east-1 \
#     sneaker-serving

# ---- builder ----
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CPU-only torch first, so installing the project doesn't pull the CUDA build.
RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2"

COPY pyproject.toml README.md ./
COPY stages/ ./stages/
COPY models/ ./models/
COPY dags/ ./dags/
COPY serving/ ./serving/
# Installs the package + core deps (pandas, pyarrow, s3fs, ...) + the serving
# extra (fastapi, uvicorn). torch is already present from the CPU wheel above.
RUN pip install ".[serving]"

# ---- runtime ----
FROM python:3.11-slim AS runtime

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY config/ ./config/

EXPOSE 8080
# App Runner routes traffic to the configured port (8080). The app loads the
# model at startup and reports readiness on /health.
CMD ["sh", "-c", "uvicorn serving.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
