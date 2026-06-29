# Multi-stage build: one lean image that can run ANY pipeline stage. The stage
# is selected at runtime via the entrypoint, e.g.:
#   docker run --rm ml-pipeline ingest --run-date 2025-01-01
#
# The builder installs the project + its runtime extras into a venv; the runtime
# image copies only that venv (plus the data-file config), keeping it small.
# Airflow is NOT in this image -- it runs in its own container via
# docker-compose.airflow.yml and calls these stages.

# ---- builder ----
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Build deps for any sdist-only wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy the bits the build backend needs (packages + readme referenced in
# pyproject), then install the project with the stage-runner extras. The
# package (stages/models/dags) lands in the venv, so it's importable at runtime
# without PYTHONPATH gymnastics.
COPY pyproject.toml README.md ./
COPY stages/ ./stages/
COPY models/ ./models/
COPY dags/ ./dags/
RUN pip install --upgrade pip && pip install ".[spark,train,ingest]"

# ---- runtime ----
FROM python:3.11-slim AS runtime

# Spark needs a JRE; keep it minimal.
RUN apt-get update && apt-get install -y --no-install-recommends \
        default-jre-headless && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    SPARK_LOCAL_IP=127.0.0.1

WORKDIR /app
# config/ (YAML data files) and generate_fixtures.py are not part of the
# installed package, so copy them where the stages expect them (cwd-relative).
COPY config/ ./config/
COPY generate_fixtures.py ./

# entrypoint.sh maps a stage name -> the right module. The first CLI arg is the
# stage (ingest|features|validate|train|register|fixtures); the rest pass through.
COPY infra/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["--help"]
