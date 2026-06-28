# Multi-stage build: one lean image that can run ANY pipeline stage. The stage
# is selected at runtime via the entrypoint, e.g.:
#   docker run --rm ml-pipeline ingest --run-date 2025-01-01
#
# A builder stage compiles/installs dependencies; the final stage copies only
# the installed environment + source, keeping the runtime image small.

# ---- builder ----
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Install build deps for pyarrow/spark wheels if needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt ./
# Install into a venv we can copy wholesale into the runtime image.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# Install core + spark + train extras. (Airflow runs in its own image via
# docker-compose, not in this stage-runner image.)
RUN pip install --upgrade pip && \
    pip install ".[spark,train]" || pip install -r requirements.txt

# ---- runtime ----
FROM python:3.11-slim AS runtime

# Spark needs a JRE; keep it minimal.
RUN apt-get update && apt-get install -y --no-install-recommends \
        default-jre-headless && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app
COPY stages/ ./stages/
COPY models/ ./models/
COPY dags/ ./dags/
COPY config/ ./config/
COPY generate_fixtures.py ./

# entrypoint.sh maps a stage name -> the right module. The first CLI arg is the
# stage (ingest|features|validate|train|register); the rest pass through.
COPY infra/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["--help"]
