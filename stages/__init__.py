"""Pipeline stages.

Each stage is an independently runnable CLI entrypoint. Airflow orchestrates
these stages; it does not define them. Stages communicate only through S3
(or local, in dev/CI) Parquet paths -- never shared local state.
"""
