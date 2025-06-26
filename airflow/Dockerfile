FROM apache/airflow:2.9.1

USER root

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libpq-dev \
    python3-dev \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

USER airflow

COPY requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install --no-cache-dir --force-reinstall "sqlalchemy>=1.4,<2.0"
