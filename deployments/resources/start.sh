#!/bin/bash

# Start Supabase PostgreSQL using Docker
echo "Starting Supabase PostgreSQL..."
docker run -d \
  --name postgres-db \
  -p 5432:5432 \
  -e POSTGRES_DB="${POSTGRES_DB}" \
  -e POSTGRES_USER="${POSTGRES_USER}" \
  -e POSTGRES_PASSWORD="${POSTGRES_PASSWORD}" \
  -e PGDATA="/var/lib/postgresql/data" \
  -v postgres-data:/var/lib/postgresql/data \
  supabase/postgres:15.6.1.113

# Wait for PostgreSQL to start
sleep 10

# Start MinIO using Docker
echo "Starting MinIO..."
docker run -d \
  --name mlflow-s3 \
  -p 9000:9000 \
  -p 9001:9001 \
  -e MINIO_ROOT_USER="${AWS_ACCESS_KEY_ID}" \
  -e MINIO_ROOT_PASSWORD="${AWS_SECRET_ACCESS_KEY}" \
  -v minio-data:/data \
  minio/minio:latest \
  server /data --console-address ':9001' --address ':9000'

# Keep the container running
wait
