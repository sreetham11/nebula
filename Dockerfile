# NEBULA X FastAPI backend.
#
# Self-contained image: the synthetic dataset is generated and the anomaly
# detection pipeline (baseline + autoencoder + validation) is trained and
# scored once during `docker build`, so the resulting image needs nothing
# else at deploy time -- no volumes, no pre-existing local artifacts, no
# dependency on data/ or models/ having been generated on the machine that
# built the image (both are git-ignored; see .gitignore).
#
# Designed for Google Cloud Run -- see DEPLOYMENT.md for the deploy
# commands. Cloud Run injects $PORT and expects the container to listen on
# it, which the CMD below honors (defaulting to 8080 for local `docker run`
# testing where $PORT isn't set).
#
# Build/run locally:
#   docker build -t nebula-x-api .
#   docker run --rm -p 8080:8080 nebula-x-api
#   curl http://localhost:8080/health

FROM python:3.12-slim

WORKDIR /app

# libgomp1 is needed at runtime by scikit-learn/numpy on Debian slim images
# (OpenMP shared library isn't included by default and its absence is a
# common "libgomp.so.1: cannot open shared object file" surprise otherwise).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY data_gen/ data_gen/
COPY pipeline/ pipeline/
COPY api/ api/

# Generate the synthetic dataset and train/score/validate the models once,
# at build time, so the image is immediately ready to serve on container
# startup -- no cold-start training, fully reproducible from source. This
# is the slowest step (CPU-only training, a couple of minutes); if it times
# out in Cloud Build, raise the build timeout -- see DEPLOYMENT.md.
RUN python data_gen/generate_data.py \
    && cd pipeline && python run_pipeline.py

ENV PORT=8080
EXPOSE 8080

WORKDIR /app/api
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
