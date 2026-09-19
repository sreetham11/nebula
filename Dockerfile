# RAILPULSE / NEBULA X — single-container image.
#
# Serves BOTH the FastAPI backend and the dashboard on one origin, so the
# deployed service is one Cloud Run URL with no CORS hop and nothing else to
# host. See DEPLOYMENT.md for the deploy commands.
#
# Self-contained: the synthetic dataset is generated and the detection
# pipeline (baseline + autoencoder + validation + ablation) is trained and
# scored during `docker build`, so nothing is computed at container start.
# The only artifacts copied in are models/real_* — those were trained in
# Colab on the real PS3 data and cannot be regenerated from this repo.
#
# Cloud Run injects $PORT and expects the container to listen on it; the CMD
# honours that, defaulting to 8080 for local `docker run`.
#
# Local:
#   docker build -t railpulse .
#   docker run --rm -p 8080:8080 railpulse
#   open http://localhost:8080

FROM python:3.12-slim

WORKDIR /app

# libgomp1: OpenMP runtime that scikit-learn / numpy / xgboost link against.
# Debian slim omits it, and its absence shows up as the classic
# "libgomp.so.1: cannot open shared object file" at import time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# CPU-only torch first, from PyTorch's CPU index. The default PyPI wheel
# drags in the bundled CUDA runtime — well over 2GB of libraries this
# service never executes, which is build minutes and image size for nothing.
# Installing it first means the requirements.txt line below is already
# satisfied and will not pull the CUDA build over the top.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.12.1 \
    && pip install --no-cache-dir -r requirements.txt

COPY data_gen/ data_gen/
COPY pipeline/ pipeline/
COPY api/ api/
COPY dashboard/ dashboard/

# The Colab-trained real-data artifacts. Copied before the pipeline run so
# the synthetic models it writes land alongside them in the same directory.
COPY models/ models/

# Generate the synthetic dataset, then train, score and validate. Done at
# build time so the container serves immediately on start with no cold-start
# training. This is the slow step (CPU training, a few minutes) — if Cloud
# Build times out, raise --timeout, see DEPLOYMENT.md.
RUN python data_gen/generate_data.py \
    && cd pipeline \
    && python run_pipeline.py \
    && python ablation.py

ENV PORT=8080
EXPOSE 8080

WORKDIR /app/api
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
