# RAILPULSE / NEBULA X -- one container serving the API and the dashboard.
#
# Self-contained image: the synthetic dataset is generated and the anomaly
# detection pipeline (baseline + autoencoder + validation) is trained and
# scored once during `docker build`, so the resulting image needs nothing
# else at deploy time -- no volumes, no pre-existing local artifacts. The
# real-data models (Colab-trained on the PS3 Door and Rail_Corrugation
# datasets) cannot be regenerated here, so they are copied in from the repo.
#
# The FastAPI app also serves the dashboard (dashboard/) from "/", so one
# URL is the whole product.
#
# Designed for Google Cloud Run: Cloud Run injects $PORT and expects the
# container to listen on it; the CMD below honors it (8080 by default).
#
# Build/run locally:
#   docker build -t nebula-x .
#   docker run --rm -p 8080:8080 nebula-x
#   open http://localhost:8080/
#
# API keys are NOT baked in (see .dockerignore). Set ANTHROPIC_API_KEY and
# EXA_API_KEY as environment variables on the running service.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# libgomp1: OpenMP runtime needed by scikit-learn, numpy and xgboost on
# Debian slim images.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU-only PyTorch first. The default Linux wheel bundles ~2 GB of CUDA
# libraries that Cloud Run cannot use; the CPU wheel is a fraction of that
# and starts faster. requirements.txt then sees torch==2.12.1 as satisfied.
RUN pip install --no-cache-dir torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY data_gen/ data_gen/
COPY pipeline/ pipeline/
COPY api/ api/

# Generate the synthetic dataset and train/score/validate the models once,
# at build time, so the image is immediately ready to serve on container
# startup -- no cold-start training. This is the slowest step (CPU-only
# training, a couple of minutes); if it times out in Cloud Build, raise the
# build timeout.
RUN python data_gen/generate_data.py \
    && cd pipeline && python run_pipeline.py

# Placed after the slow step so editing them never retrains the models.
COPY models/real_door_autoencoder.pt models/real_door_scaler.pkl models/real_door_threshold.pkl models/real_rail_classifier.pkl models/real_rail_label_encoder.pkl models/
COPY validation_outputs/ablation_comparison.csv validation_outputs/
COPY dashboard/ dashboard/

# Fail the build rather than ship a secret.
RUN if find /app -name ".env*" | grep -q .; then echo "refusing to build: a .env file is in the image" >&2; exit 1; fi

ENV PORT=8080
EXPOSE 8080

WORKDIR /app/api
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
