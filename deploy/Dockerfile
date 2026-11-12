# Brain-MRI world model web app (CPU inference).
# Build from the repository root:  docker build -f deploy/Dockerfile -t brainwm .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/mpl SRC_DIR=/srv/src MODEL_DIR=/srv/models

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch (about 200 MB instead of several GB of CUDA libraries)
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install torch --index-url ${TORCH_INDEX}

COPY deploy/requirements-app.txt /tmp/req.txt
RUN pip install -r /tmp/req.txt

RUN useradd --create-home --uid 10001 app
WORKDIR /srv
COPY src/ /srv/src/
COPY deploy/app/ /srv/app/
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz').status==200 else 1)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", \
     "--forwarded-allow-ips", "*", "--workers", "1", "--timeout-keep-alive", "30"]
