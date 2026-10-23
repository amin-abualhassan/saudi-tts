FROM python:3.10.19-slim-bookworm@sha256:23f63358922e79a794f71be8f3723c84e5ccca9638af3f74456dc73d6184499e
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    PYTHONPATH=/opt/fish-source:/opt/runtime OMP_NUM_THREADS=8
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.lock /opt/requirements.lock
RUN pip install --no-cache-dir -r /opt/requirements.lock
COPY fish-source /opt/fish-source
COPY runtime /opt/runtime
WORKDIR /data
ENTRYPOINT ["python", "/opt/runtime/server.py"]
CMD ["--container", "--model", "/data/model", "--codec", "/original/codec.pth", "--reference-root", "/references", "--port", "3012"]
