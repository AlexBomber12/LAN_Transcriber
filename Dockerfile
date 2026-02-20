FROM python:3.12-slim AS base

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

FROM base AS runtime-full
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir -r requirements.txt \
      --extra-index-url https://download.pytorch.org/whl/cu121 \
    && python -m pip install --no-cache-dir whisperx==3.4.2
CMD ["python", "web_transcribe.py"]

FROM base AS runtime-lite
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir -r ci-requirements.txt \
    && python -m pip install --no-cache-dir -e .[test] \
    && python -m pip install --no-cache-dir whisperx==3.4.2
ENV CI=true
CMD ["uvicorn", "web_transcribe:app", "--host", "0.0.0.0", "--port", "7860"]
