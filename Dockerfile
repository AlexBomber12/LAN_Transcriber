FROM python:3.12-slim AS base

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

FROM base AS runtime-full
RUN apt-get update \
    && apt-get install -y --no-install-recommends patchelf binutils \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir -r requirements-cu121.txt \
    && python -m pip check
RUN python - <<'PY'
import glob, site, subprocess, sys
root = site.getsitepackages()[0]
libs = glob.glob(root + "/**/libctranslate2*.so*", recursive=True)
print("Found libctranslate2 libs:", len(libs))
if not libs:
    raise SystemExit("No libctranslate2*.so* found; expected ctranslate2 to be installed")
for p in libs:
    subprocess.check_call(["patchelf", "--clear-execstack", p])
    out = subprocess.check_output(["readelf", "-W", "-l", p], text=True)
    gnu = next((l for l in out.splitlines() if "GNU_STACK" in l), "")
    print(p, gnu)
    if "RWE" in gnu:
        raise SystemExit(f"GNU_STACK still RWE for {p}")
PY
RUN python -c "import whisperx; import faster_whisper; import ctranslate2; print('deps ok')"
CMD ["uvicorn", "lan_app.api:app", "--host", "0.0.0.0", "--port", "7860"]

FROM base AS runtime-lite
RUN apt-get update \
    && apt-get install -y --no-install-recommends patchelf binutils \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip check
RUN python - <<'PY'
import glob, site, subprocess, sys
root = site.getsitepackages()[0]
libs = glob.glob(root + "/**/libctranslate2*.so*", recursive=True)
print("Found libctranslate2 libs:", len(libs))
if not libs:
    raise SystemExit("No libctranslate2*.so* found; expected ctranslate2 to be installed")
for p in libs:
    subprocess.check_call(["patchelf", "--clear-execstack", p])
    out = subprocess.check_output(["readelf", "-W", "-l", p], text=True)
    gnu = next((l for l in out.splitlines() if "GNU_STACK" in l), "")
    print(p, gnu)
    if "RWE" in gnu:
        raise SystemExit(f"GNU_STACK still RWE for {p}")
PY
RUN python -c "import whisperx; import faster_whisper; import ctranslate2; print('deps ok')"
ENV CI=true
CMD ["uvicorn", "lan_app.api:app", "--host", "0.0.0.0", "--port", "7860"]
