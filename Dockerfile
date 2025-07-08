FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt constraints.txt .
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY web_transcribe.py .

# Default CMD just shows help; real run done via compose/systemd.
CMD ["python", "web_transcribe.py", "--help"]
