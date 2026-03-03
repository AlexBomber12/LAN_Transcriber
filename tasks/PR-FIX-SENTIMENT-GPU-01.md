PR-FIX-SENTIMENT-GPU-01
=======================

Prompt (copy as-is into Codex)
------------------------------

```text
You are Codex Agent working on the LAN-Transcriber repository.

Goal
Fix 2 production issues:
1) Sentiment scoring crashes on long inputs with:
   RuntimeError: The size of tensor a (754) must match the size of tensor b (512)
   This is caused by running a transformers sentiment-analysis pipeline on text that exceeds the model max sequence length.
2) Worker is running on CPU because CUDA devices are not visible inside the container:
   torch.cuda.is_available() == False, torch.cuda.device_count() == 0
   Fix docker-compose.yml so the worker container can access the GPU.

Constraints
- Micro PR focused on these 2 fixes only.
- Keep the existing 100% statement and branch coverage enforcement intact.
- Do not introduce any new external services.
- CI does not have a GPU; tests must not require a GPU to pass.

Implementation requirements

A) Fix sentiment pipeline length crash (and make it non-fatal)
1) Locate sentiment scoring code
- In lan_transcriber/pipeline_steps/orchestrator.py there is a helper like _sentiment_score(text) used in run_pipeline:
  friendly = _sentiment_score(clean_text)
- Current implementation calls:
  hf_pipeline("sentiment-analysis")(text[:4000])[0]
  This does not guarantee truncation and can exceed max_length=512 for DistilBERT, causing a crash.

2) Apply truncation correctly
- Update _sentiment_score to call the pipeline with truncation enabled:
  - truncation=True
  - max_length=512
- Keep current behavior of slicing to reduce compute, but truncation must be enforced at the tokenizer level.
- Also pin the model explicitly to remove the runtime warning:
  - model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
  - set a specific revision if the repo already pins revisions elsewhere (optional but recommended for determinism).
- Ensure the sentiment device is explicitly CPU unless there is a good reason to use GPU:
  - device=-1
  This prevents the small sentiment model from competing with GPU resources used by ASR.

3) Make sentiment scoring non-fatal
- Sentiment is a convenience signal; it must never fail the recording processing.
- Wrap pipeline inference in try/except and return a neutral score (0.0) on any exception.
- Add a single warning log line when sentiment fails (include exception class), but do not spam logs.

B) Enable GPU visibility for worker container (docker-compose.yml)
1) Update docker-compose.yml
- In the worker service definition, add GPU access using the compose-native key:
  - gpus: all
- Do not use Swarm-only deploy reservations.
- If the project uses docker-compose.dev.yml, apply the same change there (worker service).

2) Add an explicit runtime check in logs
- In worker startup (or in orchestrator when selecting device), log one concise INFO line:
  - torch.cuda.is_available, device_count, torch.version.cuda
This makes it obvious in logs whether GPU is actually available.
- This log must not break tests; guard imports accordingly or keep it in a place already importing torch.

3) Documentation (small)
- In README/runbook, add a short section:
  - "GPU setup" and mention that worker uses gpus: all and requires NVIDIA Container Toolkit installed.
  - Provide a one-liner to verify on host:
    docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

C) Tests (100% coverage for touched code)
1) Add unit tests for sentiment truncation and non-fatal behavior
- Create/extend a test module that covers:
  a) _sentiment_score calls hf_pipeline with truncation=True and max_length=512 and device=-1
     - monkeypatch the hf_pipeline factory to a stub that records call kwargs and returns a stable result
  b) When the pipeline raises an exception, _sentiment_score returns 0.0 and does not raise
- Ensure 100% statement and branch coverage for the modified sentiment code path.

2) Add a smoke/config test for compose GPU config (optional)
- CI cannot verify GPU availability, but you can add a lightweight test that parses docker-compose.yml as YAML and asserts worker has "gpus: all".
- If you add such a test, keep it simple and fully covered.

Verification steps (include in PR description)
1) Rebuild and restart worker:
   docker compose build --no-cache worker
   docker compose up -d --force-recreate --no-deps worker
2) In the worker container:
   python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.version.cuda)"
   It should show True and device_count >= 1 on the GPU host.
3) Reprocess a recording that previously crashed in sentiment.
   The job must complete; sentiment must not crash processing even for long text.

Deliverables
- Updated sentiment scoring implementation with truncation and safe fallback
- docker-compose.yml updated so worker sees GPU (gpus: all), plus dev compose if applicable
- Small doc update for GPU verification
- Unit tests keeping 100% statement + branch coverage

Success criteria
- No more sentiment crash due to token length; _sentiment_score never raises.
- Worker container has GPU access when run on the NVIDIA host (torch.cuda.is_available True).
- CI remains green; tests do not require GPU.
```
