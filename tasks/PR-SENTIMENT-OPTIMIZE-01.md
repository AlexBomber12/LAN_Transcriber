Run PLANNED PR

PR_ID: PR-SENTIMENT-OPTIMIZE-01
Branch: pr-sentiment-optimize-01
Title: Cache sentiment pipeline or replace with LLM-derived friendly score

Follow AGENTS.md exactly for work mode, queue handling, CI, artifacts, MCP usage, and scope control. This is a MICRO PR.

Context
In orchestrator.py:565, _sentiment_score creates a new transformers pipeline (distilbert-base-uncased-finetuned-sst-2-english) on EVERY call. This loads a full transformer model into memory, runs inference on truncated text, then discards the model. The result is a single "friendly" integer (0-100) used in summary metadata.

This is expensive for a secondary signal. Two options: cache the pipeline globally, or eliminate the local model entirely and extract friendliness from the LLM summary output.

Phase 1 - Inspect
Read:
- lan_transcriber/pipeline_steps/orchestrator.py: _sentiment_score (line ~565), where friendly is used (line ~3451)
- lan_transcriber/pipeline_steps/summary_builder.py: how summary is structured, whether "tone" or "friendliness" could be extracted from LLM output
- Check if friendly score is displayed anywhere in UI or used in any logic beyond summary metadata

Phase 2 - Implement
Choose ONE of these approaches (prefer Option B for simplicity):

Option A: Cache the pipeline
Create a module-level cached pipeline:

_SENTIMENT_PIPELINE = None
_SENTIMENT_PIPELINE_LOCK = threading.Lock()

def _get_sentiment_pipeline():
    global _SENTIMENT_PIPELINE
    if _SENTIMENT_PIPELINE is None:
        with _SENTIMENT_PIPELINE_LOCK:
            if _SENTIMENT_PIPELINE is None:
                from transformers import pipeline as hf_pipeline
                _SENTIMENT_PIPELINE = hf_pipeline(
                    "sentiment-analysis",
                    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
                    device=-1,
                )
    return _SENTIMENT_PIPELINE

Option B: Extract from LLM output (preferred)
Add a "tone_score" field to the structured LLM summary prompt. The LLM already analyzes the full text for emotional_summary. Ask it to also output a tone_score (0-100, where 100 = very positive/friendly). Use this instead of the local transformer model.

Remove _sentiment_score entirely. Remove the transformers sentiment dependency.

Phase 3 - Test and verify
- Run full CI.
- Verify friendly score is still present in summary metadata.
- If Option B: verify LLM output includes tone_score and it's reasonable.
- If Option A: verify pipeline is loaded only once across multiple recordings.
- Verify no regression in summary quality.

Success criteria:
- No transformer model loaded per call.
- friendly/tone score still available in metadata.
- Reduced memory usage and processing time for summary stage.
- No existing tests break.
