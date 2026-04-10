Run PLANNED PR

PR_ID: PR-SPEAKER-MERGE-EMBEDDINGS-01
Branch: pr-speaker-merge-embeddings-01
Title: Auto-merge diarization speakers with similar voice embeddings and no temporal overlap

Follow AGENTS.md exactly for work mode, queue handling, CI, artifacts, MCP usage, and scope control. This is a focused BIG PR, not a MICRO PR. Keep the scope strict.

This task must be executed in 4 internal phases within a single run.

Phase 1 - Inspect and map
Read and confirm the current state of these files before coding:
- lan_transcriber/pipeline_steps/orchestrator.py (lines 2780-2830: where diarization result becomes diar_segments and feeds into build_speaker_turns)
- lan_transcriber/pipeline_steps/diarization_quality.py (smooth_speaker_turns, SpeakerTurnSmoothingResult, filter_flickering_speakers if present)
- lan_transcriber/pipeline_steps/speaker_turns.py (_diarization_segments, build_speaker_turns)
- lan_app/diarization_loader.py (how pyannote pipeline is loaded, what model is used)
- lan_app/speaker_bank.py (existing cosine similarity / embedding infrastructure)
- lan_app/config.py (Settings / AppSettings, existing diarization_* config fields)

Phase 2 - Design the embedding extraction
The pyannote diarization pipeline internally uses a speaker embedding model but does not expose per-speaker embeddings in its output. We need to extract them ourselves.

Approach: use the pyannote embedding model (the same one pyannote/speaker-diarization-3.1 uses internally, which is pyannote/wespeaker-voxceleb-resnet34-LM or similar) to compute embeddings from raw audio segments.

Before coding, verify which embedding model is bundled with the loaded diarization pipeline. Check:
- The pipeline's config.yaml or params for the embedding model reference
- Whether the pipeline object has an accessible .embedding attribute or sub-model
- If not directly accessible, use pyannote.audio.Inference with the embedding model separately

If the embedding model requires a separate download/load, ensure it shares the same device (GPU/CPU) as the diarization pipeline to avoid unnecessary transfers.

Phase 3 - Implement
Implement exactly these changes. Do not add anything beyond these.

CHANGE 1: Create a new module lan_transcriber/pipeline_steps/speaker_merge.py
This module implements the merge logic.

Function 1: extract_speaker_embeddings(audio_path, diar_segments, *, embedding_model, max_segments_per_speaker=5, segment_duration_sec=3.0) -> dict[str, np.ndarray]
- For each unique speaker in diar_segments, select up to max_segments_per_speaker of their longest segments.
- For each selected segment, extract the audio window and run it through the embedding model to get a vector.
- Average all vectors for that speaker into a single centroid embedding.
- Return a dict mapping speaker_label -> centroid_embedding.
- If a speaker has no usable segments (too short, extraction fails), skip them.

Function 2: compute_pairwise_similarity(embeddings: dict[str, np.ndarray]) -> list[tuple[str, str, float]]
- For every pair of speakers, compute cosine similarity between their centroid embeddings.
- Return list of (speaker_a, speaker_b, similarity) sorted by similarity descending.

Function 3: speakers_overlap(speaker_a, speaker_b, diar_segments, *, tolerance_sec=0.1) -> bool
- Check if speaker_a and speaker_b ever have overlapping diarization segments (within tolerance).
- Return True if any overlap exists, False otherwise.
- This is the contextual signal: if two "speakers" never talk simultaneously, they are likely the same person.

Function 4: merge_similar_speakers(diar_segments, *, audio_path, embedding_model, similarity_threshold=0.80, max_segments_per_speaker=5) -> tuple[list[dict], dict[str, str]]
- Call extract_speaker_embeddings to get centroids.
- Call compute_pairwise_similarity.
- For each pair with similarity >= similarity_threshold AND speakers_overlap returns False:
  - Merge the speaker with less total speech time into the speaker with more total speech time.
  - Reassign all diar_segments of the merged speaker to the dominant speaker.
  - Record the mapping in a merge_map dict (merged_label -> kept_label).
- Apply merges transitively (if A merges into B and B merges into C, A should map to C).
- Return (updated_diar_segments, merge_map).
- Log each merge: "Merged speaker {X} into {Y}: similarity={sim:.3f}, overlap=False, X_seconds={n}, Y_seconds={m}"

CHANGE 2: Load the embedding model in the orchestrator
In orchestrator.py, after loading the pyannote diarization pipeline (or alongside it):
- Attempt to access the embedding sub-model from the diarization pipeline object. Check pipeline._embedding or pipeline.embedding or pipeline.parameters()["embedding"] etc.
- If not directly accessible, load pyannote.audio.Inference separately with the model "pyannote/wespeaker-voxceleb-resnet34-LM" (or whatever the pipeline config references) on the same device.
- Cache the embedding model alongside the diarisation pipeline so it is loaded once, not per recording.
- If embedding model loading fails (e.g. model not available, no GPU memory), log a warning and skip the merge step entirely. This must be graceful - the pipeline should still work without the merge step.

CHANGE 3: Wire merge step into the pipeline
In the orchestrator, between the line that computes diar_segments = _diarization_segments(diarization) and the line that calls build_speaker_turns:
- Call merge_similar_speakers(diar_segments, audio_path=..., embedding_model=...).
- Use the returned updated diar_segments for the rest of the pipeline.
- Store the merge_map in diarization_metadata.json under a new key "speaker_merges" for traceability.
- If merge step is skipped (no embedding model), log and continue with original diar_segments.

CHANGE 4: Add config knobs
In the pipeline Settings (orchestrator.py or config.py, wherever diarization_* settings live):
- speaker_merge_enabled: bool = True
- speaker_merge_similarity_threshold: float = 0.80
- speaker_merge_max_segments: int = 5

Phase 4 - Test and verify
Add tests in a new file tests/test_speaker_merge.py:
- test_merge_identical_embeddings_no_overlap: 2 speakers with identical embeddings and no overlapping segments. Should merge into 1 speaker.
- test_no_merge_different_embeddings: 2 speakers with very different embeddings (similarity < 0.5). Should remain 2 speakers.
- test_no_merge_overlapping_speakers: 2 speakers with high embedding similarity BUT overlapping segments. Should NOT merge (they really are different people talking over each other).
- test_no_merge_below_threshold: 2 speakers with similarity 0.75 (below 0.80 threshold). Should NOT merge.
- test_three_speakers_two_merge: 3 speakers, A and B have similar embeddings and no overlap, C is different. A and B merge, C stays.
- test_transitive_merge: 3 speakers, A similar to B, B similar to C, no overlaps. All should merge into the one with most speech.
- test_merge_disabled: speaker_merge_enabled=False. No merging regardless of similarity.
- test_graceful_fallback_no_model: embedding_model=None. Should return original segments unchanged.

Mock the embedding model in tests: create a fake that returns predetermined vectors so tests are deterministic and do not require GPU/model download.

Run full CI. All existing tests must pass.

Success criteria:
- The false SPEAKER_00 in single-speaker recordings is automatically detected (high embedding similarity to SPEAKER_01, no temporal overlap) and merged into SPEAKER_01.
- Real multi-speaker recordings where speakers have different voices are NOT affected.
- Real multi-speaker recordings where speakers overlap (talk simultaneously) are NOT affected even if voices happen to be similar.
- The merge step is logged and traceable via diarization_metadata.json.
- The merge step is gracefully skipped if the embedding model is unavailable.
- The merge step adds minimal latency (embedding extraction for a few segments per speaker is fast compared to full diarization).
- Config knobs allow tuning or disabling the feature.
