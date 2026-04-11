Run PLANNED PR

PR_ID: PR-SPEAKER-MERGE-DIAGNOSTICS-01
Branch: pr-speaker-merge-diagnostics-01
Title: Add diagnostics to speaker merge step: log pairwise similarities, overlap checks, and skip reasons

Follow AGENTS.md exactly for work mode, queue handling, CI, artifacts, MCP usage, and scope control. This is a focused MICRO PR. Keep the scope strict.

This task must be executed in 3 internal phases within a single run.

Phase 1 - Inspect and map
Read and confirm the current state of these files before coding:
- lan_transcriber/pipeline_steps/speaker_merge.py (merge_similar_speakers function, lines 362-383 where pairs are iterated and skipped silently)
- lan_transcriber/pipeline_steps/orchestrator.py (lines 3019-3050 where merge is called, line 1278 where speaker_merges is written to metadata)

Phase 2 - Implement
Implement exactly these changes. Do not add anything beyond these fixes.

CHANGE 1: Return diagnostics from merge_similar_speakers
Change the return type of merge_similar_speakers from tuple[list, dict] to tuple[list, dict, dict] where the third element is a diagnostics dict containing:
- "embedding_model_available": bool (was embedding_model not None)
- "speakers_found": list of speaker labels
- "centroids_computed": list of speaker labels that got a centroid
- "pairwise_scores": list of {"speaker_a": str, "speaker_b": str, "similarity": float, "overlap": bool, "action": str} where action is one of "merged", "skipped_low_similarity", "skipped_overlap", "skipped_already_merged"
- "merges_applied": dict (the merge_map)

In the pair iteration loop (lines 362-383), instead of silent continue, log at INFO level and record the skip reason:
- If similarity < threshold: log "speaker_merge: skip {a}<->{b} similarity={sim:.3f} < threshold={thr:.3f}" and record action="skipped_low_similarity"
- If overlap detected: log "speaker_merge: skip {a}<->{b} similarity={sim:.3f} overlap=True" and record action="skipped_overlap"
- If already merged (left_target == right_target): record action="skipped_already_merged"

When embedding_model is None, return diagnostics with embedding_model_available=False and empty lists.

CHANGE 2: Store diagnostics in diarization_metadata.json
In orchestrator.py, receive the diagnostics dict from merge_similar_speakers and store it in diarization_metadata.json under a new key "speaker_merge_diagnostics". Replace the existing "speaker_merges" key with just the merge_map from diagnostics (keep backward compat).

Also store diagnostics when merge is skipped entirely (embedding model unavailable, dummy fallback, single speaker, etc.) with the appropriate reason.

CHANGE 3: Log embedding model resolution outcome
In _resolve_pyannote_embedding_model, add an INFO log when the model IS successfully resolved (not just warnings on failure). Format: "speaker_merge: embedding model ready (source={source})" where source is "pipeline_attribute" or "standalone_inference".

Phase 3 - Test and verify
- Update existing speaker_merge tests to accept the new 3-tuple return.
- Add test_diagnostics_low_similarity: verify diagnostics contain "skipped_low_similarity" when similarity is below threshold.
- Add test_diagnostics_overlap: verify diagnostics contain "skipped_overlap" when speakers overlap.
- Add test_diagnostics_merged: verify diagnostics contain "merged" for successful merge.
- Add test_diagnostics_no_model: verify diagnostics contain embedding_model_available=False.
- Run full CI. All existing tests must pass.

Success criteria:
- After processing a recording, diarization_metadata.json contains speaker_merge_diagnostics with pairwise similarity scores and skip/merge reasons.
- Operator can look at the metadata and immediately see: (a) whether embedding model loaded, (b) what the similarity was between each speaker pair, (c) why merge was or was not applied.
- Existing behavior is unchanged. The diagnostics are purely additive.
- All existing tests pass with the updated return type.
