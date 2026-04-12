Run PLANNED PR

PR_ID: PR-SPEAKER-MERGE-DUAL-THRESHOLD-01
Branch: pr-speaker-merge-dual-threshold-01
Title: Dual similarity threshold and overlap ratio tolerance for speaker merge

Follow AGENTS.md exactly for work mode, queue handling, CI, artifacts, MCP usage, and scope control. This is a MICRO PR.

Context
Two issues prevent speaker merge from working on a real single-speaker recording:

1. SIMILARITY THRESHOLD: similarity=0.786 with overlap=false should merge, but the single threshold 0.80 blocks it. Fix: use a relaxed threshold (0.70) when speakers never overlap, strict (0.80) when they do.

2. OVERLAP DETECTION: pyannote creates tiny boundary overlaps (0.456 sec out of 454 sec total speech, ratio=0.001). The current speakers_overlap function treats ANY overlap > 0.1 sec as "real overlap", which triggers the strict threshold. Fix: use overlap RATIO instead of absolute seconds. An overlap ratio of 0.1% is noise, not simultaneous speech.

Both fixes are needed together. Fix 1 without fix 2 still fails because overlap=true forces the strict threshold. Fix 2 without fix 1 would help (overlap becomes false, so similarity 0.786 > 0.70 would merge) but the dual threshold is still correct design.

Phase 1 - Inspect
Read:
- lan_transcriber/pipeline_steps/speaker_merge.py: speakers_overlap (the per-segment-pair check), merge_similar_speakers (pair iteration loop), DEFAULT_SPEAKER_MERGE_OVERLAP_TOLERANCE_SEC
- lan_transcriber/pipeline_steps/orchestrator.py: Settings class, speaker_merge config fields
- lan_app/config.py: AppSettings speaker_merge fields

Phase 2 - Implement

FIX 1: Replace speakers_overlap with speakers_overlap_ratio
Rename or replace the speakers_overlap function to compute the TOTAL overlap ratio between two speakers instead of just checking if ANY overlap exists:

def speakers_overlap_ratio(
    speaker_a: str,
    speaker_b: str,
    diar_segments: Sequence[dict[str, Any]],
) -> float:
    """Return the fraction of the shorter speaker's total speech that overlaps with the other speaker."""

Logic:
- Compute total speech seconds for speaker_a and speaker_b.
- Compute total overlap seconds between them (sum of all pairwise segment overlaps, merged to avoid double-counting).
- Return overlap_seconds / min(speaker_a_total, speaker_b_total). Using the shorter speaker as denominator because even a small absolute overlap is significant if the speaker only has 10 seconds of speech.
- Return 0.0 if either speaker has no segments.

Keep the old speakers_overlap function for backward compatibility but make merge_similar_speakers use speakers_overlap_ratio.

FIX 2: Add overlap ratio threshold config
Add constants:
DEFAULT_SPEAKER_MERGE_OVERLAP_RATIO_THRESHOLD = 0.05

This means: if less than 5% of the shorter speaker's speech overlaps with the other, consider it noise (boundary artifacts), not real simultaneous speech.

Add to Settings:
speaker_merge_overlap_ratio_threshold: float = Field(default=0.05, ge=0.0, le=1.0, ...)

FIX 3: Dual similarity threshold
Add constant:
DEFAULT_SPEAKER_MERGE_NO_OVERLAP_SIMILARITY_THRESHOLD = 0.70

Add to Settings:
speaker_merge_no_overlap_similarity_threshold: float = Field(default=0.70, ge=0.0, le=1.0, ...)

Add parameter to merge_similar_speakers:
no_overlap_similarity_threshold: float = DEFAULT_SPEAKER_MERGE_NO_OVERLAP_SIMILARITY_THRESHOLD

FIX 4: Update merge logic
In the pair iteration loop of merge_similar_speakers:

    overlap_ratio = speakers_overlap_ratio(left_target, right_target, overlap_segments)
    has_meaningful_overlap = overlap_ratio > overlap_ratio_threshold

    effective_threshold = similarity_threshold if has_meaningful_overlap else no_overlap_similarity_threshold

    if similarity < effective_threshold:
        diagnostics["pairwise_scores"].append({
            "speaker_a": ..., "speaker_b": ...,
            "similarity": similarity,
            "overlap_ratio": overlap_ratio,
            "has_meaningful_overlap": has_meaningful_overlap,
            "effective_threshold": effective_threshold,
            "action": "skipped_low_similarity",
        })
        continue

    if has_meaningful_overlap:
        diagnostics["pairwise_scores"].append({
            ...,
            "action": "skipped_overlap",
        })
        continue

    # Merge
    ...

Note: the old boolean "overlap" field in diagnostics should be replaced with "overlap_ratio" (float) and "has_meaningful_overlap" (bool) for better debugging.

FIX 5: Wire config
Wire speaker_merge_no_overlap_similarity_threshold and speaker_merge_overlap_ratio_threshold through:
- orchestrator.py Settings
- lan_app/config.py AppSettings
- worker_tasks.py _build_pipeline_settings

Phase 3 - Test and verify
- test_merge_no_meaningful_overlap: overlap_ratio=0.001 (0.1%), similarity=0.786, relaxed_threshold=0.70. Should MERGE.
- test_no_merge_meaningful_overlap: overlap_ratio=0.15 (15%), similarity=0.786, strict_threshold=0.80. Should NOT MERGE.
- test_no_merge_below_relaxed: similarity=0.65, overlap_ratio=0.0. Should NOT MERGE (below 0.70).
- test_real_case: simulate the exact scenario: SPEAKER_01 with 443 sec, SPEAKER_00 with 11.5 sec, overlap 0.456 sec. overlap_ratio = 0.456/11.575 = 0.039 < 0.05 threshold. has_meaningful_overlap=false. similarity=0.786 > 0.70. MERGE.
- Run full CI.

Success criteria:
- The Plaud comparison recording now produces a single SPEAKER_01 with no false SPEAKER_00.
- Real multi-speaker recordings with meaningful overlap (> 5% of shorter speaker's speech) are not affected.
- Diagnostics show overlap_ratio, has_meaningful_overlap, and effective_threshold for each pair.
- All config values are tunable via environment variables.
- No existing tests break.
