Run PLANNED PR

PR_ID: PR-SPEAKER-MERGE-DUAL-THRESHOLD-01
Branch: pr-speaker-merge-dual-threshold-01
Title: Dual similarity threshold for speaker merge: strict when overlap, relaxed when no overlap

Follow AGENTS.md exactly for work mode, queue handling, CI, artifacts, MCP usage, and scope control. This is a MICRO PR.

Context
Speaker merge uses a single similarity_threshold (default 0.80). A real single-speaker recording produced similarity=0.786 between SPEAKER_00 and SPEAKER_01 with overlap=false, but merge was skipped because 0.786 < 0.80. The absence of temporal overlap is a strong contextual signal that two labels represent the same person. The merge logic should use two thresholds: a strict one when speakers overlap (real different people can have similar voices), and a relaxed one when they never overlap (almost certainly the same person).

Phase 1 - Inspect
Read these files:
- lan_transcriber/pipeline_steps/speaker_merge.py: merge_similar_speakers, the pair iteration loop where similarity_threshold is compared
- lan_transcriber/pipeline_steps/orchestrator.py: Settings class where speaker_merge_similarity_threshold is defined
- lan_app/config.py: AppSettings where speaker_merge_similarity_threshold is defined

Phase 2 - Implement

CHANGE 1: Add a second threshold constant and config
In speaker_merge.py, add:
DEFAULT_SPEAKER_MERGE_NO_OVERLAP_SIMILARITY_THRESHOLD = 0.70

In orchestrator.py Settings, add:
speaker_merge_no_overlap_similarity_threshold: float = Field(
    default=DEFAULT_SPEAKER_MERGE_NO_OVERLAP_SIMILARITY_THRESHOLD,
    ge=0.0, le=1.0,
    validation_alias=AliasChoices(
        "speaker_merge_no_overlap_similarity_threshold",
        "SPEAKER_MERGE_NO_OVERLAP_SIMILARITY_THRESHOLD",
        "LAN_SPEAKER_MERGE_NO_OVERLAP_SIMILARITY_THRESHOLD",
    ),
)

In lan_app/config.py AppSettings, add the same field.
Wire it through _build_pipeline_settings in worker_tasks.py.

CHANGE 2: Update merge_similar_speakers signature
Add parameter no_overlap_similarity_threshold: float = DEFAULT_SPEAKER_MERGE_NO_OVERLAP_SIMILARITY_THRESHOLD.

CHANGE 3: Implement dual threshold logic
In the pair iteration loop of merge_similar_speakers, change the logic from:

    if similarity < similarity_threshold:
        break

To:
    # First pass filter: skip pairs below the relaxed threshold entirely
    if similarity < no_overlap_similarity_threshold:
        # record diagnostics and break/continue
        ...

    # Check overlap
    has_overlap = speakers_overlap(...)

    # Apply the appropriate threshold
    effective_threshold = similarity_threshold if has_overlap else no_overlap_similarity_threshold

    if similarity < effective_threshold:
        # record diagnostics: skipped_low_similarity (note which threshold was used)
        ...
        continue

    if has_overlap:
        # record diagnostics: skipped_overlap
        ...
        continue

    # Merge
    ...

Important: the pairs list is sorted by similarity descending. The old code could `break` when below threshold because all remaining pairs would also be below. With dual thresholds, use `break` when below no_overlap_similarity_threshold (the lower one), and `continue` when between the two thresholds but overlap is true.

CHANGE 4: Update diagnostics
In pairwise_scores entries, add an "effective_threshold" field showing which threshold was applied (0.80 for overlap, 0.70 for no-overlap). This makes it clear in metadata why a pair was merged or skipped.

Phase 3 - Test and verify
Update existing tests in tests/test_speaker_merge.py:
- test_merge_relaxed_threshold_no_overlap: similarity=0.75, overlap=false, strict_threshold=0.80, relaxed_threshold=0.70. Should MERGE (0.75 >= 0.70 and no overlap).
- test_no_merge_strict_threshold_with_overlap: similarity=0.85, overlap=true, strict_threshold=0.80. Should NOT MERGE (overlap blocks even above strict threshold... wait, 0.85 > 0.80 so it should merge? No - overlap=true means we skip regardless of similarity). Revisit: the original logic skips when overlap=true regardless of similarity. Keep that behavior.
- test_no_merge_below_relaxed: similarity=0.65, overlap=false, relaxed_threshold=0.70. Should NOT MERGE.
- test_real_case_0786: similarity=0.786, overlap=false, strict=0.80, relaxed=0.70. Should MERGE. This is the exact case from the Plaud comparison recording.
- Run full CI.

Success criteria:
- The recording with SPEAKER_00 (similarity=0.786, overlap=false) is now correctly merged into SPEAKER_01.
- Real multi-speaker recordings with overlap are NOT affected (strict threshold still applies, but overlap blocks merge anyway).
- The dual threshold is configurable via environment variables.
- Diagnostics show which threshold was applied for each pair.
- No existing tests break.
