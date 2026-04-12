Run PLANNED PR

PR_ID: PR-NOISE-SPEAKER-FILTER-01
Branch: pr-noise-speaker-filter-01
Title: Detect and flag noise-only diarization speakers based on snippet VAD analysis

Follow AGENTS.md exactly for work mode, queue handling, CI, artifacts, MCP usage, and scope control. This is a focused BIG PR, not a MICRO PR. Keep the scope strict.

Context
Pyannote sometimes creates a speaker cluster for background noise, music, or non-speech audio. These "speakers" have snippets that contain no actual speech. In a 6-speaker recording, SPEAKER_01 had 25.5 sec (2%) with a noise-only snippet. The speaker merge correctly did not merge it (low similarity to real speakers), but it still appears in the speaker list and transcript.

The fix: after snippet export, run VAD on each speaker's snippet. If the snippet contains no speech (or speech ratio below threshold), flag the speaker as "noise" in the diarization metadata and UI.

Phase 1 - Inspect and map
Read:
- lan_transcriber/pipeline_steps/snippets.py: how snippets are extracted and manifest is written
- lan_transcriber/pipeline_steps/orchestrator.py: snippet export stage, where snippets are available
- lan_app/templates/partials/speaker_review_cards.html: how speakers are rendered in UI
- lan_app/system_status.py or lan_app/diagnostics.py: existing VAD infrastructure (Silero VAD is already loaded for precheck)

Phase 2 - Implement

CHANGE 1: Add VAD check on speaker snippets
After snippet export, for each speaker's accepted snippet:
- Run Silero VAD (already available in the pipeline) on the snippet audio.
- Compute speech_ratio = speech_frames / total_frames.
- If speech_ratio < 0.3 (configurable), flag the speaker as noise_suspected=true.
- Store the result in snippets_manifest.json: add "speech_ratio" and "noise_suspected" per speaker entry.

CHANGE 2: Add noise indicator in diarization metadata
In diarization_metadata.json (or a new field in speaker_turns), add:
- "noise_speakers": ["SPEAKER_01"] (list of speakers flagged as noise)
- Include speech_ratio for each flagged speaker for traceability.

CHANGE 3: Show noise flag in UI
In the speaker review UI, if a speaker is flagged as noise:
- Show a "Likely noise" badge (gray/muted) next to "Unknown Speaker"
- Optionally collapse or de-emphasize noise speakers in the transcript

CHANGE 4: Exclude noise speakers from transcript (optional, configurable)
Add a config flag: EXCLUDE_NOISE_SPEAKERS_FROM_TRANSCRIPT (default: false).
When true, turns from noise-flagged speakers are excluded from the transcript export. When false, they are kept but marked.

Phase 3 - Test and verify
- Add test: snippet with no speech (silence/noise WAV) -> noise_suspected=true.
- Add test: snippet with clear speech -> noise_suspected=false.
- Add test: speech_ratio boundary (0.29 vs 0.31).
- Run full CI.

Success criteria:
- Noise-only speakers are flagged in metadata and UI.
- Real speakers with low but audible speech are NOT flagged (threshold 0.3 is conservative).
- Noise detection does not add significant processing time (VAD on a 3-8 sec snippet is fast).
- No existing tests break.
