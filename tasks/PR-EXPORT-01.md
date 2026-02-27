PR-EXPORT-01

```text
Role: You are Codex Agent working inside the LAN_Transcriber repository.

Branch: pr/export-01
PR title: PR-EXPORT-01 Export-only output: OneNote-ready markdown + ZIP download per recording
Base branch: main

Goal:
1) Provide an export-only workflow: generate a clean markdown note suitable for pasting into OneNote.
2) Add a Download ZIP endpoint per recording containing the markdown note plus core artifacts.
3) Add an Export section to the recording detail overview tab:
   - preview textarea with the markdown
   - Copy button
   - Download ZIP link

Scope:
- Do not remove Microsoft Graph / OneNote publish yet. This PR only adds export features.
- Export must work even when summary or transcript artifacts are not ready yet (graceful placeholders).

A) Export builder module
- Add lan_app/exporter.py with functions:
  1) build_onenote_markdown(recording: dict, *, settings: AppSettings) -> str
     - Build a single markdown document.
     - Sections (only include when data exists):
       - Title: topic (or recording filename) + captured_at
       - Metadata: recording id, filename, duration, language (if available)
       - Summary bullets
       - Decisions
       - Action items (render as checklist)
       - Questions (optional)
       - Emotional summary (optional)
       - Metrics (optional short block: interruptions, questions, decisions, action items)
       - Transcript (speaker turns if available, else transcript text)
     - Use derived artifacts when present:
       - derived/summary.json
       - derived/transcript.json
       - derived/speaker_turns.json
       - derived/metrics.json
     - Keep output stable and deterministic (same inputs -> same markdown).
  2) build_export_zip_bytes(recording_id: str, *, settings: AppSettings, include_snippets: bool = False) -> bytes
     - Build ZIP in memory using io.BytesIO + zipfile.
     - Always include:
       - onenote.md (from build_onenote_markdown)
       - manifest.json (recording_id, created_at, filenames included)
     - Include if present:
       - derived/summary.json
       - derived/transcript.json
       - derived/speaker_turns.json
       - derived/metrics.json
       - derived/segments.json
       - derived/lang_spans.json (if present)
     - If include_snippets is True:
       - include derived/snippets/* (wav files), preserving relative paths inside ZIP.
     - Never fail when a file is missing; just skip it.

B) UI endpoints
- In lan_app/ui_routes.py:
  1) Add GET /ui/recordings/{recording_id}/export.zip
     - Validate recording exists.
     - Query param: include_snippets=1 to include snippets.
     - Return Response with:
       - media_type "application/zip"
       - Content-Disposition attachment filename "export_<recording_id>.zip"
  2) In ui_recording_detail:
     - For overview tab, compute export_text using exporter.build_onenote_markdown(rec, settings=_settings)
     - Pass export_text into template context.

C) Template changes
- In lan_app/templates/recording_detail.html, overview tab:
  - Add a new section "Export" above the existing OneNote publish section (do not remove OneNote section in this PR).
  - Include:
    1) <textarea id="export-text" ...> with export_text
    2) Copy button that copies textarea content to clipboard (JS)
    3) Download ZIP link to /ui/recordings/{{ rec.id }}/export.zip
  - Keep styling consistent with the existing UI. Avoid introducing <br> tags.

D) Tests
- Add tests/test_export.py:
  1) Setup cfg like tests/test_ui_routes.py (monkeypatch api._settings and ui_routes._settings).
  2) Seed a recording and derived artifacts:
     - summary.json with topic, summary_bullets, decisions, action_items
     - speaker_turns.json with 2 turns
  3) Call GET /ui/recordings/<id>/export.zip and assert 200.
  4) Open zip from response bytes and assert:
     - onenote.md exists and contains the topic
     - manifest.json exists
     - summary.json exists (when seeded)

Local verification:
- scripts/ci.sh
- Manual:
  - Open a Ready recording and verify Export section shows markdown, Copy works, and ZIP downloads.

Success criteria:
- Export markdown is generated for any recording (even partially processed).
- ZIP download works and contains onenote.md and available artifacts.
- UI shows Export section with preview, copy, and download.
- scripts/ci.sh is green.
```
