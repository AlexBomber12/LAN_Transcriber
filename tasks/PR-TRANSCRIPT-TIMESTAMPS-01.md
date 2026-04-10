Run PLANNED PR

PR_ID: PR-TRANSCRIPT-TIMESTAMPS-01
Branch: pr-transcript-timestamps-01
Title: Add timestamps to transcript export (markdown and JSON) as default behavior

Follow AGENTS.md exactly for work mode, queue handling, CI, artifacts, MCP usage, and scope control. This is a focused BIG PR, not a MICRO PR. Keep the scope strict.

This task must be executed in 3 internal phases within a single run.

Phase 1 - Inspect and map
Read and confirm the current state of these files before coding:
- lan_app/exporter.py (_transcript_section, build_onenote_markdown)
- lan_app/templates/ (any template that renders transcript turns, search for "speaker" and "text" in partials)
- lan_app/ui_routes.py (any route that renders transcript for the UI)
- tests/test_export.py (existing export tests)
- tests/test_cov_lan_app_export_ops_jobs.py (existing coverage tests)

Phase 2 - Implement
Implement exactly these changes. Do not add anything beyond these fixes.

CHANGE 1: Add timestamp formatting helper
In lan_app/exporter.py, add a helper function:

def _format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

CHANGE 2: Update _transcript_section to include timestamps
In lan_app/exporter.py _transcript_section, change the turn rendering line from:
  lines.append(f"- **{speaker}:** {text}")
to:
  start = safe_float(turn.get("start"), default=None)
  timestamp = f"{_format_timestamp(start)} " if start is not None else ""
  lines.append(f"- **{timestamp}{speaker}:** {text}")

Import safe_float from lan_transcriber.utils at the top of exporter.py if not already imported.

CHANGE 3: Update transcript UI rendering
Find the template(s) that render speaker turns in the recording detail / inspector view. Add the timestamp before the speaker label in the same MM:SS or HH:MM:SS format. Look for the Jinja loop that iterates over speaker_turns and renders speaker + text. Add:
  {{ "%02d:%02d"|format(turn.start // 60, turn.start % 60) }}
before the speaker name. If turn.start is not available in the template context, ensure the route passes it through.

CHANGE 4: Update export ZIP transcript.json
No change needed here since speaker_turns.json already contains start/end fields. Confirm this in the inspect phase and skip if already correct.

Phase 3 - Test and verify
- Update existing export tests to expect timestamps in the markdown output:
  - test that a turn with start=65.0 renders as "01:05" in the markdown.
  - test that a turn with start=3661.0 renders as "01:01:01" in the markdown.
  - test that a turn with no start field renders without timestamp prefix.
- Add a new test for _format_timestamp:
  - 0 -> "00:00"
  - 65 -> "01:05"
  - 3661 -> "01:01:01"
  - negative -> "00:00"
- Run full CI. All existing tests must pass.

Success criteria:
- Every turn in the markdown export has a timestamp prefix by default (no config flag needed, timestamps are always on).
- The UI transcript view shows timestamps next to each speaker turn.
- Format is MM:SS for recordings under 1 hour, HH:MM:SS for recordings over 1 hour.
- speaker_turns.json already has start/end, so no changes to the JSON artifact.
- No existing tests are broken.
- New tests cover the timestamp formatting logic.
