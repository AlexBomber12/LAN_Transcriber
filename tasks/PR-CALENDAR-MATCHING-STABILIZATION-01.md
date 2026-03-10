PR-CALENDAR-MATCHING-STABILIZATION-01
======================================

Branch: pr-calendar-matching-stabilization-01

Prompt (copy as-is into Codex)
------------------------------

```text
You are Codex Agent working on the LAN-Transcriber repository.

Goal
Stabilize automatic calendar matching now that ICS import, attendee parsing, glossary seeding, and capture-time correction exist. The current lan_app/calendar/matching.py logic already scores candidates by start proximity, presence, overlap, duration similarity, and token overlap, but it was designed before corrected upload capture-time semantics and still has rough edges around ambiguous candidates, suspicious timestamps, and operator feedback. Improve matching reliability and diagnostics without redesigning the whole calendar subsystem.

Constraints
- This PR assumes PR-CAPTURE-TIME-SEMANTICS-01 is already merged. Matching must rely on corrected captured_at values.
- Do not rebuild ICS import/storage from scratch; that was already delivered.
- Keep manual override UI intact and improve it only where it helps stability/clarity.
- Keep the implementation production-safe and deterministic.
- Maintain 100% statement and branch coverage for every changed/new module.

Implementation requirements

1) Make matching explicitly aware of capture-time provenance
- Use the new capture-time provenance fields introduced in the time-semantics PR.
- If a recording has suspicious or low-confidence capture-time provenance, do not silently behave as if the timestamp were unquestionably correct.
- Add a small warning path or scoring penalty when matching relies on weak timing data.
- For recordings with corrected inferred local timestamps, matching should behave normally.

2) Refine candidate scoring using the now-correct time model
- Revisit lan_app/calendar/matching.py scoring weights and edge-case logic so matching benefits from correct capture-time semantics.
- Preserve the existing ingredients but improve the stability of ranking, especially for:
  - recordings captured slightly before the meeting starts
  - recordings that start inside the meeting window
  - long recordings that overlap much of the event but do not start exactly at the event start
  - ambiguous adjacent meetings on the same day
- Make the scoring math explicit and testable.

3) Strengthen overlap and duration reasoning
- Improve how overlap_score and duration_score are computed/used for long recordings.
- Avoid over-privileging pure start-time proximity when overlap evidence is stronger.
- Ensure durationless recordings still behave sensibly.

4) Improve textual hint usage without overfitting
- Keep token overlap from filename/title/location/attendees, but make it more robust and interpretable.
- Consider light improvements such as:
  - better stopword handling
  - safer token normalization
  - bounded attendee/title boosts
- Do not introduce brittle fuzzy matching magic or heavy dependencies.

5) Make auto-selection safer and more explainable
- Revisit _auto_selection(...) thresholds/margins so automatic selection is conservative but useful.
- Preserve manual override as the fallback for ambiguous cases.
- Store or expose enough rationale so the operator can see why event A beat event B.
- Keep candidates_json and selected_event_id behavior compatible with the rest of the app.

6) Surface suspicious-timestamp or weak-match diagnostics in the UI
- On the recording detail calendar tab, expose clear warnings when:
  - capture time is weak/suspicious
  - no candidate is strong enough for auto-select
  - multiple candidates are close together
- Reuse the improved observability/root-cause style where appropriate, but keep this PR focused on calendar matching.

7) Preserve downstream context behavior
- The selected event must continue feeding summary context and glossary context correctly through existing downstream helpers.
- Do not regress attendee parsing, calendar_summary_context(...), or manual selection routes.

8) Tests with full coverage
Add deterministic offline tests for at least these cases:
- corrected upload capture time now produces the expected best calendar candidate
- a recording starting slightly before an event but overlapping most of it scores well
- ambiguous adjacent meetings do not auto-select when the confidence margin is too small
- strong overlap can outrank a slightly closer but weaker candidate when appropriate
- suspicious/weak timestamp provenance triggers a warning or conservative behavior
- manual override still works and is preserved
- rationale/candidate payload remains well-formed and UI-safe

9) Documentation and operator notes
- Update README/runbook briefly to explain that matching now relies on corrected upload capture times and is intentionally conservative when timestamps are weak or ambiguous.
- Mention where operators can manually override the selected event.

Verification steps (must be included in PR description)
- Confirm the example upload timestamp case that was previously off by 1 hour now matches the correct event after the time-semantics fix.
- Confirm ambiguous same-day meetings remain candidates but are not auto-selected without sufficient margin.
- Confirm the calendar tab shows clear rationale/warnings for weak or ambiguous cases.
- Run scripts/ci.sh and keep CI green.

Deliverables
- More stable calendar candidate scoring/ranking
- Safer auto-selection and clearer rationale/warnings
- Preserved manual override and downstream summary/glossary integration
- Updated tests and docs

Success criteria
- Calendar matching becomes more reliable after the capture-time fix.
- Weak or ambiguous cases are surfaced clearly instead of being silently mis-selected.
- Manual override remains straightforward.
- CI remains green.
```
