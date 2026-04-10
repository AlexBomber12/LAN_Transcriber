Run PLANNED PR

PR_ID: PR-MERGE-DECAPITALIZE-01
Branch: pr-merge-decapitalize-01
Title: Decapitalize false sentence starts left over from segment merging

Follow AGENTS.md exactly for work mode, queue handling, CI, artifacts, MCP usage, and scope control. This is a focused MICRO PR. Keep the scope strict.

This task must be executed in 3 internal phases within a single run.

Phase 1 - Inspect and map
Read and confirm the current state of these files before coding:
- lan_transcriber/pipeline_steps/speaker_turns.py (build_speaker_turns, merge_short_turns or wherever segment texts are concatenated)
- lan_transcriber/pipeline_steps/diarization_quality.py (smooth_speaker_turns, where adjacent turns are merged and text is concatenated)
- tests/test_pipeline_steps_speaker_turns.py
- tests/test_diarization_quality.py

Identify every code path where two segment/turn texts are concatenated with a space. There should be at least two: one in build_speaker_turns (word-level merging) and one in smooth_speaker_turns or merge_short_turns (turn-level merging).

Phase 2 - Implement
Implement exactly these changes. Do not add anything beyond these fixes.

CHANGE 1: Add a decapitalization helper
In lan_transcriber/pipeline_steps/speaker_turns.py (or a shared location if more appropriate), add:

def _decapitalize_join(existing_text: str, appended_text: str) -> str:
    """Join two text fragments, lowercasing the appended fragment's first letter
    when it looks like a false sentence start from segment splitting."""
    if not existing_text or not appended_text:
        return (existing_text + " " + appended_text).strip()
    last_char = existing_text.rstrip()[-1] if existing_text.strip() else ""
    if last_char in ".!?":
        return f"{existing_text} {appended_text}"
    first_word = appended_text.split()[0] if appended_text.split() else ""
    if not first_word or not first_word[0].isupper():
        return f"{existing_text} {appended_text}"
    if first_word.isupper() and len(first_word) > 1:
        return f"{existing_text} {appended_text}"
    if first_word == "I":
        return f"{existing_text} {appended_text}"
    fixed = appended_text[0].lower() + appended_text[1:]
    return f"{existing_text} {fixed}"

Rules encoded:
- If existing_text ends with sentence-ending punctuation (. ! ?), keep capitalization (it is a real new sentence).
- If the first word of appended_text is fully uppercase and longer than 1 char (e.g. "ISO", "NASA", "API"), keep it (likely an acronym).
- If the first word is "I" (English pronoun), keep it.
- Otherwise, lowercase the first character of appended_text.

CHANGE 2: Use _decapitalize_join in all merge points
Replace every instance of text concatenation like:
  current["text"] = f"{current['text']} {word['word']}".strip()
or:
  merged_text = f"{prev_text} {next_text}"
with a call to _decapitalize_join(current["text"], word["word"]) or equivalent.

Apply this in:
- build_speaker_turns (word-level merge loop)
- merge_short_turns (if it exists, turn-level merge)
- smooth_speaker_turns in diarization_quality.py (adjacent turn merge)

Do NOT apply this for the very first word/segment of a turn (the turn start should remain capitalized).

Phase 3 - Test and verify
Add tests:
- test_decapitalize_mid_sentence: "энергии. Он может" + "Понизить" -> "энергии. Он может понизить" (no sentence-ending punct before "Понизить")
- test_keep_after_period: "решение." + "Но мы понимаем" -> "решение. Но мы понимаем" (period keeps capital)
- test_keep_acronym: "стандарту" + "ISO" -> "стандарту ISO" (all-caps word kept)
- test_keep_english_I: "think" + "I will" -> "think I will" (English "I" kept)
- test_keep_already_lowercase: "просто" + "уходить" -> "просто уходить" (no change needed)
- test_empty_strings: "" + "Hello" -> "Hello", "text" + "" -> "text"
- test_real_example: simulate the compressor paragraph from the Plaud comparison and verify no false capitals remain after merge.

Run full CI. All existing tests must pass.

Success criteria:
- The merged transcript text no longer has false capital letters at former segment boundaries.
- Real sentence starts (after . ! ?) remain capitalized.
- Acronyms (ISO, API, etc.) remain uppercase.
- No existing tests are broken.
