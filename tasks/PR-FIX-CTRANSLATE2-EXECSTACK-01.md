PR-FIX-CTRANSLATE2-EXECSTACK-01

```text
Role: You are Codex Agent working inside the LAN_Transcriber repository.

Branch: pr/fix-ctranslate2-execstack-01
PR title: PR-FIX-CTRANSLATE2-EXECSTACK-01 Fix ctranslate2 executable-stack loader failure (no rebuild needed)
Base branch: main

Problem
Transcription fails with an error similar to:
  libctranslate2-*.so.*: cannot enable executable stack as shared object requires: Invalid argument.
This indicates the shared library is marked as requiring an executable stack (PT_GNU_STACK with PF_X), and the loader cannot enable it.

Goals
1) Prevent the failure by clearing the executable-stack requirement on libctranslate2 shared objects before whisperx imports faster-whisper/ctranslate2.
2) Implement the fix in pure Python so it works without installing patchelf/execstack.
3) Add unit tests for the ELF patcher using a synthetic ELF file.

Non-goals
- Do not change model behavior or ASR output.
- Do not pin or downgrade ctranslate2 versions in this PR.

Implementation

A) Add a pure-Python ELF patcher
Create: lan_transcriber/native_fixups.py
- Provide functions:
  1) clear_execstack_flag(path: Path) -> bool
     - Open file in r+b mode.
     - Validate ELF magic.
     - Support ELF64 little-endian and ELF32 little-endian.
     - Locate the PT_GNU_STACK program header (p_type == 0x6474E551).
     - If p_flags has PF_X bit set, clear PF_X and write flags back in-place.
     - Return True when a change was made, else False.

  2) find_libctranslate2_candidates() -> list[Path]
     - Use site.getsitepackages() and glob for patterns:
       - **/libctranslate2*.so*
     - Return sorted unique paths.

  3) ensure_ctranslate2_no_execstack() -> list[str]
     - Patch candidates once per process (module-level guard).
     - Return a list of patched file paths as strings.
     - Never raise on patch failures; log and continue.

B) Call the fixup before importing whisperx
File: lan_transcriber/pipeline_steps/orchestrator.py
- In the ASR path (in the new _whisperx_asr helper introduced by PR-FIX-WHISPERX-API-01):
  - Call ensure_ctranslate2_no_execstack() before "import whisperx".
  - If any files were patched, append a short step log line.
  - This must run before whisperx import so the loader sees the patched library.

C) Unit tests
Add: tests/test_native_fixups_execstack.py
- Test 1: patches a synthetic ELF64 file
  - Create a minimal ELF64 little-endian file with 1 program header PT_GNU_STACK.
  - Set p_flags to PF_R|PF_W|PF_X (7).
  - Call clear_execstack_flag(path).
  - Assert it returns True.
  - Re-read p_flags and assert PF_X is cleared (flags == 6).

- Test 2: no-op when PF_X is already cleared
  - p_flags == 6 -> returns False.

- Test 3: no-op on non-ELF
  - file with random bytes -> returns False.

D) Manual verification
- Run a single transcription after merge.
- The ctranslate2 loader error must disappear.

Local verification
- scripts/ci.sh

Success criteria
- The application no longer fails with the libctranslate2 executable stack error.
- Unit tests for the patcher pass.
- scripts/ci.sh is green.
```
