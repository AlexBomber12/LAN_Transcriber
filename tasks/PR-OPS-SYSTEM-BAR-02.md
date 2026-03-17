Run PLANNED PR

PR_ID: PR-OPS-SYSTEM-BAR-02
Branch: pr-ops-system-bar-02
Title: Simplify the bottom system bar to a compact Node/GPU/LLM row and make GPU runtime status truthful

Follow AGENTS.md exactly for work mode, queue handling, CI, artifacts, MCP usage, and scope control. This is a focused BIG PR, not a MICRO PR. Keep the scope strict. Do not drift into unrelated Control Center cleanup, inspector work, worklist redesign, speaker UI, or broader Stitch restyling.

The connected Google Stitch MCP project named "Control Center" may be used only as a consistency reference for the already approved visual language: calm dark shell, soft rounded corners, restrained spacing, and compact footer presentation. Do not use Stitch to invent a new footer dashboard or a new layout for this task. Do not paste raw Stitch output into the repo. Keep the implementation Jinja and HTMX based.

This task must be executed in 3 internal phases within a single run.

Phase 1 - Inspect and map
Inspect the current bottom system-bar implementation and the data flow that feeds it. Confirm the exact files involved before coding. Expected files include:
- lan_app/system_status.py
- lan_app/ui_routes.py
- lan_app/templates/partials/control_center/system_bar.html
- lan_app/templates/base.html
- tests/test_system_status.py
- tests/test_ui_routes.py
- tests/test_cov_lan_app_ui_routes.py
- tests_playwright/test_ui_smoke_playwright.py (only if the current assertions will break)
Use this inspection to map the exact current system-bar items, markup, CSS, and GPU runtime probe logic. Do not broaden the task.

Phase 2 - Implement
Implement only the bottom system-bar simplification and the truthful GPU runtime fix described below.

Phase 3 - Verify
Run scripts/ci.sh until exit code 0, generate required review artifacts, update only the current PR status line in tasks/QUEUE.md according to AGENTS.md, and provide a concise final changelog.

Current problem
The bottom system bar still behaves like a mini dashboard. It is too heavy, partially duplicated, and not truthful enough for the operator. The operator needs only a compact runtime line that answers 3 questions:
1. Is the node / DGX / LLM endpoint up?
2. Is GPU runtime actually available?
3. Which LLM model is configured?

The current bar shows too many unrelated items and one of the critical items is misleading:
- "Inbox view" duplicates information already visible in the worklist
- "Active jobs" does not belong in the bottom system bar for this workflow
- "DGX / Spark" is too narrow a label; rename it to a more general node-level label
- "GPU runtime" currently reports "CPU only" in a case where GPU is actually available; this must be fixed truthfully, not cosmetically
- "Inference mode" is not needed in the bottom bar
- "Inference target" is useful, but its presentation should be simplified to the LLM model only
- the current system bar still carries dashboard-card structure and note text instead of a compact runtime row
- the current sticky behavior or scrolling movement must be removed

Required product contract for the bottom system bar
The system bar must become a compact static footer row, not a sticky multi-card strip.
It must show exactly 3 compact runtime items and nothing else:
1. Node status
2. GPU runtime
3. LLM

Target compact footer content
- First item: colored status dot + "Node status" + short status value such as Online / Busy / Offline / Unknown
- Second item: "GPU runtime" + truthful short value such as GPU ready / GPU busy / GPU unavailable / CPU only
- Third item: "LLM: <configured model>" such as "LLM: gpt-oss:120b"

Target visual behavior
- compact, inline, footer-like presentation
- no dashboard card grid
- no large blocks
- no explanatory runtime note paragraph
- no sticky or floating movement with page scroll
- no extra host/IP/CUDA/debug text in the visible compact footer row unless the current task needs a tiny tooltip/title attribute for diagnostics

Exact required changes

1. Remove duplicated and unwanted system-bar items
Remove these items from the bottom system bar completely:
- Inbox view
- Active jobs
- Inference mode
Also remove the long runtime explanatory note paragraph.
Do not move these items somewhere else in this PR.

2. Keep and rename the DGX status item
Keep the current DGX / Spark runtime truth source, but rename the visible label from:
- "DGX / Spark"
to:
- "Node status"
The visible value must be simplified to a short operator-facing status:
- Online
- Busy
- Offline
- Unknown
Use the same underlying Spark/node truth where possible.
Add a colored status dot to this item. Map tones like this unless the current implementation already has a better equivalent tone mapping:
- healthy -> green dot, Online
- busy or active LLM work -> yellow dot, Busy
- offline -> red dot, Offline
- degraded or unknown -> gray dot, Unknown

3. Keep GPU runtime, but make it truthful
Keep GPU runtime in the footer because it is needed.
However, fix the current false "CPU only" result when GPU is actually available.
This must be a real runtime fix, not a hardcoded green state.

Implementation requirements for truthful GPU runtime:
- inspect the current CUDA runtime probe logic in lan_app/system_status.py and any helpers it relies on
- keep using existing runtime facts where valid
- if the current torch-only visibility check is insufficient in the real deployment, add a lightweight fallback probe using tools already available in the runtime environment, such as nvidia-smi, without introducing new Python dependencies
- do not hardcode healthy state
- do not fake green based only on config asking for cuda
- the final status must reflect actual runtime GPU visibility as truthfully as possible inside the existing deployment model

Operator-facing GPU runtime values should be short and readable, for example:
- GPU ready
- GPU busy
- GPU unavailable
- CPU only

Tone expectations:
- actual visible working GPU -> healthy/green
- visible but busy GPU -> busy/yellow
- no visible GPU when GPU is expected -> offline/red
- CPU-only runtime when GPU is not expected -> degraded/neutral depending on existing tone system

Important separation rule:
Do not conflate GPU runtime with inference mode. This PR removes "Inference mode" from the footer. GPU runtime should answer only whether the runtime sees and can use GPU resources, not whether a specific active stage is currently on a GPU or CPU path.

4. Keep inference target, but simplify it to LLM only
Keep the configured LLM target because the operator needs it.
However, simplify the visible label/presentation.
Do not render it as a full dashboard card called "Inference target".
Render it in compact footer form as:
- "LLM: gpt-oss:120b"
or the configured model value from settings/runtime.
Do not display host/IP/advertised-by-Spark details in the visible compact footer row.

5. Remove sticky movement
The system bar must stop moving with scroll.
Remove sticky/floating behavior from the Control Center system bar.
Render it as a normal static footer section.
Make sure the final result does not visually fight with the page when the user scrolls.

6. Replace the current card-grid markup with compact footer-row markup
Refactor the footer template and context shape as needed so the bottom bar renders as one compact runtime row instead of grouped large cards.
Use the smallest maintainable Jinja structure that supports the 3 required items.
Preserve HTMX refresh if it is already used and still appropriate for the new compact footer row.
If the same shared footer partial is used on the full-page recording inspector, keep the simplified runtime row consistent there as well.

Hard constraints
- Do not change the top navigation
- Do not change the upload area
- Do not change the recordings worklist
- Do not change the compact inspector
- Do not change speaker UI
- Do not redesign the whole Control Center
- Do not move runtime diagnostics into another part of the page in this PR
- Do not add new helper text
- Do not add extra host/IP/debug paragraphs to the visible compact footer
- Do not add new dependencies
- Do not degrade or remove current HTMX refresh behavior unless needed to support the simplified footer

Verification requirements
- run scripts/ci.sh until exit code 0
- generate required review artifacts per AGENTS.md
- update tests that assert the old footer structure or old labels
- add or update tests for truthful GPU runtime rendering in the compact footer context
- make sure route rendering tests and system-status tests reflect the new compact contract
- keep coverage green

Expected test areas to update
At minimum inspect and update as needed:
- tests/test_system_status.py
- tests/test_ui_routes.py
- tests/test_cov_lan_app_ui_routes.py
- tests_playwright/test_ui_smoke_playwright.py only if current UI text expectations break

Success criteria
- the bottom system bar no longer shows Inbox view
- the bottom system bar no longer shows Active jobs
- the bottom system bar no longer shows Inference mode
- the long runtime note paragraph is gone
- the old multi-card dashboard-like footer presentation is gone
- the visible DGX / Spark label is replaced by Node status
- Node status shows a colored status dot and a short value like Online / Busy / Offline / Unknown
- GPU runtime remains visible in the footer
- GPU runtime is truthful and no longer falsely shows CPU only when GPU is actually visible/available in the runtime
- the footer still shows the configured LLM model in compact form as LLM: <model>
- the footer no longer moves with scroll
- no unrelated parts of the Control Center changed
- CI passes

Final output requirements
Provide a concise changelog with these sections only:
- Removed
- Renamed
- Runtime truth fix
- Template/CSS changes
- Tests updated
Do not include unrelated cleanup.
