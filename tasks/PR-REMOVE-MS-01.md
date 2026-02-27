PR-REMOVE-MS-01

```text
Role: You are Codex Agent working inside the LAN_Transcriber repository.

Branch: pr/remove-ms-01
PR title: PR-REMOVE-MS-01 Remove Microsoft Graph, calendar matching UI, OneNote publish, and msal dependency
Base branch: main

Goal:
1) Remove all Microsoft Graph functionality from the runtime:
   - Device code auth
   - Calendar matching
   - OneNote publish
2) Keep the app fully functional using the export-only workflow (PR-EXPORT-01).
3) Remove msal from dependencies and remove all MS-related tests so CI stays green.

A) Remove runtime settings and deps
- In lan_app/config.py AppSettings:
  - Remove fields:
    - ms_tenant_id
    - ms_client_id
    - ms_scopes
    - msal_cache_path
    - calendar_match_window_minutes
    - calendar_auto_match_threshold
  - Keep routing_auto_select_threshold.
- Remove msal from:
  - requirements.in
  - requirements.txt
  - ci-requirements.txt
  - pyproject.toml

B) Remove MS modules
- Delete files:
  - lan_app/ms_graph.py
  - lan_app/calendar.py
  - lan_app/onenote.py
- Ensure no remaining imports reference these modules.

C) API cleanup
- In lan_app/api.py:
  - Remove all /api/connections/ms/* endpoints and related imports.
  - Remove calendar endpoints:
    - GET /api/recordings/{id}/calendar
    - POST /api/recordings/{id}/calendar/select
  - Remove publish endpoint:
    - POST /api/recordings/{id}/publish
  - Remove now-unused imports.

D) UI cleanup
1) Connections page
- In lan_app/ui_routes.py:
  - Update /connections to no longer call ms_connection_state.
  - Pass only gdrive state (for now).
- In lan_app/templates/connections.html:
  - Remove the Microsoft Graph card and all related JS.

2) Recording detail
- In lan_app/ui_routes.py:
  - Remove "calendar" from tabs list.
  - Remove calendar context loading blocks.
- In lan_app/templates/recording_detail.html:
  - Remove the entire calendar tab block.
  - Remove the OneNote Publish section in overview.
  - Keep Export section as the replacement.

3) Projects page
- In lan_app/ui_routes.py:
  - Simplify /projects: remove browse_notebook_id and browse UI logic.
  - Remove POST /projects/{project_id}/onenote route.
- In lan_app/templates/projects.html:
  - Remove OneNote mapping UI.
  - Keep minimal project CRUD: list projects, create, delete.

E) Tests
- Remove MS-related tests:
  - tests/test_ms_auth.py
  - tests/test_calendar.py
  - tests/test_onenote.py
- Update tests/test_ui_routes.py:
  - Remove test_recording_detail_calendar_tab.
  - Update any assertions that mention OneNote publish UI.
  - Ensure /projects tests do not expect OneNote mapping UI.
- Update tests/test_imports.py if it imports MS modules.

Local verification:
- scripts/ci.sh

Success criteria:
- The server starts with no MS_* env variables and no msal dependency installed.
- /connections renders without Microsoft Graph sections.
- Recording detail has no calendar tab and no OneNote publish actions.
- Export-only flow works: markdown preview and ZIP download are available.
- scripts/ci.sh is green.
```
