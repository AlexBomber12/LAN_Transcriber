PR-REMOVE-GDRIVE-01

```text
Role: You are Codex Agent working inside the LAN_Transcriber repository.

Branch: pr/remove-gdrive-01
PR title: PR-REMOVE-GDRIVE-01 Remove Google Drive ingest, Connections page, ingest lock, and Google API deps
Base branch: main

Goal:
1) Remove all Google Drive ingest functionality and its dependencies.
2) Replace the old Connections-driven ingest flow with the Upload page as the only ingestion path.
3) Remove ingest lock code and tests that are only used for Drive ingest.
4) Keep CI green.

A) Remove Google modules and endpoints
- Delete lan_app/gdrive.py.
- In lan_app/api.py:
  - Remove POST /api/actions/ingest and related imports:
    - try_acquire_ingest_lock
    - release_ingest_lock
- Delete lan_app/locks.py (only used for ingest lock).
- In lan_app/config.py AppSettings:
  - Remove Google Drive settings:
    - gdrive_sa_json_path
    - gdrive_inbox_folder_id
    - gdrive_poll_interval_seconds
  - Remove ingest lock setting:
    - ingest_lock_ttl_seconds

B) UI cleanup
- Remove Connections page entirely:
  - Delete /connections route in lan_app/ui_routes.py
  - Delete template lan_app/templates/connections.html
  - Remove nav entry from lan_app/templates/base.html
- Ensure Upload page is the primary path in navigation.

C) Dependencies
- Remove Google API deps from:
  - requirements.txt:
    - google-auth
    - google-api-python-client
  - ci-requirements.txt:
    - google-auth
    - google-api-python-client
  - pyproject.toml:
    - google-auth
    - google-api-python-client

D) Tests
- Remove Google Drive and lock tests:
  - tests/test_gdrive.py
  - tests/test_locks.py
- Update any remaining tests importing gdrive or locks.
- Update tests/test_ui_routes.py if it references /connections.

Local verification:
- scripts/ci.sh

Success criteria:
- No Google Drive code remains in lan_app.
- No Connections page in UI, and base nav does not link to it.
- requirements and CI no longer include Google API libraries.
- scripts/ci.sh is green.
```
