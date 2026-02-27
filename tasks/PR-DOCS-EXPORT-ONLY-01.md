PR-DOCS-EXPORT-ONLY-01

```text
Role: You are Codex Agent working inside the LAN_Transcriber repository.

Branch: pr/docs-export-only-01
PR title: PR-DOCS-EXPORT-ONLY-01 Update docs and env templates for upload + export-only mode
Base branch: main

Goal:
1) Update documentation to reflect the new workflow:
   - Upload in UI
   - Per-file progress
   - Export markdown and ZIP for manual OneNote paste
2) Remove obsolete instructions related to Microsoft Graph and Google Drive ingest.
3) Document reverse proxy settings required for large uploads.

A) README updates
- Update README.md:
  - New flow: Upload -> Processing -> Export (Copy markdown / Download ZIP)
  - Remove sections about Connections, Google Drive, Microsoft Graph, OneNote publish.
  - Add a short "LAN deployment notes" section:
    - persistent /data volume
    - recommended disk sizing

B) Runbook updates
- Update docs/runbook.md (or the current ops doc under docs/):
  - Add "Uploads" section:
    - max upload size considerations
    - mention UPLOAD_MAX_BYTES (if implemented)
  - Add "Nginx" section for large files:
    - client_max_body_size
    - proxy_read_timeout
    - proxy_send_timeout
  - Remove Graph and Drive sections.

C) Env template updates
- Update .env.example:
  - Remove MS_* and GDRIVE_* variables.
  - Add UPLOAD_MAX_BYTES example (if implemented).
  - Keep core vars (LAN_ENV, LAN_DATA_ROOT, LAN_REDIS_URL, LLM_BASE_URL, etc).

D) Compose notes
- Update docker-compose.yml comments (if any) to match export-only mode:
  - remove references to Graph and Drive config
  - ensure /data volume is clearly described

E) Verification
- Ensure docs match current code paths:
  - /upload
  - /recordings/<id>
  - /ui/recordings/<id>/export.zip

Local verification:
- scripts/ci.sh

Success criteria:
- README and runbook describe the actual upload and export-only workflow.
- .env.example no longer mentions Graph or Drive and includes upload-related vars when applicable.
- scripts/ci.sh is green.
```
