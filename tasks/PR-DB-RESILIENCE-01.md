PR-DB-RESILIENCE-01

```text
Role: You are Codex Agent working inside the LAN_Transcriber repository.

Branch: pr/db-resilience-01
PR title: PR-DB-RESILIENCE-01 SQLite resilience and migration hygiene
Base branch: main

Goal:
Improve reliability of SQLite under concurrent API + worker access and make migrations easier to maintain.
1) Add explicit busy timeout and retry-on-locked for write operations.
2) Make migrations file-based while preserving PRAGMA user_version behavior.
3) Improve connection management safely without rewriting the whole DB layer.

A) Connection initialization
- File: lan_app/db.py
- Centralize sqlite3.connect creation in 1 helper:
  - connect_db(path) -> sqlite3.Connection
- Ensure PRAGMAs:
  - journal_mode=WAL
  - synchronous=NORMAL
  - foreign_keys=ON
  - busy_timeout from LAN_SQLITE_BUSY_TIMEOUT_MS (default 30000)
- Keep timeout parameter too.

B) Retry on lock and busy
- Implement with_db_retry(fn, retries=5, base_sleep_ms=50).
- Wrap write-heavy operations (create_job, start_job, finish_job, fail_job, set_recording_status, migrations apply).
- Retry only for sqlite3.OperationalError lock or busy messages.
- On final failure raise the original error.

C) File-based migrations
- Create folder: lan_app/migrations/
- Move SQL from inline tuples into numbered files:
  - 001_init.sql
  - 002_add_jobs.sql
  - continue for all existing versions
- Migration runner:
  - Read user_version.
  - Apply files with version > user_version in order.
  - After each file, set user_version to that migration number.
- Schema must remain unchanged.

D) Optional minimal connection reuse
- Do not add a full pool.
- If helpful, use threading.local to cache a connection per thread.
- If caching is implemented, add close_thread_connections() and call it at app shutdown.

E) Tests
- Migrations apply from files on a fresh DB.
- Retry helper retries on a forced lock using 2 connections and BEGIN EXCLUSIVE.
- Keep existing tests green.

Local verification:
- scripts/ci.sh

Success criteria:
- busy_timeout and WAL are reliably set.
- Transient locks are retried.
- Migrations are stored in files and still versioned via user_version.
- scripts/ci.sh is green.
```
