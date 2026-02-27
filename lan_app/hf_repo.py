from __future__ import annotations


def split_repo_id_and_revision(value: str) -> tuple[str, str | None]:
    normalized = value.strip()
    if "@" not in normalized:
        return normalized, None
    repo_id, revision = normalized.split("@", 1)
    repo_id = repo_id.strip()
    revision = revision.strip()
    return repo_id, revision or None
