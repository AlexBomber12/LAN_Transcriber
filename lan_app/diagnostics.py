from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any, Sequence

from lan_transcriber.gpu_policy import is_gpu_oom_error
from lan_transcriber.llm_client import LLMEmptyContentError, LLMTruncatedResponseError

_LLM_CHUNK_FAILURE_RE = re.compile(
    r"^LLM chunk (?P<chunk_index>[^/]+)/(?P<chunk_total>\d+) failed "
    r"\[(?P<code>[^\]]+)\]: (?P<detail>.+)$"
)
_LLM_MERGE_FAILURE_RE = re.compile(
    r"^LLM merge failed \[(?P<code>[^\]]+)\]: (?P<detail>.+)$"
)
_LLM_PROGRESS_STAGE_RE = re.compile(r"^llm_chunk_(?P<chunk_index>[^_]+)_of_(?P<chunk_total>\d+)$")
_WRAPPER_ONLY_CODES = {"job_retry_limit_reached"}
_GENERIC_CODES = {"processing_runtime_error"}


def _clean_text(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _int_or_none(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_timestamp(value: object) -> datetime | None:
    text = _clean_text(value)
    if text is None:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _elapsed_seconds(
    *,
    started_at: object,
    duration_ms: object,
    now: datetime,
    is_running: bool,
) -> float | None:
    duration_value = _int_or_none(duration_ms)
    if duration_value is not None and duration_value >= 0:
        return round(duration_value / 1000.0, 3)
    if not is_running:
        return None
    started = _parse_timestamp(started_at)
    if started is None:
        return None
    delta = (now - started).total_seconds()
    return round(max(delta, 0.0), 3)


def _metadata_payload(row: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(row, dict):
        return {}
    payload = row.get("metadata_json")
    return dict(payload) if isinstance(payload, dict) else {}


def _sort_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("updated_at") or ""),
        str(row.get("finished_at") or ""),
        str(row.get("started_at") or ""),
    )


def _progress_stage_chunk_position(stage_code: object) -> tuple[str | None, int | None]:
    text = _clean_text(stage_code)
    if text is None:
        return None, None
    match = _LLM_PROGRESS_STAGE_RE.fullmatch(text)
    if match is None:
        return None, None
    return match.group("chunk_index"), _int_or_none(match.group("chunk_total"))


def stage_name_for_progress(stage_code: object) -> str | None:
    text = _clean_text(stage_code)
    if text is None:
        return None
    if text.startswith("llm_chunk_") or text in {"llm", "llm_extract", "llm_merge"}:
        return "llm_extract"
    return text


def _cause_specificity(code: str | None) -> int:
    if code in _WRAPPER_ONLY_CODES:
        return 0
    if code in _GENERIC_CODES:
        return 1
    if code:
        return 2
    return -1


def _humanize_unknown_code(code: str | None) -> str:
    if code is None:
        return "Processing issue"
    return code.replace("_", " ").strip().capitalize() or "Processing issue"


def root_cause_summary(
    code: str | None,
    *,
    explicit_text: str | None = None,
    chunk_index: str | None = None,
    chunk_total: int | None = None,
) -> str:
    if code == "gpu_oom":
        return "The worker ran out of GPU memory while loading or running a heavy model."
    if code == "llm_chunk_timeout":
        if chunk_index and chunk_total:
            return f"LLM chunk {chunk_index}/{chunk_total} timed out."
        return "An LLM chunk timed out."
    if code == "llm_chunk_request_timeout":
        if chunk_index and chunk_total:
            return f"LLM chunk {chunk_index}/{chunk_total} timed out waiting on the model service."
        return "An LLM chunk request timed out waiting on the model service."
    if code == "llm_chunk_parse_error":
        if chunk_index and chunk_total:
            return f"LLM chunk {chunk_index}/{chunk_total} returned an invalid response."
        return "An LLM chunk returned an invalid response."
    if code == "llm_chunk_connection_error":
        if chunk_index and chunk_total:
            return f"LLM chunk {chunk_index}/{chunk_total} lost the model connection."
        return "An LLM chunk lost the model connection."
    if code == "llm_chunk_runtime_error":
        if chunk_index and chunk_total:
            return f"LLM chunk {chunk_index}/{chunk_total} failed during processing."
        return "An LLM chunk failed during processing."
    if code == "llm_merge_timeout":
        return "The final LLM merge step timed out."
    if code == "llm_merge_request_timeout":
        return "The final LLM merge request timed out waiting on the model service."
    if code == "llm_merge_connection_error":
        return "The final LLM merge step lost the model connection."
    if code in {"llm_merge_parse_error", "llm_merge_error"}:
        return "The final LLM merge response could not be parsed."
    if code == "cancelled_by_user":
        return "Processing was cancelled by the user."
    if code == "force_stopped_by_user":
        return "Processing had to be force-stopped after the grace timeout."
    if code == "calendar_time_mismatch":
        return "The recording time does not match the selected calendar event closely enough."
    if code == "suspicious_capture_time":
        return "The recording capture time looks suspicious and should be checked."
    if code == "llm_truncated":
        return "The LLM output was truncated repeatedly."
    if code == "llm_empty_content":
        return "The LLM returned empty content repeatedly."
    if code == "processing_runtime_error":
        return explicit_text or "Processing failed with a runtime error."
    if code == "job_retry_limit_reached":
        return explicit_text or "Processing hit the retry limit after repeated failures."
    if explicit_text:
        return explicit_text
    return _humanize_unknown_code(code)


def _root_cause_payload(
    *,
    code: str | None,
    explicit_text: str | None = None,
    detail_text: str | None = None,
    chunk_index: str | None = None,
    chunk_total: int | None = None,
    source: str,
) -> dict[str, Any]:
    summary = root_cause_summary(
        code,
        explicit_text=explicit_text,
        chunk_index=chunk_index,
        chunk_total=chunk_total,
    )
    detail = detail_text or explicit_text
    if detail == summary:
        detail = None
    return {
        "code": code,
        "text": summary,
        "detail": detail,
        "chunk_index": chunk_index,
        "chunk_total": chunk_total,
        "source": source,
    }


def root_cause_from_exception(exc: Exception) -> dict[str, Any]:
    message = _clean_text(str(exc))
    lowered = (message or "").lower()
    if is_gpu_oom_error(exc):
        return _root_cause_payload(
            code="gpu_oom",
            explicit_text=message,
            detail_text=message,
            source="exception",
        )
    if isinstance(exc, LLMTruncatedResponseError) or "finish_reason=length" in lowered:
        return _root_cause_payload(
            code="llm_truncated",
            explicit_text="The LLM output was truncated repeatedly; manual review required.",
            detail_text=message,
            source="exception",
        )
    if isinstance(exc, LLMEmptyContentError) or "empty message.content" in lowered:
        return _root_cause_payload(
            code="llm_empty_content",
            explicit_text="The LLM returned empty content repeatedly; manual review required.",
            detail_text=message,
            source="exception",
        )
    if message:
        match = _LLM_CHUNK_FAILURE_RE.fullmatch(message)
        if match is not None:
            return _root_cause_payload(
                code=match.group("code"),
                detail_text=match.group("detail"),
                chunk_index=match.group("chunk_index"),
                chunk_total=_int_or_none(match.group("chunk_total")),
                source="exception",
            )
        match = _LLM_MERGE_FAILURE_RE.fullmatch(message)
        if match is not None:
            return _root_cause_payload(
                code=match.group("code"),
                detail_text=match.group("detail"),
                source="exception",
            )
    if message == "max attempts exceeded":
        return _root_cause_payload(
            code="job_retry_limit_reached",
            explicit_text="Processing hit the retry limit; manual review required.",
            detail_text=message,
            source="exception",
        )
    return _root_cause_payload(
        code="processing_runtime_error",
        explicit_text=(
            f"Processing failed with {type(exc).__name__}: {message}"
            if message
            else f"Processing failed with {type(exc).__name__}."
        ),
        detail_text=message,
        source="exception",
    )


def root_cause_from_stage_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    metadata = _metadata_payload(row)
    status = _clean_text(row.get("status")) or ""
    if status == "cancelled" or metadata.get("cancelled_by_user"):
        code = _clean_text(metadata.get("root_cause_code"))
        if code is None:
            code = "force_stopped_by_user" if metadata.get("force_stopped") else "cancelled_by_user"
        return _root_cause_payload(
            code=code,
            explicit_text=_clean_text(metadata.get("root_cause_text"))
            or _clean_text(metadata.get("cancel_reason_text")),
            detail_text=_clean_text(metadata.get("root_cause_detail"))
            or _clean_text(metadata.get("cancel_reason_text")),
            chunk_index=_clean_text(metadata.get("cancel_chunk_index"))
            or _clean_text(metadata.get("chunk_index")),
            chunk_total=_int_or_none(metadata.get("cancel_chunk_total"))
            or _int_or_none(metadata.get("chunk_total")),
            source="stage",
        )
    code = _clean_text(metadata.get("root_cause_code"))
    text = _clean_text(metadata.get("root_cause_text"))
    detail = _clean_text(metadata.get("root_cause_detail")) or _clean_text(row.get("error_text"))
    chunk_index = _clean_text(metadata.get("chunk_index"))
    chunk_total = _int_or_none(metadata.get("chunk_total"))
    if code is None and detail:
        parsed = root_cause_from_message(detail, source="stage")
        code = parsed.get("code")
        text = parsed.get("text")
        detail = parsed.get("detail")
        chunk_index = parsed.get("chunk_index") or chunk_index
        chunk_total = parsed.get("chunk_total") or chunk_total
    if code is None and status != "failed":
        return None
    return _root_cause_payload(
        code=code or _clean_text(row.get("error_code")) or "processing_runtime_error",
        explicit_text=text,
        detail_text=detail,
        chunk_index=chunk_index,
        chunk_total=chunk_total,
        source="stage",
    )


def root_cause_from_chunk_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    status = _clean_text(row.get("status")) or ""
    if status not in {"failed", "cancelled", "split"}:
        return None
    code = _clean_text(row.get("error_code"))
    if status == "cancelled":
        code = "cancelled_by_user"
    return _root_cause_payload(
        code=code or "processing_runtime_error",
        detail_text=_clean_text(row.get("error_text")),
        chunk_index=_clean_text(row.get("chunk_index")),
        chunk_total=_int_or_none(row.get("chunk_total")),
        source="chunk",
    )


def root_cause_from_message(message: str, *, source: str) -> dict[str, Any]:
    text = _clean_text(message)
    if text is None:
        return _root_cause_payload(
            code="processing_runtime_error",
            explicit_text="Processing failed with a runtime error.",
            source=source,
        )
    chunk_match = _LLM_CHUNK_FAILURE_RE.fullmatch(text)
    if chunk_match is not None:
        return _root_cause_payload(
            code=chunk_match.group("code"),
            detail_text=chunk_match.group("detail"),
            chunk_index=chunk_match.group("chunk_index"),
            chunk_total=_int_or_none(chunk_match.group("chunk_total")),
            source=source,
        )
    merge_match = _LLM_MERGE_FAILURE_RE.fullmatch(text)
    if merge_match is not None:
        return _root_cause_payload(
            code=merge_match.group("code"),
            detail_text=merge_match.group("detail"),
            source=source,
        )
    return _root_cause_payload(
        code="processing_runtime_error",
        explicit_text=text,
        detail_text=text,
        source=source,
    )


def _recording_review_cause(recording: dict[str, Any]) -> dict[str, Any] | None:
    code = _clean_text(recording.get("review_reason_code"))
    text = _clean_text(recording.get("review_reason_text"))
    if code is None and text is None:
        return None
    return _root_cause_payload(
        code=code,
        explicit_text=text,
        detail_text=text,
        source="recording",
    )


def _recording_stop_cause(recording: dict[str, Any]) -> dict[str, Any] | None:
    status = _clean_text(recording.get("status")) or ""
    if status != "Stopped":
        return None
    reason_text = _clean_text(recording.get("cancel_reason_text"))
    forced = bool(reason_text and "force-stopped" in reason_text.lower())
    return _root_cause_payload(
        code="force_stopped_by_user" if forced else "cancelled_by_user",
        explicit_text=reason_text,
        detail_text=reason_text,
        source="recording",
    )


def _latest_job_wrapper(jobs: Sequence[dict[str, Any]]) -> str | None:
    for job in jobs:
        error = _clean_text(job.get("error"))
        if error is None:
            continue
        if error == "cancelled_by_user":
            continue
        if error == "max attempts exceeded":
            return "Automatic retries hit the configured retry limit."
        return error
    return None


def _select_current_stage_row(
    recording: dict[str, Any],
    stage_rows: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    target_stage = stage_name_for_progress(recording.get("pipeline_stage"))
    if target_stage is not None:
        for row in stage_rows:
            if _clean_text(row.get("stage_name")) == target_stage:
                return row
    if not stage_rows:
        return None
    return max(stage_rows, key=_sort_key)


def _select_current_chunk_row(
    recording: dict[str, Any],
    chunk_rows: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    if not chunk_rows:
        return None
    chunk_index, _chunk_total = _progress_stage_chunk_position(recording.get("pipeline_stage"))
    if chunk_index is not None:
        for row in chunk_rows:
            if _clean_text(row.get("chunk_index")) == chunk_index:
                return row
    priority = {"running": 5, "cancelled": 4, "failed": 3, "split": 2, "completed": 1}
    return max(
        chunk_rows,
        key=lambda row: (
            priority.get(_clean_text(row.get("status")) or "", 0),
            _sort_key(row),
        ),
    )


def build_recording_diagnostics(
    *,
    recording: dict[str, Any],
    stage_rows: Sequence[dict[str, Any]],
    chunk_rows: Sequence[dict[str, Any]],
    jobs: Sequence[dict[str, Any]],
    now: datetime | None = None,
) -> dict[str, Any]:
    current_time = now or datetime.now(tz=timezone.utc)
    current_stage_row = _select_current_stage_row(recording, stage_rows)
    current_chunk_row = _select_current_chunk_row(recording, chunk_rows)
    current_stage_code = (
        _clean_text(recording.get("pipeline_stage"))
        or _clean_text(current_stage_row.get("stage_name") if current_stage_row else None)
        or "waiting"
    )
    current_chunk_index, current_chunk_total = _progress_stage_chunk_position(current_stage_code)
    if current_chunk_row is not None:
        current_chunk_index = _clean_text(current_chunk_row.get("chunk_index")) or current_chunk_index
        current_chunk_total = _int_or_none(current_chunk_row.get("chunk_total")) or current_chunk_total

    stop_requested = bool(
        _clean_text(recording.get("cancel_requested_at"))
        or (_clean_text(recording.get("status")) in {"Stopping", "Stopped"})
    )
    stop_mode = None
    stop_reason_text = _clean_text(recording.get("cancel_reason_text"))
    if stop_reason_text and "force-stopped" in stop_reason_text.lower():
        stop_mode = "forced"
    elif stop_requested:
        stop_mode = "soft"

    recording_cause = _recording_review_cause(recording)
    stop_cause = _recording_stop_cause(recording)
    stage_cause = next(
        (
            root_cause_from_stage_row(row)
            for row in sorted(stage_rows, key=_sort_key, reverse=True)
            if root_cause_from_stage_row(row) is not None
        ),
        None,
    )
    chunk_cause = next(
        (
            root_cause_from_chunk_row(row)
            for row in sorted(chunk_rows, key=_sort_key, reverse=True)
            if root_cause_from_chunk_row(row) is not None
        ),
        None,
    )

    candidates = [candidate for candidate in (recording_cause, stage_cause, chunk_cause) if candidate is not None]
    primary = stop_cause
    if primary is None and candidates:
        prioritized: list[tuple[int, int, dict[str, Any]]] = []
        source_priority = {"recording": 3, "stage": 2, "chunk": 1}
        for candidate in candidates:
            prioritized.append(
                (
                    _cause_specificity(candidate.get("code")),
                    source_priority.get(str(candidate.get("source") or ""), 0),
                    candidate,
                )
            )
        prioritized.sort(key=lambda item: (item[0], item[1]), reverse=True)
        primary = prioritized[0][2]

    wrapper_text = None
    latest_wrapper = _latest_job_wrapper(jobs)
    if (
        recording_cause is not None
        and recording_cause.get("code") in _WRAPPER_ONLY_CODES
        and (primary or {}).get("code") != recording_cause.get("code")
    ):
        wrapper_text = latest_wrapper or str(recording_cause.get("text") or "")
    elif latest_wrapper is not None and latest_wrapper != (primary or {}).get("detail"):
        wrapper_text = latest_wrapper

    current_stage_status = _clean_text(
        current_stage_row.get("status") if current_stage_row else None
    ) or ("running" if _clean_text(recording.get("status")) in {"Processing", "Stopping"} else None)
    is_resuming = False
    if current_stage_row is not None and (_int_or_none(current_stage_row.get("attempt")) or 0) > 1:
        is_resuming = True
    elif current_chunk_row is not None and (_int_or_none(current_chunk_row.get("attempt")) or 0) > 1:
        is_resuming = True
    elif current_stage_row is not None and bool(_metadata_payload(current_stage_row).get("resumed")):
        is_resuming = True

    return {
        "current_stage_code": current_stage_code,
        "current_stage_status": current_stage_status,
        "current_stage_attempt": (
            max(_int_or_none(current_stage_row.get("attempt")) or 0, 0)
            if current_stage_row is not None
            else 0
        ),
        "stage_elapsed_seconds": _elapsed_seconds(
            started_at=current_stage_row.get("started_at") if current_stage_row else None,
            duration_ms=current_stage_row.get("duration_ms") if current_stage_row else None,
            now=current_time,
            is_running=current_stage_status == "running",
        ),
        "current_chunk_index": current_chunk_index,
        "current_chunk_total": current_chunk_total,
        "current_chunk_attempt": (
            max(_int_or_none(current_chunk_row.get("attempt")) or 0, 0)
            if current_chunk_row is not None
            else 0
        ),
        "chunk_elapsed_seconds": _elapsed_seconds(
            started_at=current_chunk_row.get("started_at") if current_chunk_row else None,
            duration_ms=current_chunk_row.get("duration_ms") if current_chunk_row else None,
            now=current_time,
            is_running=_clean_text(current_chunk_row.get("status") if current_chunk_row else None) == "running",
        ),
        "is_resuming": is_resuming,
        "is_stopping": _clean_text(recording.get("status")) == "Stopping",
        "is_stopped": _clean_text(recording.get("status")) == "Stopped",
        "stop_requested": stop_requested,
        "stop_mode": stop_mode,
        "stop_reason_text": stop_reason_text or ("Stop requested by user." if stop_requested else None),
        "primary_reason_code": _clean_text(primary.get("code")) if primary else None,
        "primary_reason_text": _clean_text(primary.get("text")) if primary else None,
        "primary_reason_detail": _clean_text(primary.get("detail")) if primary else None,
        "wrapper_reason_text": _clean_text(wrapper_text),
    }


__all__ = [
    "build_recording_diagnostics",
    "root_cause_from_chunk_row",
    "root_cause_from_exception",
    "root_cause_from_message",
    "root_cause_from_stage_row",
    "root_cause_summary",
    "stage_name_for_progress",
]
