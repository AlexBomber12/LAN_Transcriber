from __future__ import annotations

from datetime import datetime, timezone

from lan_app import diagnostics
from lan_transcriber.llm_client import LLMEmptyContentError, LLMTruncatedResponseError


def test_diagnostics_internal_helpers_cover_scalar_and_time_paths() -> None:
    now = datetime(2026, 3, 12, 12, 0, 5, tzinfo=timezone.utc)

    assert diagnostics._clean_text("  hello  ") == "hello"  # noqa: SLF001
    assert diagnostics._clean_text("   ") is None  # noqa: SLF001
    assert diagnostics._int_or_none("7") == 7  # noqa: SLF001
    assert diagnostics._int_or_none("bad") is None  # noqa: SLF001
    assert diagnostics._parse_timestamp("2026-03-12T12:00:00Z") == datetime(  # noqa: SLF001
        2026,
        3,
        12,
        12,
        0,
        0,
        tzinfo=timezone.utc,
    )
    assert diagnostics._parse_timestamp("bad") is None  # noqa: SLF001
    assert diagnostics._elapsed_seconds(  # noqa: SLF001
        started_at="2026-03-12T12:00:00Z",
        duration_ms=1500,
        now=now,
        is_running=False,
    ) == 1.5
    assert diagnostics._elapsed_seconds(  # noqa: SLF001
        started_at="2026-03-12T12:00:00Z",
        duration_ms=None,
        now=now,
        is_running=False,
    ) is None
    assert diagnostics._elapsed_seconds(  # noqa: SLF001
        started_at="2026-03-12T12:00:00Z",
        duration_ms=None,
        now=now,
        is_running=True,
    ) == 5.0
    assert diagnostics._elapsed_seconds(  # noqa: SLF001
        started_at="not-a-time",
        duration_ms=None,
        now=now,
        is_running=True,
    ) is None
    assert diagnostics._metadata_payload(None) == {}  # noqa: SLF001
    assert diagnostics._metadata_payload({"metadata_json": []}) == {}  # noqa: SLF001
    assert diagnostics._metadata_payload({"metadata_json": {"a": 1}}) == {"a": 1}  # noqa: SLF001
    assert diagnostics._sort_key({"updated_at": "u", "finished_at": "f", "started_at": "s"}) == (  # noqa: SLF001
        "u",
        "f",
        "s",
    )
    assert diagnostics._progress_stage_chunk_position("llm_chunk_2_of_5") == ("2", 5)  # noqa: SLF001
    assert diagnostics._progress_stage_chunk_position("diarize") == (None, None)  # noqa: SLF001
    assert diagnostics.stage_name_for_progress("llm_chunk_2_of_5") == "llm_extract"
    assert diagnostics.stage_name_for_progress("llm_merge") == "llm_extract"
    assert diagnostics.stage_name_for_progress("diarize") == "diarize"
    assert diagnostics.stage_name_for_progress(None) is None
    assert diagnostics._cause_specificity("job_retry_limit_reached") == 0  # noqa: SLF001
    assert diagnostics._cause_specificity("processing_runtime_error") == 1  # noqa: SLF001
    assert diagnostics._cause_specificity("gpu_oom") == 2  # noqa: SLF001
    assert diagnostics._cause_specificity(None) == -1  # noqa: SLF001
    assert diagnostics._humanize_unknown_code(None) == "Processing issue"  # noqa: SLF001
    assert diagnostics._humanize_unknown_code("chunk_parse_error") == "Chunk parse error"  # noqa: SLF001


def test_root_cause_summary_and_message_paths() -> None:
    assert diagnostics.root_cause_summary("gpu_oom") == (
        "The worker ran out of GPU memory while loading or running a heavy model."
    )
    assert diagnostics.root_cause_summary(
        "llm_chunk_timeout",
        chunk_index="3",
        chunk_total=10,
    ) == "LLM chunk 3/10 timed out."
    assert diagnostics.root_cause_summary("llm_chunk_timeout") == "An LLM chunk timed out."
    assert diagnostics.root_cause_summary("llm_chunk_request_timeout") == (
        "An LLM chunk request timed out waiting on the model service."
    )
    assert diagnostics.root_cause_summary(
        "llm_chunk_request_timeout",
        chunk_index="4",
        chunk_total=8,
    ) == "LLM chunk 4/8 timed out waiting on the model service."
    assert diagnostics.root_cause_summary("llm_chunk_parse_error") == (
        "An LLM chunk returned an invalid response."
    )
    assert diagnostics.root_cause_summary("llm_chunk_connection_error") == (
        "An LLM chunk lost the model connection."
    )
    assert diagnostics.root_cause_summary("llm_chunk_runtime_error") == (
        "An LLM chunk failed during processing."
    )
    assert diagnostics.root_cause_summary(
        "llm_chunk_runtime_error",
        chunk_index="6",
        chunk_total=9,
    ) == "LLM chunk 6/9 failed during processing."
    assert diagnostics.root_cause_summary("llm_merge_request_timeout") == (
        "The final LLM merge request timed out waiting on the model service."
    )
    assert diagnostics.root_cause_summary("calendar_time_mismatch") == (
        "The recording time does not match the selected calendar event closely enough."
    )
    assert diagnostics.root_cause_summary("suspicious_capture_time") == (
        "The recording capture time looks suspicious and should be checked."
    )
    assert diagnostics.root_cause_summary("llm_merge_error") == (
        "The final LLM merge response could not be parsed."
    )
    assert diagnostics.root_cause_summary(
        "processing_runtime_error",
        explicit_text="boom",
    ) == "boom"
    assert diagnostics.root_cause_summary("job_retry_limit_reached") == (
        "Processing hit the retry limit after repeated failures."
    )
    assert diagnostics.root_cause_summary("custom_reason", explicit_text="custom text") == "custom text"

    empty = diagnostics.root_cause_from_message("", source="stage")
    assert empty["code"] == "processing_runtime_error"
    assert empty["text"] == "Processing failed with a runtime error."
    assert empty["detail"] is None

    chunk = diagnostics.root_cause_from_message(
        "LLM chunk 3/10 failed [llm_chunk_timeout]: timed out after 12s",
        source="stage",
    )
    assert chunk == {
        "code": "llm_chunk_timeout",
        "text": "LLM chunk 3/10 timed out.",
        "detail": "timed out after 12s",
        "chunk_index": "3",
        "chunk_total": 10,
        "source": "stage",
    }

    merge = diagnostics.root_cause_from_message(
        "LLM merge failed [llm_merge_connection_error]: network down",
        source="chunk",
    )
    assert merge["code"] == "llm_merge_connection_error"
    assert merge["text"] == "The final LLM merge step lost the model connection."
    assert merge["detail"] == "network down"

    generic = diagnostics.root_cause_from_message("plain boom", source="stage")
    assert generic["code"] == "processing_runtime_error"
    assert generic["text"] == "plain boom"
    assert generic["detail"] is None
    assert diagnostics.root_cause_summary("mystery_code") == "Mystery code"


def test_root_cause_from_exception_covers_specific_and_fallback_paths() -> None:
    truncated = diagnostics.root_cause_from_exception(
        LLMTruncatedResponseError(
            host="localhost",
            model="test-model",
            max_tokens=123,
            request_id="req-1",
            raw_response={},
        )
    )
    assert truncated["code"] == "llm_truncated"
    assert truncated["detail"]

    empty = diagnostics.root_cause_from_exception(
        LLMEmptyContentError(
            host="localhost",
            model="test-model",
            max_tokens=123,
            finish_reason="stop",
            request_id="req-2",
            raw_response={},
        )
    )
    assert empty["code"] == "llm_empty_content"
    assert empty["detail"]

    gpu = diagnostics.root_cause_from_exception(
        RuntimeError("CUDA out of memory while loading faster-whisper")
    )
    assert gpu["code"] == "gpu_oom"
    assert gpu["detail"] == "CUDA out of memory while loading faster-whisper"

    chunk = diagnostics.root_cause_from_exception(
        RuntimeError("LLM chunk 2/4 failed [llm_chunk_parse_error]: json_object_not_found")
    )
    assert chunk["code"] == "llm_chunk_parse_error"
    assert chunk["chunk_index"] == "2"
    assert chunk["chunk_total"] == 4

    merge = diagnostics.root_cause_from_exception(
        RuntimeError("LLM merge failed [llm_merge_timeout]: timed out after 20s")
    )
    assert merge["code"] == "llm_merge_timeout"
    assert merge["detail"] == "timed out after 20s"

    retry_limit = diagnostics.root_cause_from_exception(RuntimeError("max attempts exceeded"))
    assert retry_limit["code"] == "job_retry_limit_reached"

    generic = diagnostics.root_cause_from_exception(RuntimeError())
    assert generic["code"] == "processing_runtime_error"
    assert generic["text"] == "Processing failed with RuntimeError."
    assert generic["detail"] is None


def test_root_cause_from_stage_and_chunk_rows_cover_cancelled_and_fallbacks() -> None:
    assert diagnostics.root_cause_from_stage_row(None) is None
    assert diagnostics.root_cause_from_chunk_row(None) is None

    forced_cancel = diagnostics.root_cause_from_stage_row(
        {
            "status": "cancelled",
            "metadata_json": {
                "cancelled_by_user": True,
                "force_stopped": True,
                "cancel_reason_text": "Force-stopped after grace timeout",
                "cancel_chunk_index": "4",
                "cancel_chunk_total": 8,
            },
        }
    )
    assert forced_cancel == {
        "code": "force_stopped_by_user",
        "text": "Processing had to be force-stopped after the grace timeout.",
        "detail": "Force-stopped after grace timeout",
        "chunk_index": "4",
        "chunk_total": 8,
        "source": "stage",
    }

    stage_metadata = diagnostics.root_cause_from_stage_row(
        {
            "status": "failed",
            "error_code": "RuntimeError",
            "error_text": "plain failure",
            "metadata_json": {
                "root_cause_code": "llm_merge_parse_error",
                "root_cause_text": "The final LLM merge response could not be parsed.",
                "root_cause_detail": "json_object_not_found",
            },
        }
    )
    assert stage_metadata["code"] == "llm_merge_parse_error"
    assert stage_metadata["detail"] == "json_object_not_found"

    parsed_stage = diagnostics.root_cause_from_stage_row(
        {
            "status": "failed",
            "error_code": "RuntimeError",
            "error_text": "LLM chunk 5/9 failed [llm_chunk_connection_error]: network down",
            "metadata_json": {},
        }
    )
    assert parsed_stage["code"] == "llm_chunk_connection_error"
    assert parsed_stage["chunk_index"] == "5"
    assert parsed_stage["chunk_total"] == 9

    assert diagnostics.root_cause_from_stage_row(
        {"status": "completed", "error_text": None, "metadata_json": {}}
    ) is None

    assert diagnostics.root_cause_from_chunk_row(
        {"status": "running", "chunk_index": "1", "chunk_total": 2}
    ) is None
    cancelled_chunk = diagnostics.root_cause_from_chunk_row(
        {"status": "cancelled", "chunk_index": "7", "chunk_total": 9, "error_text": "ignored"}
    )
    assert cancelled_chunk["code"] == "cancelled_by_user"
    assert diagnostics.root_cause_from_chunk_row(
        {
            "status": "split",
            "error_code": "llm_chunk_timeout",
            "error_text": "timed out after 20s",
            "chunk_index": "3",
            "chunk_total": 6,
        }
    ) is None


def test_build_recording_diagnostics_prefers_specific_stage_cause_over_retry_wrapper() -> None:
    now = datetime(2026, 3, 12, 12, 0, 5, tzinfo=timezone.utc)
    diagnostics_payload = diagnostics.build_recording_diagnostics(
        recording={
            "status": "NeedsReview",
            "review_reason_code": "job_retry_limit_reached",
            "review_reason_text": "Processing hit the retry limit after repeated failures.",
            "pipeline_stage": None,
        },
        stage_rows=[
            {
                "stage_name": "llm_extract",
                "status": "failed",
                "attempt": 2,
                "started_at": "2026-03-12T12:00:00Z",
                "duration_ms": 3000,
                "updated_at": "2026-03-12T12:00:03Z",
                "error_code": "llm_chunk_timeout",
                "error_text": "LLM chunk 3/10 failed [llm_chunk_timeout]: timed out after 120s",
                "metadata_json": {
                    "root_cause_code": "llm_chunk_timeout",
                    "root_cause_text": "LLM chunk 3/10 timed out.",
                    "root_cause_detail": "timed out after 120s",
                    "chunk_index": "3",
                    "chunk_total": 10,
                    "resumed": True,
                },
            }
        ],
        chunk_rows=[
            {
                "chunk_index": "3",
                "chunk_total": 10,
                "status": "failed",
                "attempt": 2,
                "started_at": "2026-03-12T12:00:01Z",
                "duration_ms": 2000,
                "updated_at": "2026-03-12T12:00:03Z",
                "error_code": "llm_chunk_timeout",
                "error_text": "timed out after 120s",
            }
        ],
        jobs=[{"error": "max attempts exceeded"}],
        now=now,
    )

    assert diagnostics_payload["current_stage_code"] == "llm_extract"
    assert diagnostics_payload["current_stage_status"] == "failed"
    assert diagnostics_payload["current_stage_attempt"] == 2
    assert diagnostics_payload["stage_elapsed_seconds"] == 3.0
    assert diagnostics_payload["current_chunk_index"] == "3"
    assert diagnostics_payload["current_chunk_total"] == 10
    assert diagnostics_payload["current_chunk_attempt"] == 2
    assert diagnostics_payload["chunk_elapsed_seconds"] == 2.0
    assert diagnostics_payload["is_resuming"] is True
    assert diagnostics_payload["primary_reason_code"] == "llm_chunk_timeout"
    assert diagnostics_payload["primary_reason_text"] == "LLM chunk 3/10 timed out."
    assert diagnostics_payload["primary_reason_detail"] == "timed out after 120s"
    assert diagnostics_payload["wrapper_reason_text"] == "Automatic retries hit the configured retry limit."


def test_build_recording_diagnostics_stopping_and_stopped_paths() -> None:
    now = datetime(2026, 3, 12, 12, 0, 5, tzinfo=timezone.utc)

    stopping = diagnostics.build_recording_diagnostics(
        recording={
            "status": "Stopping",
            "pipeline_stage": "llm_chunk_2_of_4",
            "cancel_requested_at": "2026-03-12T12:00:04Z",
            "cancel_reason_text": None,
        },
        stage_rows=[
            {
                "stage_name": "llm_extract",
                "status": "running",
                "attempt": 1,
                "started_at": "2026-03-12T12:00:00Z",
                "duration_ms": None,
                "updated_at": "2026-03-12T12:00:04Z",
                "metadata_json": {},
            }
        ],
        chunk_rows=[
            {
                "chunk_index": "2",
                "chunk_total": 4,
                "status": "running",
                "attempt": 1,
                "started_at": "2026-03-12T12:00:02Z",
                "duration_ms": None,
                "updated_at": "2026-03-12T12:00:04Z",
            }
        ],
        jobs=[{"error": "cancelled_by_user"}],
        now=now,
    )
    assert stopping["current_stage_code"] == "llm_chunk_2_of_4"
    assert stopping["current_chunk_index"] == "2"
    assert stopping["current_chunk_total"] == 4
    assert stopping["stop_requested"] is True
    assert stopping["stop_mode"] == "soft"
    assert stopping["stop_reason_text"] == "Stop requested by user."
    assert stopping["primary_reason_code"] is None
    assert stopping["wrapper_reason_text"] is None
    assert stopping["stage_elapsed_seconds"] == 5.0
    assert stopping["chunk_elapsed_seconds"] == 3.0

    stopped = diagnostics.build_recording_diagnostics(
        recording={
            "status": "Stopped",
            "pipeline_stage": None,
            "cancel_reason_text": "Force-stopped after grace timeout",
        },
        stage_rows=[
            {
                "stage_name": "llm_extract",
                "status": "cancelled",
                "attempt": 1,
                "started_at": "2026-03-12T12:00:00Z",
                "duration_ms": 4000,
                "updated_at": "2026-03-12T12:00:04Z",
                "metadata_json": {
                    "cancelled_by_user": True,
                    "force_stopped": True,
                    "cancel_reason_text": "Force-stopped after grace timeout",
                    "root_cause_code": "force_stopped_by_user",
                    "root_cause_text": "Processing had to be force-stopped after the grace timeout.",
                },
            }
        ],
        chunk_rows=[],
        jobs=[{"error": "cancelled_by_user"}],
        now=now,
    )
    assert stopped["current_stage_code"] == "llm_extract"
    assert stopped["stop_requested"] is True
    assert stopped["stop_mode"] == "forced"
    assert stopped["primary_reason_code"] == "force_stopped_by_user"
    assert stopped["primary_reason_text"] == (
        "Processing had to be force-stopped after the grace timeout."
    )
    assert stopped["wrapper_reason_text"] is None


def test_build_recording_diagnostics_ignores_split_chunk_rows_for_primary_reason() -> None:
    diagnostics_payload = diagnostics.build_recording_diagnostics(
        recording={
            "status": "Ready",
            "pipeline_stage": None,
        },
        stage_rows=[
            {
                "stage_name": "llm_extract",
                "status": "completed",
                "attempt": 1,
                "started_at": "2026-03-12T12:00:00Z",
                "duration_ms": 4000,
                "updated_at": "2026-03-12T12:00:04Z",
                "metadata_json": {},
            }
        ],
        chunk_rows=[
            {
                "chunk_index": "3",
                "chunk_total": 6,
                "status": "split",
                "attempt": 1,
                "updated_at": "2026-03-12T12:00:03Z",
                "error_code": "llm_chunk_timeout",
                "error_text": "timed out after 20s",
            },
            {
                "chunk_index": "3a",
                "chunk_total": 6,
                "status": "completed",
                "attempt": 1,
                "updated_at": "2026-03-12T12:00:04Z",
            },
        ],
        jobs=[],
        now=datetime(2026, 3, 12, 12, 0, 5, tzinfo=timezone.utc),
    )

    assert diagnostics_payload["primary_reason_code"] is None
    assert diagnostics_payload["primary_reason_text"] is None
    assert diagnostics_payload["primary_reason_detail"] is None


def test_diagnostics_selection_helpers_cover_match_and_fallback_paths() -> None:
    assert diagnostics._select_current_stage_row(  # noqa: SLF001
        {"pipeline_stage": "llm_merge"},
        [
            {"stage_name": "asr", "updated_at": "2026-03-12T12:00:01Z"},
            {"stage_name": "llm_extract", "updated_at": "2026-03-12T12:00:02Z"},
        ],
    )["stage_name"] == "llm_extract"
    assert diagnostics._select_current_stage_row(  # noqa: SLF001
        {"pipeline_stage": None},
        [
            {"stage_name": "asr", "updated_at": "2026-03-12T12:00:01Z"},
            {"stage_name": "llm_extract", "updated_at": "2026-03-12T12:00:03Z"},
        ],
    )["stage_name"] == "llm_extract"
    assert diagnostics._select_current_stage_row({"pipeline_stage": "missing"}, []) is None  # noqa: SLF001

    assert diagnostics._select_current_chunk_row(  # noqa: SLF001
        {"pipeline_stage": "llm_chunk_2_of_4"},
        [
            {"chunk_index": "1", "status": "completed", "updated_at": "1"},
            {"chunk_index": "2", "status": "running", "updated_at": "2"},
        ],
    )["chunk_index"] == "2"
    assert diagnostics._select_current_chunk_row(  # noqa: SLF001
        {"pipeline_stage": "llm_chunk_3_of_4"},
        [
            {"chunk_index": "1", "status": "completed", "updated_at": "1"},
            {"chunk_index": "2", "status": "failed", "updated_at": "2"},
        ],
    )["chunk_index"] == "2"
    assert diagnostics._select_current_chunk_row({"pipeline_stage": "asr"}, []) is None  # noqa: SLF001
    assert diagnostics._select_current_chunk_row(  # noqa: SLF001
        {"pipeline_stage": "asr"},
        [
            {"chunk_index": "1", "status": "completed", "updated_at": "1"},
            {"chunk_index": "2", "status": "failed", "updated_at": "2"},
        ],
    )["chunk_index"] == "2"

    empty = diagnostics.build_recording_diagnostics(
        recording={"status": "Queued", "pipeline_stage": None},
        stage_rows=[],
        chunk_rows=[],
        jobs=[],
        now=datetime(2026, 3, 12, 12, 0, 0, tzinfo=timezone.utc),
    )
    assert empty["current_stage_attempt"] == 0
    assert empty["current_stage_code"] == "waiting"

    chunk_resume = diagnostics.build_recording_diagnostics(
        recording={"status": "Processing", "pipeline_stage": "llm_chunk_1_of_2"},
        stage_rows=[
            {
                "stage_name": "llm_extract",
                "status": "running",
                "attempt": 1,
                "started_at": "2026-03-12T12:00:00Z",
                "updated_at": "2026-03-12T12:00:01Z",
                "metadata_json": {},
            }
        ],
        chunk_rows=[
            {
                "chunk_index": "1",
                "chunk_total": 2,
                "status": "running",
                "attempt": 2,
                "started_at": "2026-03-12T12:00:01Z",
                "updated_at": "2026-03-12T12:00:02Z",
            }
        ],
        jobs=[],
        now=datetime(2026, 3, 12, 12, 0, 3, tzinfo=timezone.utc),
    )
    assert chunk_resume["is_resuming"] is True
