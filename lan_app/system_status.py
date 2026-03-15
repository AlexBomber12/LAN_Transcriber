from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx

from lan_transcriber.gpu_policy import (
    collect_cuda_runtime_facts,
    is_gpu_device,
    resolve_scheduler_decision,
)

from .config import AppSettings
from .constants import JOB_STATUS_QUEUED, JOB_STATUS_STARTED
from .db import get_recording, list_jobs
from .healthchecks import check_redis_health, check_worker_health

_SPARK_PROBE_TIMEOUT = httpx.Timeout(2.0, connect=0.5)


def _stage_label(value: object) -> str:
    text = str(value or "").strip().replace("-", "_")
    if not text:
        return "Unknown"
    labels: list[str] = []
    for part in [chunk for chunk in text.split("_") if chunk]:
        lowered = part.lower()
        if lowered == "llm":
            labels.append("LLM")
        elif lowered == "stt":
            labels.append("STT")
        elif part.isdigit():
            labels.append(part)
        else:
            labels.append(part.title())
    return " ".join(labels) or "Unknown"


def _recording_label(recording: dict[str, Any] | None, fallback: str) -> str:
    if not isinstance(recording, dict):
        return fallback
    source_filename = str(recording.get("source_filename") or "").strip()
    if source_filename:
        return source_filename
    recording_id = str(recording.get("id") or "").strip()
    return recording_id or fallback


def _load_json_dict(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _base_url_host(base_url: str | None) -> str:
    normalized = str(base_url or "").strip()
    if not normalized:
        return "unconfigured"
    try:
        parsed = httpx.URL(normalized)
    except Exception:
        return normalized.rstrip("/")
    host = parsed.host
    if not host:
        return normalized.rstrip("/")
    port = parsed.port
    if port and port not in {80, 443}:
        return f"{host}:{port}"
    return str(host)


def _spark_probe_url(base_url: str | None) -> str:
    normalized = str(base_url or "").strip().rstrip("/")
    if not normalized:
        return ""
    if normalized.endswith("/v1/chat/completions"):
        prefix = normalized[: -len("/chat/completions")]
        return f"{prefix}/models"
    if normalized.endswith("/v1"):
        return f"{normalized}/models"
    return f"{normalized}/v1/models"


def _extract_model_ids(payload: object) -> list[str]:
    if not isinstance(payload, dict):
        return []
    rows = payload.get("data")
    if not isinstance(rows, list):
        return []
    model_ids: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        model_id = str(row.get("id") or "").strip()
        if model_id:
            model_ids.append(model_id)
    return model_ids


def _probe_spark_runtime(settings: AppSettings) -> dict[str, Any]:
    base_url = str(getattr(settings, "llm_base_url", "") or "").strip()
    host = _base_url_host(base_url)
    if not base_url:
        return {
            "state": "unknown",
            "value": "Unknown",
            "detail": "LLM_BASE_URL is not configured",
            "host": host,
            "advertised_models": [],
            "model_verified": None,
        }

    headers = {"Accept": "application/json"}
    api_key = os.getenv("LLM_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        with httpx.Client(timeout=_SPARK_PROBE_TIMEOUT, follow_redirects=True) as client:
            response = client.get(_spark_probe_url(base_url), headers=headers)
    except httpx.TimeoutException:
        return {
            "state": "offline",
            "value": "Offline",
            "detail": f"{host} timed out",
            "host": host,
            "advertised_models": [],
            "model_verified": None,
        }
    except httpx.HTTPError as exc:
        return {
            "state": "offline",
            "value": "Offline",
            "detail": f"{host} probe failed: {type(exc).__name__}",
            "host": host,
            "advertised_models": [],
            "model_verified": None,
        }

    if response.status_code == 200:
        try:
            payload = response.json()
        except ValueError:
            return {
                "state": "degraded",
                "value": "Unknown",
                "detail": f"{host} returned non-JSON from /v1/models",
                "host": host,
                "advertised_models": [],
                "model_verified": None,
            }
        model_ids = _extract_model_ids(payload)
        configured_model = str(getattr(settings, "llm_model", "") or "").strip()
        return {
            "state": "healthy",
            "value": "Online",
            "detail": f"{host} responded to /v1/models",
            "host": host,
            "advertised_models": model_ids,
            "model_verified": configured_model in model_ids if configured_model else None,
        }

    if response.status_code in {401, 403}:
        return {
            "state": "degraded",
            "value": "Auth error",
            "detail": f"{host} rejected the runtime probe",
            "host": host,
            "advertised_models": [],
            "model_verified": None,
        }
    if response.status_code in {404, 405}:
        return {
            "state": "unknown",
            "value": "Unknown",
            "detail": f"{host} responded but /v1/models is unavailable",
            "host": host,
            "advertised_models": [],
            "model_verified": None,
        }
    if response.status_code >= 500:
        return {
            "state": "offline",
            "value": "Unavailable",
            "detail": f"{host} returned HTTP {response.status_code}",
            "host": host,
            "advertised_models": [],
            "model_verified": None,
        }
    return {
        "state": "degraded",
        "value": "Degraded",
        "detail": f"{host} returned HTTP {response.status_code}",
        "host": host,
        "advertised_models": [],
        "model_verified": None,
    }


def _safe_is_gpu_device(device: str | None) -> bool:
    try:
        return is_gpu_device(device)
    except ValueError:
        return False


def _safe_get_recording(
    recording_id: str,
    *,
    settings: AppSettings,
) -> dict[str, Any] | None:
    if not recording_id:
        return None
    try:
        recording = get_recording(recording_id, settings=settings)
    except Exception:
        return None
    return recording if isinstance(recording, dict) else None


def _active_job_snapshot(settings: AppSettings) -> dict[str, Any]:
    try:
        started_rows, started_total = list_jobs(
            settings=settings,
            status=JOB_STATUS_STARTED,
            limit=1,
            offset=0,
        )
        queued_rows, queued_total = list_jobs(
            settings=settings,
            status=JOB_STATUS_QUEUED,
            limit=1,
            offset=0,
        )
    except Exception as exc:
        return {
            "started_total": 0,
            "queued_total": 0,
            "active_job": None,
            "queued_job": None,
            "active_recording": None,
            "active_detail": None,
            "active_stage": "",
            "error": str(exc),
        }

    active_job = started_rows[0] if started_rows else None
    queued_job = queued_rows[0] if queued_rows else None
    active_recording = None
    if isinstance(active_job, dict):
        active_recording = _safe_get_recording(
            str(active_job.get("recording_id") or ""),
            settings=settings,
        )
    active_stage = str(
        (active_recording or {}).get("pipeline_stage")
        or (active_recording or {}).get("status")
        or (active_job or {}).get("type")
        or ""
    ).strip()
    active_detail = None
    if isinstance(active_job, dict):
        active_detail = (
            f"{_recording_label(active_recording, str(active_job.get('recording_id') or 'recording'))} "
            f"· {_stage_label(active_stage)}"
        )
    elif isinstance(queued_job, dict):
        queued_recording = _safe_get_recording(
            str(queued_job.get("recording_id") or ""),
            settings=settings,
        )
        active_detail = (
            f"Next: {_recording_label(queued_recording, str(queued_job.get('recording_id') or 'recording'))}"
        )

    return {
        "started_total": int(started_total),
        "queued_total": int(queued_total),
        "active_job": active_job,
        "queued_job": queued_job,
        "active_recording": active_recording,
        "active_detail": active_detail,
        "active_stage": active_stage,
        "error": None,
    }


def _active_runtime_metadata(
    settings: AppSettings,
    *,
    recording_id: str,
) -> dict[str, Any]:
    if not recording_id:
        return {}
    derived_dir = settings.recordings_root / recording_id / "derived"
    for name in ("diarization_metadata.json", "diarization_status.json"):
        payload = _load_json_dict(derived_dir / name)
        if payload:
            return payload
    return {}


def collect_control_center_runtime_status(settings: AppSettings) -> dict[str, Any]:
    worker_health = check_worker_health(settings)
    redis_health = check_redis_health(settings)
    spark = _probe_spark_runtime(settings)
    queue = _active_job_snapshot(settings)
    cuda_facts = collect_cuda_runtime_facts()

    active_stage = str(queue.get("active_stage") or "").strip().lower()
    active_llm = active_stage.startswith("llm")
    gpu_busy = bool(queue.get("started_total")) and not active_llm
    active_detail = str(queue.get("active_detail") or "").strip()

    if queue.get("error"):
        active_jobs_item = {
            "label": "Active jobs",
            "value": "Queue unknown",
            "detail": str(queue["error"]),
            "tone": "degraded",
        }
    elif not redis_health["ok"]:
        active_jobs_item = {
            "label": "Active jobs",
            "value": "Queue offline",
            "detail": str(redis_health["detail"]),
            "tone": "offline",
        }
    elif queue["started_total"] or queue["queued_total"]:
        active_jobs_item = {
            "label": "Active jobs",
            "value": f"{queue['started_total']} active · {queue['queued_total']} queued",
            "detail": active_detail or str(worker_health["detail"]),
            "tone": "busy" if worker_health["ok"] else "degraded",
        }
    else:
        active_jobs_item = {
            "label": "Active jobs",
            "value": "Idle",
            "detail": str(worker_health["detail"]),
            "tone": "healthy" if worker_health["ok"] else "degraded",
        }

    spark_tone = "degraded" if spark["state"] == "unknown" else spark["state"]
    spark_value = spark["value"]
    spark_detail = spark["detail"]
    if spark["state"] == "healthy":
        spark_tone = "busy" if active_llm else "healthy"
        spark_value = "Busy" if active_llm else "Online"
        spark_detail = (
            f"{active_detail} · {spark['host']}"
            if active_llm and active_detail
            else spark["detail"]
        )
    spark_item = {
        "label": "DGX / Spark",
        "value": spark_value,
        "detail": spark_detail,
        "tone": spark_tone,
    }

    explicit_gpu_requested = _safe_is_gpu_device(getattr(settings, "asr_device", None)) or _safe_is_gpu_device(
        getattr(settings, "diarization_device", None)
    )
    visible_devices = cuda_facts.visible_devices or "default"
    torch_cuda = cuda_facts.torch_cuda_version or "none"
    if cuda_facts.is_available and cuda_facts.device_count > 0:
        gpu_item = {
            "label": "GPU runtime",
            "value": "GPU active" if gpu_busy else "GPU ready",
            "detail": f"{cuda_facts.device_count} visible · torch CUDA {torch_cuda}",
            "tone": "busy" if gpu_busy else "healthy",
        }
    else:
        gpu_item = {
            "label": "GPU runtime",
            "value": "GPU unavailable" if explicit_gpu_requested else "CPU only",
            "detail": f"visible={visible_devices} · torch CUDA {torch_cuda}",
            "tone": "offline" if explicit_gpu_requested else "degraded",
        }

    active_recording = queue.get("active_recording") or {}
    active_recording_id = str(active_recording.get("id") or "").strip()
    runtime_metadata = _active_runtime_metadata(
        settings,
        recording_id=active_recording_id,
    )
    if runtime_metadata:
        effective_device = str(runtime_metadata.get("effective_device") or "").strip() or None
        runtime_mode = str(runtime_metadata.get("mode") or "").strip() or "unknown"
        if _safe_is_gpu_device(effective_device):
            inference_item = {
                "label": "Inference mode",
                "value": "GPU path",
                "detail": (
                    f"{_recording_label(active_recording, active_recording_id or 'recording')} "
                    f"· {_stage_label(active_stage)} · {effective_device}"
                ),
                "tone": "busy" if gpu_busy else "healthy",
            }
        else:
            fallback_tone = "offline" if explicit_gpu_requested else "degraded"
            inference_item = {
                "label": "Inference mode",
                "value": "CPU fallback" if runtime_mode != "unknown" else "CPU path",
                "detail": (
                    f"{_recording_label(active_recording, active_recording_id or 'recording')} "
                    f"· {_stage_label(active_stage)} · {runtime_mode}"
                ),
                "tone": fallback_tone,
            }
    else:
        try:
            scheduler = resolve_scheduler_decision(
                getattr(settings, "gpu_scheduler_mode", "auto"),
                asr_device=getattr(settings, "asr_device", "auto"),
                diarization_device=getattr(settings, "diarization_device", "auto"),
                diarization_is_heavy=True,
                cuda_facts=cuda_facts,
            )
            asr_gpu = _safe_is_gpu_device(scheduler.asr_device)
            diarization_gpu = _safe_is_gpu_device(scheduler.diarization_device)
            if asr_gpu and diarization_gpu:
                inference_item = {
                    "label": "Inference mode",
                    "value": "GPU path",
                    "detail": (
                        f"ASR {scheduler.asr_device} · Diarization {scheduler.diarization_device} "
                        f"· {scheduler.effective_mode}"
                    ),
                    "tone": "busy" if gpu_busy else "healthy",
                }
            elif asr_gpu or diarization_gpu:
                inference_item = {
                    "label": "Inference mode",
                    "value": "Mixed path",
                    "detail": (
                        f"ASR {scheduler.asr_device} · Diarization {scheduler.diarization_device} "
                        f"· {scheduler.effective_mode}"
                    ),
                    "tone": "busy" if gpu_busy else "degraded",
                }
            else:
                inference_item = {
                    "label": "Inference mode",
                    "value": "CPU fallback" if explicit_gpu_requested else "CPU path",
                    "detail": (
                        f"ASR {scheduler.asr_device} · Diarization {scheduler.diarization_device} "
                        f"· {scheduler.effective_mode}"
                    ),
                    "tone": "offline" if explicit_gpu_requested else "degraded",
                }
        except Exception as exc:
            inference_item = {
                "label": "Inference mode",
                "value": "Blocked",
                "detail": str(exc),
                "tone": "offline",
            }

    configured_model = str(getattr(settings, "llm_model", "") or "").strip() or "Unknown"
    if spark["state"] == "healthy":
        target_tone = "busy" if active_llm else "healthy"
        if spark["model_verified"] is False:
            target_tone = "degraded"
        target_detail = (
            f"{spark['host']} · advertised by Spark"
            if spark["model_verified"] is not False
            else f"{spark['host']} · configured target not advertised"
        )
        if active_llm and active_detail:
            target_detail = f"{active_detail} · {spark['host']}"
    elif spark["state"] == "offline":
        target_tone = "offline"
        target_detail = f"{spark['host']} · configured target only"
    else:
        target_tone = "degraded"
        target_detail = spark["detail"]
    target_item = {
        "label": "Inference target",
        "value": configured_model,
        "detail": target_detail,
        "tone": target_tone,
    }

    return {
        "active_jobs_item": active_jobs_item,
        "secondary_items": [
            spark_item,
            gpu_item,
            inference_item,
            target_item,
        ],
        "note": (
            "Runtime snapshot uses worker heartbeat, DB job rows, local CUDA visibility, "
            "and a lightweight Spark /v1/models probe. CPU fallback is confirmed from "
            "active diarization metadata when available and otherwise inferred from settings."
        ),
    }


__all__ = ["collect_control_center_runtime_status"]
