from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

import httpx

from lan_transcriber.gpu_policy import (
    collect_cuda_runtime_facts,
    is_gpu_device,
)

from .config import AppSettings
from .constants import JOB_STATUS_QUEUED, JOB_STATUS_STARTED
from .db import get_recording, list_jobs

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


def _gpu_visibility_disabled() -> bool:
    raw = os.getenv("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return False
    return raw.strip().lower() in {"", "-1", "none", "void"}


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _probe_nvidia_smi() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {
            "available": False,
            "device_count": 0,
            "busy": False,
            "detail": "nvidia-smi unavailable",
        }

    try:
        completed = subprocess.run(
            [
                executable,
                "--query-gpu=index,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except subprocess.TimeoutExpired:
        return {
            "available": False,
            "device_count": 0,
            "busy": False,
            "detail": "nvidia-smi timed out",
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        stderr = str(getattr(exc, "stderr", "") or "").strip()
        detail = stderr or f"nvidia-smi failed: {type(exc).__name__}"
        return {
            "available": False,
            "device_count": 0,
            "busy": False,
            "detail": detail,
        }

    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not rows:
        return {
            "available": False,
            "device_count": 0,
            "busy": False,
            "detail": "nvidia-smi reported no visible GPUs",
        }

    utilizations = []
    for row in rows:
        parts = [part.strip() for part in row.split(",")]
        utilization = _parse_int(parts[1] if len(parts) > 1 else None)
        if utilization is not None:
            utilizations.append(max(utilization, 0))

    detail = f"nvidia-smi sees {len(rows)} GPU(s)"
    if utilizations:
        detail = f"{detail} · util up to {max(utilizations)}%"
    return {
        "available": True,
        "device_count": len(rows),
        "busy": any(utilization > 0 for utilization in utilizations),
        "detail": detail,
    }


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


def _node_status_item(
    *,
    spark: dict[str, Any],
    queue: dict[str, Any],
    active_llm: bool,
    active_detail: str,
) -> dict[str, Any]:
    if spark["state"] == "healthy":
        node_busy = bool(queue.get("started_total")) or active_llm
        detail = active_detail if node_busy and active_detail else spark["detail"]
        return {
            "label": "Node status",
            "value": "Busy" if node_busy else "Online",
            "detail": detail,
            "tone": "busy" if node_busy else "healthy",
            "show_dot": True,
        }
    if spark["state"] == "offline":
        return {
            "label": "Node status",
            "value": "Offline",
            "detail": spark["detail"],
            "tone": "offline",
            "show_dot": True,
        }
    return {
        "label": "Node status",
        "value": "Unknown",
        "detail": spark["detail"],
        "tone": "degraded",
        "show_dot": True,
    }


def _gpu_runtime_item(
    *,
    settings: AppSettings,
    queue: dict[str, Any],
    active_llm: bool,
    cuda_facts: Any,
) -> dict[str, Any]:
    explicit_gpu_requested = _safe_is_gpu_device(
        getattr(settings, "asr_device", None)
    ) or _safe_is_gpu_device(getattr(settings, "diarization_device", None))
    gpu_busy = bool(queue.get("started_total")) and not active_llm
    torch_cuda = cuda_facts.torch_cuda_version or "none"
    nvidia_smi = _probe_nvidia_smi()

    if cuda_facts.is_available and cuda_facts.device_count > 0:
        detail = f"torch sees {cuda_facts.device_count} GPU(s) · CUDA {torch_cuda}"
        if nvidia_smi["available"]:
            detail = f"{detail} · {nvidia_smi['detail']}"
        return {
            "label": "GPU runtime",
            "value": "GPU busy" if gpu_busy else "GPU ready",
            "detail": detail,
            "tone": "busy" if gpu_busy else "healthy",
        }

    if _gpu_visibility_disabled():
        return {
            "label": "GPU runtime",
            "value": "GPU unavailable" if explicit_gpu_requested else "CPU only",
            "detail": "CUDA_VISIBLE_DEVICES disables GPU visibility for this process",
            "tone": "offline" if explicit_gpu_requested else "degraded",
        }

    if nvidia_smi["available"]:
        detail = nvidia_smi["detail"]
        if torch_cuda != "none":
            detail = f"{detail} · torch CUDA {torch_cuda} but torch.cuda.is_available() is false"
        else:
            detail = f"{detail} · torch CUDA unavailable"
        return {
            "label": "GPU runtime",
            "value": "GPU unavailable",
            "detail": detail,
            "tone": "offline",
        }

    visible_devices = cuda_facts.visible_devices or "default"
    return {
        "label": "GPU runtime",
        "value": "GPU unavailable" if explicit_gpu_requested else "CPU only",
        "detail": f"visible={visible_devices} · torch CUDA {torch_cuda}",
        "tone": "offline" if explicit_gpu_requested else "degraded",
    }


def _llm_runtime_item(
    *,
    settings: AppSettings,
    spark: dict[str, Any],
    active_llm: bool,
) -> dict[str, Any]:
    configured_model = str(getattr(settings, "llm_model", "") or "").strip() or "Unknown"

    if spark["state"] == "healthy":
        tone = "busy" if active_llm else "healthy"
        if spark["model_verified"] is False:
            tone = "degraded"
        detail = (
            f"{spark['host']} · configured model is advertised"
            if spark["model_verified"] is not False
            else f"{spark['host']} · configured model is not advertised"
        )
    elif spark["state"] == "offline":
        tone = "offline"
        detail = f"{spark['host']} · endpoint offline"
    else:
        tone = "degraded"
        detail = spark["detail"]

    return {
        "label": "LLM:",
        "value": configured_model,
        "detail": detail,
        "tone": tone,
    }


def collect_control_center_runtime_status(settings: AppSettings) -> dict[str, Any]:
    spark = _probe_spark_runtime(settings)
    queue = _active_job_snapshot(settings)
    cuda_facts = collect_cuda_runtime_facts()

    active_stage = str(queue.get("active_stage") or "").strip().lower()
    active_llm = active_stage.startswith("llm")
    active_detail = str(queue.get("active_detail") or "").strip()

    return {
        "items": [
            _node_status_item(
                spark=spark,
                queue=queue,
                active_llm=active_llm,
                active_detail=active_detail,
            ),
            _gpu_runtime_item(
                settings=settings,
                queue=queue,
                active_llm=active_llm,
                cuda_facts=cuda_facts,
            ),
            _llm_runtime_item(
                settings=settings,
                spark=spark,
                active_llm=active_llm,
            ),
        ],
    }


__all__ = ["collect_control_center_runtime_status"]
