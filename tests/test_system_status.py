from __future__ import annotations

import json
from types import SimpleNamespace

import httpx

from lan_app import system_status
from lan_transcriber.gpu_policy import CudaRuntimeFacts


def _settings(tmp_path, **overrides):
    base = {
        "recordings_root": tmp_path / "recordings",
        "llm_base_url": "http://dgx.local:8000",
        "llm_model": "gpt-oss:120b",
        "asr_device": "auto",
        "diarization_device": "auto",
        "gpu_scheduler_mode": "auto",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class _FakeResponse:
    def __init__(self, status_code: int, payload=None, *, json_error: Exception | None = None):
        self.status_code = status_code
        self._payload = payload
        self._json_error = json_error

    def json(self):
        if self._json_error is not None:
            raise self._json_error
        return self._payload


def _client_factory(result, seen_headers: list[dict[str, str]] | None = None):
    class _FakeClient:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url: str, headers: dict[str, str] | None = None):
            del url
            if seen_headers is not None:
                seen_headers.append(headers or {})
            if isinstance(result, Exception):
                raise result
            return result

    return _FakeClient


def _nvidia_smi_probe(**overrides):
    probe = {
        "available": False,
        "device_count": 0,
        "busy": False,
        "detail": "nvidia-smi unavailable",
    }
    probe.update(overrides)
    return probe


def test_system_status_small_helpers_cover_edge_cases(tmp_path, monkeypatch):
    assert system_status._stage_label("") == "Unknown"  # noqa: SLF001
    assert system_status._stage_label("llm_chunk_2_of_5") == "LLM Chunk 2 Of 5"  # noqa: SLF001
    assert system_status._stage_label("stt") == "STT"  # noqa: SLF001
    assert system_status._recording_label({"source_filename": "meeting.mp3"}, "rec-1") == "meeting.mp3"  # noqa: SLF001
    assert system_status._recording_label({"id": "rec-2"}, "fallback") == "rec-2"  # noqa: SLF001
    assert system_status._recording_label(None, "fallback") == "fallback"  # noqa: SLF001

    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps({"ok": True}), encoding="utf-8")
    list_path = tmp_path / "payload-list.json"
    list_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    broken_path = tmp_path / "broken.json"
    broken_path.write_text("{", encoding="utf-8")
    assert system_status._load_json_dict(payload_path) == {"ok": True}  # noqa: SLF001
    assert system_status._load_json_dict(list_path) == {}  # noqa: SLF001
    assert system_status._load_json_dict(broken_path) == {}  # noqa: SLF001
    assert system_status._load_json_dict(tmp_path / "missing.json") == {}  # noqa: SLF001

    assert system_status._base_url_host(None) == "unconfigured"  # noqa: SLF001
    assert system_status._base_url_host("http://dgx.local:8000/v1") == "dgx.local:8000"  # noqa: SLF001
    assert system_status._base_url_host("http://dgx.local") == "dgx.local"  # noqa: SLF001
    monkeypatch.setattr(
        system_status.httpx,
        "URL",
        lambda _value: SimpleNamespace(host=None, port=None),
    )
    assert system_status._base_url_host("http://missing-host") == "http://missing-host"  # noqa: SLF001
    monkeypatch.undo()
    monkeypatch.setattr(
        system_status.httpx,
        "URL",
        lambda _value: (_ for _ in ()).throw(ValueError("bad url")),
    )
    assert system_status._base_url_host("not-a-url") == "not-a-url"  # noqa: SLF001
    monkeypatch.undo()
    assert system_status._spark_probe_url("") == ""  # noqa: SLF001
    assert system_status._spark_probe_url("http://dgx.local:8000/v1") == "http://dgx.local:8000/v1/models"  # noqa: SLF001
    assert system_status._spark_probe_url("http://dgx.local:8000/v1/chat/completions") == (  # noqa: SLF001
        "http://dgx.local:8000/v1/models"
    )
    assert system_status._spark_probe_url("http://dgx.local:8000") == "http://dgx.local:8000/v1/models"  # noqa: SLF001
    assert system_status._extract_model_ids({"data": [{"id": "a"}, "skip", {"id": ""}, {"id": "b"}]}) == [  # noqa: SLF001
        "a",
        "b",
    ]
    assert system_status._extract_model_ids({"data": "bad"}) == []  # noqa: SLF001
    assert system_status._extract_model_ids([]) == []  # noqa: SLF001
    assert system_status._safe_is_gpu_device("cuda:0") is True  # noqa: SLF001
    assert system_status._safe_is_gpu_device("bogus") is False  # noqa: SLF001
    assert system_status._safe_get_recording("", settings=_settings(tmp_path)) is None  # noqa: SLF001
    assert system_status._parse_int(None) is None  # noqa: SLF001
    assert system_status._parse_int("4") == 4  # noqa: SLF001
    assert system_status._parse_int("bad") is None  # noqa: SLF001

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert system_status._gpu_visibility_disabled() is False  # noqa: SLF001
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert system_status._gpu_visibility_disabled() is True  # noqa: SLF001
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert system_status._gpu_visibility_disabled() is False  # noqa: SLF001

    monkeypatch.setattr(system_status.shutil, "which", lambda _name: None)
    assert system_status._probe_nvidia_smi()["available"] is False  # noqa: SLF001

    monkeypatch.setattr(system_status.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        system_status.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="0, 0\n1, 76\n"),
    )
    probe = system_status._probe_nvidia_smi()  # noqa: SLF001
    assert probe["available"] is True
    assert probe["device_count"] == 2
    assert probe["busy"] is True
    assert probe["detail"].endswith("76%")

    monkeypatch.setattr(
        system_status.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            system_status.subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=2.0)
        ),
    )
    assert system_status._probe_nvidia_smi()["detail"] == "nvidia-smi timed out"  # noqa: SLF001

    monkeypatch.setattr(
        system_status.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            system_status.subprocess.CalledProcessError(
                1,
                ["nvidia-smi"],
                stderr="driver mismatch",
            )
        ),
    )
    assert system_status._probe_nvidia_smi()["detail"] == "driver mismatch"  # noqa: SLF001

    monkeypatch.setattr(
        system_status.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=""),
    )
    assert system_status._probe_nvidia_smi()["detail"] == "nvidia-smi reported no visible GPUs"  # noqa: SLF001

    monkeypatch.setattr(
        system_status.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="0\n1, not-a-number\n"),
    )
    probe_without_util = system_status._probe_nvidia_smi()  # noqa: SLF001
    assert probe_without_util["available"] is True
    assert probe_without_util["busy"] is False
    assert probe_without_util["detail"] == "nvidia-smi sees 2 GPU(s)"


def test_probe_spark_runtime_variants(tmp_path, monkeypatch):
    unknown = system_status._probe_spark_runtime(_settings(tmp_path, llm_base_url=""))
    assert unknown["state"] == "unknown"

    seen_headers: list[dict[str, str]] = []
    monkeypatch.setenv("LLM_API_KEY", "secret-token")
    monkeypatch.setattr(
        system_status.httpx,
        "Client",
        _client_factory(
            _FakeResponse(200, {"data": [{"id": "gpt-oss:120b"}]}),
            seen_headers,
        ),
    )
    healthy = system_status._probe_spark_runtime(_settings(tmp_path))
    assert healthy["state"] == "healthy"
    assert healthy["model_verified"] is True
    assert seen_headers[-1]["Authorization"] == "Bearer secret-token"

    monkeypatch.setattr(
        system_status.httpx,
        "Client",
        _client_factory(_FakeResponse(200, None, json_error=ValueError("bad json"))),
    )
    assert system_status._probe_spark_runtime(_settings(tmp_path))["state"] == "degraded"

    monkeypatch.setattr(
        system_status.httpx,
        "Client",
        _client_factory(_FakeResponse(401, {})),
    )
    assert system_status._probe_spark_runtime(_settings(tmp_path))["value"] == "Auth error"

    monkeypatch.setattr(
        system_status.httpx,
        "Client",
        _client_factory(_FakeResponse(404, {})),
    )
    assert system_status._probe_spark_runtime(_settings(tmp_path))["state"] == "unknown"

    monkeypatch.setattr(
        system_status.httpx,
        "Client",
        _client_factory(_FakeResponse(503, {})),
    )
    assert system_status._probe_spark_runtime(_settings(tmp_path))["value"] == "Unavailable"

    monkeypatch.setattr(
        system_status.httpx,
        "Client",
        _client_factory(_FakeResponse(418, {})),
    )
    assert system_status._probe_spark_runtime(_settings(tmp_path))["value"] == "Degraded"

    monkeypatch.setattr(
        system_status.httpx,
        "Client",
        _client_factory(httpx.ReadTimeout("slow")),
    )
    assert system_status._probe_spark_runtime(_settings(tmp_path))["detail"].endswith("timed out")

    monkeypatch.setattr(
        system_status.httpx,
        "Client",
        _client_factory(httpx.HTTPError("boom")),
    )
    assert "probe failed" in system_status._probe_spark_runtime(_settings(tmp_path))["detail"]


def test_active_job_snapshot_paths(tmp_path, monkeypatch):
    queued_row = {"recording_id": "rec-queued-1", "type": "precheck"}
    started_row = {"recording_id": "rec-started-1", "type": "precheck"}

    def _list_jobs_started(*, settings, status, limit, offset):
        del settings, limit, offset
        if status == "started":
            return [started_row], 1
        return [queued_row], 2

    monkeypatch.setattr(system_status, "list_jobs", _list_jobs_started)
    monkeypatch.setattr(
        system_status,
        "get_recording",
        lambda recording_id, settings: {
            "id": recording_id,
            "source_filename": "meeting.mp3",
            "pipeline_stage": "speaker_turns",
        },
    )
    started = system_status._active_job_snapshot(_settings(tmp_path))  # noqa: SLF001
    assert started["active_detail"] == "meeting.mp3 · Speaker Turns"

    def _list_jobs_queued(*, settings, status, limit, offset):
        del settings, limit, offset
        if status == "started":
            return [], 0
        return [queued_row], 1

    monkeypatch.setattr(system_status, "list_jobs", _list_jobs_queued)
    queued = system_status._active_job_snapshot(_settings(tmp_path))  # noqa: SLF001
    assert queued["active_detail"] == "Next: meeting.mp3"

    monkeypatch.setattr(
        system_status,
        "list_jobs",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("db unavailable")),
    )
    failed = system_status._active_job_snapshot(_settings(tmp_path))  # noqa: SLF001
    assert failed["error"] == "db unavailable"

    monkeypatch.setattr(system_status, "list_jobs", _list_jobs_started)
    monkeypatch.setattr(
        system_status,
        "get_recording",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("lookup unavailable")),
    )
    started_lookup_failed = system_status._active_job_snapshot(_settings(tmp_path))  # noqa: SLF001
    assert started_lookup_failed["error"] is None
    assert started_lookup_failed["active_detail"] == "rec-started-1 · Precheck"

    monkeypatch.setattr(system_status, "list_jobs", _list_jobs_queued)
    queued_lookup_failed = system_status._active_job_snapshot(_settings(tmp_path))  # noqa: SLF001
    assert queued_lookup_failed["error"] is None
    assert queued_lookup_failed["active_detail"] == "Next: rec-queued-1"


def test_active_runtime_metadata_prefers_diarization_metadata(tmp_path):
    settings = _settings(tmp_path)
    assert system_status._active_runtime_metadata(settings, recording_id="") == {}  # noqa: SLF001
    assert system_status._active_runtime_metadata(settings, recording_id="rec-missing") == {}  # noqa: SLF001

    derived = settings.recordings_root / "rec-1" / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "diarization_status.json").write_text(
        json.dumps({"mode": "fallback"}),
        encoding="utf-8",
    )
    assert system_status._active_runtime_metadata(settings, recording_id="rec-1") == {  # noqa: SLF001
        "mode": "fallback"
    }

    (derived / "diarization_metadata.json").write_text(
        json.dumps({"effective_device": "cuda"}),
        encoding="utf-8",
    )
    assert system_status._active_runtime_metadata(settings, recording_id="rec-1") == {  # noqa: SLF001
        "effective_device": "cuda"
    }


def test_collect_control_center_runtime_status_healthy_gpu_path(tmp_path, monkeypatch):
    settings = _settings(tmp_path, asr_device="cuda", diarization_device="cuda")
    monkeypatch.setattr(
        system_status,
        "_probe_spark_runtime",
        lambda _settings: {
            "state": "healthy",
            "value": "Online",
            "detail": "dgx.local responded to /v1/models",
            "host": "dgx.local",
            "advertised_models": ["gpt-oss:120b"],
            "model_verified": True,
        },
    )
    monkeypatch.setattr(
        system_status,
        "_active_job_snapshot",
        lambda _settings: {
            "started_total": 1,
            "queued_total": 2,
            "active_job": {"recording_id": "rec-1"},
            "queued_job": {"recording_id": "rec-2"},
            "active_recording": {"id": "rec-1", "source_filename": "meeting.mp3"},
            "active_detail": "meeting.mp3 · Speaker Turns",
            "active_stage": "speaker_turns",
            "error": None,
        },
    )
    monkeypatch.setattr(
        system_status,
        "collect_cuda_runtime_facts",
        lambda: CudaRuntimeFacts(
            is_available=True,
            device_count=1,
            visible_devices=None,
            torch_cuda_version="12.6",
        ),
    )
    monkeypatch.setattr(
        system_status,
        "_probe_nvidia_smi",
        lambda: _nvidia_smi_probe(),
    )

    payload = system_status.collect_control_center_runtime_status(settings)

    assert payload["items"][0]["label"] == "Node status"
    assert payload["items"][0]["value"] == "Busy"
    assert payload["items"][0]["show_dot"] is True
    assert payload["items"][1]["value"] == "GPU busy"
    assert payload["items"][2]["label"] == "LLM:"
    assert payload["items"][2]["detail"] == "dgx.local · configured model is advertised"


def test_collect_control_center_runtime_status_gpu_runtime_detail_branches(tmp_path, monkeypatch):
    settings = _settings(tmp_path, asr_device="cuda", diarization_device="cuda")
    monkeypatch.setattr(
        system_status,
        "_probe_spark_runtime",
        lambda _settings: {
            "state": "healthy",
            "value": "Online",
            "detail": "dgx.local responded to /v1/models",
            "host": "dgx.local",
            "advertised_models": ["gpt-oss:120b"],
            "model_verified": True,
        },
    )
    monkeypatch.setattr(
        system_status,
        "_active_job_snapshot",
        lambda _settings: {
            "started_total": 0,
            "queued_total": 0,
            "active_job": None,
            "queued_job": None,
            "active_recording": None,
            "active_detail": "",
            "active_stage": "",
            "error": None,
        },
    )
    monkeypatch.setattr(
        system_status,
        "collect_cuda_runtime_facts",
        lambda: CudaRuntimeFacts(
            is_available=True,
            device_count=1,
            visible_devices=None,
            torch_cuda_version="12.6",
        ),
    )
    monkeypatch.setattr(
        system_status,
        "_probe_nvidia_smi",
        lambda: _nvidia_smi_probe(
            available=True,
            device_count=1,
            detail="nvidia-smi sees 1 GPU(s) · util up to 0%",
        ),
    )

    payload = system_status.collect_control_center_runtime_status(settings)
    assert payload["items"][1]["detail"].endswith(
        "nvidia-smi sees 1 GPU(s) · util up to 0%"
    )

    monkeypatch.setattr(
        system_status,
        "collect_cuda_runtime_facts",
        lambda: CudaRuntimeFacts(
            is_available=False,
            device_count=0,
            visible_devices=None,
            torch_cuda_version="12.6",
        ),
    )
    payload = system_status.collect_control_center_runtime_status(settings)
    assert payload["items"][1]["detail"].endswith("torch CUDA 12.6")

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(
        system_status,
        "_probe_nvidia_smi",
        lambda: _nvidia_smi_probe(available=True, device_count=2, busy=True),
    )
    payload = system_status.collect_control_center_runtime_status(settings)
    assert payload["items"][1]["value"] == "GPU unavailable"
    assert payload["items"][1]["detail"] == "CUDA_VISIBLE_DEVICES disables GPU visibility for this process"


def test_collect_control_center_runtime_status_runtime_metadata_branches(tmp_path, monkeypatch):
    settings = _settings(tmp_path, asr_device="cuda", diarization_device="cuda")
    monkeypatch.setattr(
        system_status,
        "_probe_spark_runtime",
        lambda _settings: {
            "state": "healthy",
            "value": "Online",
            "detail": "dgx.local responded to /v1/models",
            "host": "dgx.local",
            "advertised_models": ["gpt-oss:120b"],
            "model_verified": True,
        },
    )
    monkeypatch.setattr(
        system_status,
        "_active_job_snapshot",
        lambda _settings: {
            "started_total": 1,
            "queued_total": 0,
            "active_job": {"recording_id": "rec-1"},
            "queued_job": None,
            "active_recording": {"id": "rec-1", "source_filename": "meeting.mp3"},
            "active_detail": "meeting.mp3 · Speaker Turns",
            "active_stage": "speaker_turns",
            "error": None,
        },
    )
    monkeypatch.setattr(
        system_status,
        "collect_cuda_runtime_facts",
        lambda: CudaRuntimeFacts(
            is_available=False,
            device_count=0,
            visible_devices="default",
            torch_cuda_version=None,
        ),
    )
    monkeypatch.setattr(
        system_status,
        "_probe_nvidia_smi",
        lambda: _nvidia_smi_probe(),
    )

    cpu_fallback = system_status.collect_control_center_runtime_status(settings)
    assert cpu_fallback["items"][1]["value"] == "GPU unavailable"
    assert cpu_fallback["items"][1]["tone"] == "offline"

    monkeypatch.setattr(
        system_status,
        "_probe_nvidia_smi",
        lambda: _nvidia_smi_probe(
            available=True,
            device_count=1,
            detail="nvidia-smi sees 1 GPU(s) · util up to 0%",
        ),
    )
    gpu_runtime = system_status.collect_control_center_runtime_status(settings)
    assert gpu_runtime["items"][1]["value"] == "GPU ready"
    assert gpu_runtime["items"][1]["tone"] == "healthy"


def test_collect_control_center_runtime_status_active_llm_and_mixed_target(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    monkeypatch.setattr(
        system_status,
        "_probe_spark_runtime",
        lambda _settings: {
            "state": "healthy",
            "value": "Online",
            "detail": "dgx.local responded to /v1/models",
            "host": "dgx.local",
            "advertised_models": ["other-model"],
            "model_verified": False,
        },
    )
    monkeypatch.setattr(
        system_status,
        "_active_job_snapshot",
        lambda _settings: {
            "started_total": 1,
            "queued_total": 1,
            "active_job": {"recording_id": "rec-1"},
            "queued_job": {"recording_id": "rec-2"},
            "active_recording": {"id": "rec-1", "source_filename": "meeting.mp3"},
            "active_detail": "meeting.mp3 · LLM Chunk 1 Of 2",
            "active_stage": "llm_chunk_1_of_2",
            "error": None,
        },
    )
    monkeypatch.setattr(
        system_status,
        "collect_cuda_runtime_facts",
        lambda: CudaRuntimeFacts(
            is_available=True,
            device_count=1,
            visible_devices=None,
            torch_cuda_version="12.6",
        ),
    )
    monkeypatch.setattr(
        system_status,
        "_probe_nvidia_smi",
        lambda: _nvidia_smi_probe(),
    )

    payload = system_status.collect_control_center_runtime_status(settings)

    assert payload["items"][0]["value"] == "Busy"
    assert payload["items"][1]["value"] == "GPU ready"
    assert payload["items"][2]["tone"] == "degraded"
    assert payload["items"][2]["detail"] == "dgx.local · configured model is not advertised"


def test_collect_control_center_runtime_status_queue_offline_and_scheduler_blocked(
    tmp_path,
    monkeypatch,
):
    settings = _settings(tmp_path)
    monkeypatch.setattr(
        system_status,
        "_probe_spark_runtime",
        lambda _settings: {
            "state": "offline",
            "value": "Offline",
            "detail": "dgx.local timed out",
            "host": "dgx.local",
            "advertised_models": [],
            "model_verified": None,
        },
    )
    monkeypatch.setattr(
        system_status,
        "_active_job_snapshot",
        lambda _settings: {
            "started_total": 0,
            "queued_total": 0,
            "active_job": None,
            "queued_job": None,
            "active_recording": None,
            "active_detail": None,
            "active_stage": "",
            "error": None,
        },
    )
    monkeypatch.setattr(
        system_status,
        "collect_cuda_runtime_facts",
        lambda: CudaRuntimeFacts(
            is_available=False,
            device_count=0,
            visible_devices=None,
            torch_cuda_version=None,
        ),
    )
    monkeypatch.setattr(
        system_status,
        "_probe_nvidia_smi",
        lambda: _nvidia_smi_probe(),
    )

    payload = system_status.collect_control_center_runtime_status(settings)

    assert payload["items"][0]["value"] == "Offline"
    assert payload["items"][1]["value"] == "CPU only"
    assert payload["items"][2]["tone"] == "offline"


def test_collect_control_center_runtime_status_queue_unknown(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    monkeypatch.setattr(
        system_status,
        "_probe_spark_runtime",
        lambda _settings: {
            "state": "degraded",
            "value": "Auth error",
            "detail": "dgx.local rejected the runtime probe",
            "host": "dgx.local",
            "advertised_models": [],
            "model_verified": None,
        },
    )
    monkeypatch.setattr(
        system_status,
        "_active_job_snapshot",
        lambda _settings: {
            "started_total": 0,
            "queued_total": 0,
            "active_job": None,
            "queued_job": None,
            "active_recording": None,
            "active_detail": None,
            "active_stage": "",
            "error": "db unavailable",
        },
    )
    monkeypatch.setattr(
        system_status,
        "collect_cuda_runtime_facts",
        lambda: CudaRuntimeFacts(
            is_available=True,
            device_count=1,
            visible_devices=None,
            torch_cuda_version="12.6",
        ),
    )
    monkeypatch.setattr(
        system_status,
        "_probe_nvidia_smi",
        lambda: _nvidia_smi_probe(),
    )

    payload = system_status.collect_control_center_runtime_status(settings)

    assert payload["items"][0]["value"] == "Unknown"
    assert payload["items"][1]["value"] == "GPU ready"
    assert payload["items"][2]["tone"] == "degraded"


def test_collect_control_center_runtime_status_idle_and_unknown_spark(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    monkeypatch.setattr(
        system_status,
        "_probe_spark_runtime",
        lambda _settings: {
            "state": "unknown",
            "value": "Unknown",
            "detail": "dgx.local responded but /v1/models is unavailable",
            "host": "dgx.local",
            "advertised_models": [],
            "model_verified": None,
        },
    )
    monkeypatch.setattr(
        system_status,
        "_active_job_snapshot",
        lambda _settings: {
            "started_total": 0,
            "queued_total": 0,
            "active_job": None,
            "queued_job": None,
            "active_recording": None,
            "active_detail": None,
            "active_stage": "",
            "error": None,
        },
    )
    monkeypatch.setattr(
        system_status,
        "collect_cuda_runtime_facts",
        lambda: CudaRuntimeFacts(
            is_available=True,
            device_count=1,
            visible_devices=None,
            torch_cuda_version="12.6",
        ),
    )
    monkeypatch.setattr(
        system_status,
        "_probe_nvidia_smi",
        lambda: _nvidia_smi_probe(),
    )

    payload = system_status.collect_control_center_runtime_status(settings)

    assert payload["items"][0]["value"] == "Unknown"
    assert payload["items"][0]["tone"] == "degraded"
