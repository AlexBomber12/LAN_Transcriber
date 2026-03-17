from __future__ import annotations

import math
import os
from pathlib import Path
import shutil
import socket
import struct
import subprocess
import sys
import time
from urllib.parse import parse_qs, urlparse
import wave
import zipfile

import pytest
import requests

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import sync_playwright


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _write_synthetic_wav(path: Path) -> None:
    sample_rate = 16_000
    duration_seconds = 0.35
    total_frames = int(sample_rate * duration_seconds)
    frequency = 440.0
    amplitude = 0.2

    frames = bytearray()
    for idx in range(total_frames):
        angle = 2.0 * math.pi * frequency * (idx / sample_rate)
        sample = int(32767 * amplitude * math.sin(angle))
        frames.extend(struct.pack("<h", sample))

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(bytes(frames))


def _remaining_timeout_ms(deadline: float) -> int:
    remaining_seconds = deadline - time.monotonic()
    if remaining_seconds <= 0:
        raise TimeoutError("Playwright smoke test exceeded deadline")
    return max(1, int(remaining_seconds * 1000))


def _read_process_output(process: subprocess.Popen[str]) -> str:
    if process.stdout is None:
        return ""
    try:
        return process.stdout.read()
    except Exception:
        return ""


def _wait_for_app_ready(*, base_url: str, process: subprocess.Popen[str], deadline: float) -> None:
    health_url = f"{base_url}/healthz/app"
    last_error = "application did not become healthy"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = _read_process_output(process)
            raise RuntimeError(
                f"uvicorn exited early with code {process.returncode}.\nOutput:\n{output}"
            )

        try:
            response = requests.get(health_url, timeout=1.5)
            if response.status_code == 200:
                return
            last_error = f"{health_url} returned HTTP {response.status_code}"
        except requests.RequestException as exc:
            last_error = str(exc)

        time.sleep(0.2)

    raise TimeoutError(f"Timed out waiting for {health_url}: {last_error}")


def _wait_for_tcp_port(*, host: str, port: int, deadline: float) -> None:
    last_error = f"{host}:{port} did not become reachable"
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            try:
                sock.connect((host, port))
                return
            except OSError as exc:
                last_error = str(exc)
        time.sleep(0.2)
    raise TimeoutError(f"Timed out waiting for {host}:{port}: {last_error}")


def _stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def _start_redis_for_smoke(*, deadline: float) -> tuple[str, str | None]:
    configured_url = os.getenv("LAN_REDIS_URL", "").strip()
    if configured_url:
        return configured_url, None

    if shutil.which("docker") is None:
        pytest.skip("Playwright smoke needs LAN_REDIS_URL or local docker for ephemeral Redis")

    redis_port = _find_free_port()
    container_name = f"lan-transcriber-playwright-redis-{os.getpid()}-{redis_port}"
    try:
        subprocess.run(
            [
                "docker",
                "run",
                "--detach",
                "--rm",
                "--name",
                container_name,
                "--publish",
                f"127.0.0.1:{redis_port}:6379",
                "redis:7-alpine",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        _wait_for_tcp_port(host="127.0.0.1", port=redis_port, deadline=deadline)
    except (subprocess.CalledProcessError, TimeoutError):
        _stop_redis_for_smoke(container_name)
        pytest.skip("Playwright smoke could not start ephemeral Redis via docker")
    return f"redis://127.0.0.1:{redis_port}/14", container_name


def _stop_redis_for_smoke(container_name: str | None) -> None:
    if not container_name:
        return
    subprocess.run(
        ["docker", "rm", "--force", container_name],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def test_control_center_embedded_inspector_and_export_zip_smoke(tmp_path: Path) -> None:
    deadline = time.monotonic() + 120
    base_port = _find_free_port()
    base_url = f"http://127.0.0.1:{base_port}"
    redis_url, redis_container = _start_redis_for_smoke(deadline=deadline)

    runtime_root = tmp_path / "runtime"
    recordings_root = runtime_root / "recordings"
    db_path = runtime_root / "db" / "app.db"
    metrics_snapshot_path = runtime_root / "metrics.snap"
    wav_path = tmp_path / "smoke-upload.wav"
    zip_download_path = tmp_path / "export.zip"

    _write_synthetic_wav(wav_path)

    env = os.environ.copy()
    env.update(
        {
            "LAN_ENV": "dev",
            "LAN_DATA_ROOT": str(runtime_root),
            "LAN_RECORDINGS_ROOT": str(recordings_root),
            "LAN_DB_PATH": str(db_path),
            "LAN_PROM_SNAPSHOT_PATH": str(metrics_snapshot_path),
            "LAN_REDIS_URL": redis_url,
            "LLM_MODEL": os.getenv("LLM_MODEL", "test-llm-model"),
        }
    )

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "lan_app.api:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(base_port),
            "--log-level",
            "warning",
        ],
        cwd=str(Path(__file__).resolve().parents[1]),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        _wait_for_app_ready(base_url=base_url, process=process, deadline=deadline)

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            context = browser.new_context(accept_downloads=True)
            try:
                page = context.new_page()

                page.goto(
                    f"{base_url}/",
                    wait_until="networkidle",
                    timeout=_remaining_timeout_ms(deadline),
                )
                page.get_by_role("button", name="Choose files").wait_for(
                    state="visible",
                    timeout=_remaining_timeout_ms(deadline),
                )
                page.wait_for_selector(
                    "#file-input",
                    state="attached",
                    timeout=_remaining_timeout_ms(deadline),
                )
                page.set_input_files("#file-input", str(wav_path))

                page.wait_for_selector(
                    "#control-center-recordings-panel",
                    state="visible",
                    timeout=_remaining_timeout_ms(deadline),
                )
                page.locator("#control-center-recordings-panel").get_by_text(
                    wav_path.name
                ).wait_for(
                    state="visible",
                    timeout=_remaining_timeout_ms(deadline),
                )
                selected_row = page.locator(
                    "#control-center-recordings-panel tbody tr",
                    has_text=wav_path.name,
                ).first
                selected_row.wait_for(
                    state="visible",
                    timeout=_remaining_timeout_ms(deadline),
                )
                href = selected_row.get_attribute("data-select-href")
                assert href
                recording_id = parse_qs(urlparse(href).query).get("selected", [""])[0]
                assert recording_id
                selected_row.click(timeout=_remaining_timeout_ms(deadline))
                page.wait_for_url(
                    f"**/?selected={recording_id}",
                    timeout=_remaining_timeout_ms(deadline),
                )
                page.wait_for_selector(
                    "#control-center-inspector-pane",
                    state="visible",
                    timeout=_remaining_timeout_ms(deadline),
                )
                page.wait_for_selector(
                    "#control-center-inspector-pane [data-testid='recording-inspector-open-full-page']",
                    state="visible",
                    timeout=_remaining_timeout_ms(deadline),
                )
                page.locator(
                    "#control-center-system-bar .control-center-system-label"
                ).filter(has_text="Node status").first.wait_for(
                    state="visible",
                    timeout=_remaining_timeout_ms(deadline),
                )
                assert urlparse(page.url).path == "/"

                page.get_by_test_id("recording-inspector-tab-speakers").click(
                    timeout=_remaining_timeout_ms(deadline),
                )
                page.wait_for_url(
                    f"**/?selected={recording_id}&tab=speakers",
                    timeout=_remaining_timeout_ms(deadline),
                )
                page.wait_for_selector(
                    "#control-center-inspector-pane .tab.active:has-text('Speakers')",
                    state="visible",
                    timeout=_remaining_timeout_ms(deadline),
                )
                assert urlparse(page.url).path == "/"

                page.get_by_test_id("recording-inspector-tab-summary").click(
                    timeout=_remaining_timeout_ms(deadline),
                )
                page.wait_for_url(
                    f"**/?selected={recording_id}&tab=summary",
                    timeout=_remaining_timeout_ms(deadline),
                )
                page.wait_for_selector(
                    "#control-center-inspector-pane .tab.active:has-text('Summary')",
                    state="visible",
                    timeout=_remaining_timeout_ms(deadline),
                )
                assert urlparse(page.url).path == "/"

                page.get_by_test_id("recording-inspector-tab-export").click(
                    timeout=_remaining_timeout_ms(deadline),
                )
                page.wait_for_url(
                    f"**/?selected={recording_id}&tab=export",
                    timeout=_remaining_timeout_ms(deadline),
                )
                page.wait_for_selector(
                    "#control-center-inspector-pane .tab.active:has-text('Export')",
                    state="visible",
                    timeout=_remaining_timeout_ms(deadline),
                )
                assert urlparse(page.url).path == "/"

                page.get_by_test_id("recording-inspector-tab-overview").click(
                    timeout=_remaining_timeout_ms(deadline),
                )
                page.wait_for_url(
                    f"**/?selected={recording_id}",
                    timeout=_remaining_timeout_ms(deadline),
                )
                page.wait_for_selector(
                    "#control-center-inspector-pane .tab.active:has-text('Overview')",
                    state="visible",
                    timeout=_remaining_timeout_ms(deadline),
                )
                assert urlparse(page.url).path == "/"

                page.wait_for_selector(
                    "#control-center-inspector-pane [data-testid='recording-inspector-download-zip']",
                    state="visible",
                    timeout=_remaining_timeout_ms(deadline),
                )
                page.locator("#upload-rows").get_by_text(
                    "No active uploads. New files appear here until they enter the main inbox."
                ).wait_for(
                    state="visible",
                    timeout=_remaining_timeout_ms(deadline),
                )
                page.get_by_test_id("recording-inspector-open-full-page").click(
                    timeout=_remaining_timeout_ms(deadline),
                )
                page.wait_for_url(
                    f"**/recordings/{recording_id}",
                    timeout=_remaining_timeout_ms(deadline),
                )
                page.get_by_test_id("recording-inspector-tab-transcript").click(
                    timeout=_remaining_timeout_ms(deadline),
                )
                page.wait_for_url(
                    f"**/recordings/{recording_id}?tab=transcript",
                    timeout=_remaining_timeout_ms(deadline),
                )
                page.wait_for_selector(
                    ".transcript-section-card",
                    state="visible",
                    timeout=_remaining_timeout_ms(deadline),
                )
                page.get_by_test_id("recording-inspector-tab-export").click(
                    timeout=_remaining_timeout_ms(deadline),
                )
                page.wait_for_url(
                    f"**/recordings/{recording_id}?tab=export",
                    timeout=_remaining_timeout_ms(deadline),
                )
                with page.expect_download(timeout=_remaining_timeout_ms(deadline)) as download_info:
                    page.get_by_test_id("recording-inspector-download-zip").click(
                        timeout=_remaining_timeout_ms(deadline),
                    )
                download = download_info.value
                download.save_as(str(zip_download_path))
            finally:
                context.close()
                browser.close()

        with zipfile.ZipFile(zip_download_path) as archive:
            names = set(archive.namelist())
            assert "manifest.json" in names
            assert "onenote.md" in names
    finally:
        _stop_process(process)
        _stop_redis_for_smoke(redis_container)
