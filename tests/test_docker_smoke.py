import os
import subprocess
import time

import pytest
import requests

IMAGE = os.getenv("SMOKE_IMAGE")
if not IMAGE:
    pytest.skip("SMOKE_IMAGE not supplied; skipping container smoke test", allow_module_level=True)


def test_container_launches():
    subprocess.run(
        ["docker", "run", "-d", "--rm", "--name", "lan-smoke", "-p", "17860:7860", IMAGE],
        check=True,
        capture_output=True,
        text=True,
    )
    ready = False
    try:
        for _ in range(90):
            for path in ("healthz/app", "openapi.json"):
                try:
                    r = requests.get(f"http://127.0.0.1:17860/{path}", timeout=1)
                    if r.status_code == 200:
                        ready = True
                        break
                except requests.exceptions.ConnectionError:
                    pass
            if ready:
                break
            time.sleep(1)
        if not ready:
            logs = subprocess.run(
                ["docker", "logs", "lan-smoke"],
                check=False,
                capture_output=True,
                text=True,
            )
            pytest.fail(
                "Container did not become ready within 90 seconds.\n"
                f"docker logs:\n{logs.stdout}\n{logs.stderr}"
            )

        dep_check = subprocess.run(
            [
                "docker",
                "exec",
                "lan-smoke",
                "python",
                "-c",
                "import ctranslate2; import faster_whisper; import whisperx; from omegaconf import OmegaConf; OmegaConf.create({'ok': 1})",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if dep_check.returncode != 0:
            pytest.fail(
                "Dependency import check failed inside runtime container.\n"
                f"stdout:\n{dep_check.stdout}\n"
                f"stderr:\n{dep_check.stderr}"
            )
    finally:
        subprocess.run(
            ["docker", "stop", "lan-smoke"],
            check=False,
            capture_output=True,
            text=True,
        )
