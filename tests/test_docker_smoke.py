import pytest
import subprocess, time, requests, os, signal

IMAGE = os.environ.get("SMOKE_IMAGE", "ghcr.io/alexbomber12/lan-transcriber:latest")

def test_docker_container_runs():
    """Spin up the image and hit the root URL – max 90 s."""
    proc = subprocess.Popen(
        ["docker", "run", "--rm", "-p", "17860:7860", IMAGE],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        for _ in range(90):
            try:
                r = requests.get("http://127.0.0.1:17860/", timeout=1)
                if r.ok:
                    assert "<html" in r.text.lower()
                    return
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(1)
        pytest.fail("UI never came up inside Docker image")
    finally:
        proc.send_signal(signal.SIGINT)   # graceful stop
