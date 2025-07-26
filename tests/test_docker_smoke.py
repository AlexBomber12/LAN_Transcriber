import subprocess, requests, time, os, signal, pytest

IMAGE = os.getenv("SMOKE_IMAGE")
if not IMAGE:
    pytest.skip("SMOKE_IMAGE not supplied; skipping container smoke test", allow_module_level=True)


def test_container_launches():
    proc = subprocess.Popen(
        ["docker", "run", "--rm", "-p", "17860:7860", IMAGE],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    try:
        for _ in range(90):
            try:
                r = requests.get("http://127.0.0.1:17860/openapi.json", timeout=1)
                if r.status_code == 200:
                    return
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(1)
        pytest.fail("UI did not start or returned non-200")
    finally:
        proc.send_signal(signal.SIGINT)

