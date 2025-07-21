import argparse
import base64
from pathlib import Path
import sys
import time
import requests


def wait_health(base_url: str, timeout: int = 120) -> None:
    for _ in range(timeout):
        try:
            r = requests.get(f"{base_url}/healthz", timeout=5)
            if r.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise SystemExit("health check timed out")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--file", required=True)
    args = p.parse_args()

    wait_health(args.base_url)

    if args.file.endswith(".b64"):
        data = base64.b64decode(Path(args.file).read_text())
        files = {"file": ("audio.mp3", data, "audio/mpeg")}
        resp = requests.post(f"{args.base_url}/api/upload", files=files)
    else:
        with open(args.file, "rb") as fh:
            resp = requests.post(
                f"{args.base_url}/api/upload", files={"file": fh}
            )
    resp.raise_for_status()
    job = resp.json()
    job_id = job.get("id")
    if not job_id:
        print("missing job id")
        return 1

    for _ in range(120):
        r = requests.get(f"{args.base_url}/api/job/{job_id}", timeout=5)
        if r.status_code != 200:
            time.sleep(1)
            continue
        data = r.json()
        if data.get("status") == "done":
            if "Hello" in data.get("markdown", ""):
                print("Smoke test passed")
                return 0
            print("Missing expected text")
            return 1
        time.sleep(1)

    print("job timeout")
    return 1


if __name__ == "__main__":
    sys.exit(main())
