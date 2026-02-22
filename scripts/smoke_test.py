import argparse
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


def wait_endpoint(base_url: str, path: str, timeout: int = 60) -> bool:
    for _ in range(timeout):
        try:
            response = requests.get(f"{base_url}{path}", timeout=5)
            if response.status_code == 200:
                return True
            print(f"{path} returned status {response.status_code}; retrying")
        except requests.RequestException as exc:
            print(f"{path} request failed: {exc}")
        time.sleep(1)
    print(f"{path} did not return 200 within {timeout}s")
    return False


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    # Backward-compatible no-op retained for callers that still pass --file.
    p.add_argument("--file")
    args = p.parse_args()

    base_url = args.base_url.rstrip("/")
    wait_health(base_url)

    required_paths = [
        "/healthz/app",
        "/healthz/db",
        "/healthz/redis",
        "/healthz/worker",
        "/openapi.json",
    ]
    for path in required_paths:
        if not wait_endpoint(base_url, path):
            return 1

    print("Smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
