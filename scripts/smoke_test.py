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


def check_endpoint(base_url: str, path: str) -> bool:
    try:
        response = requests.get(f"{base_url}{path}", timeout=5)
    except requests.RequestException as exc:
        print(f"{path} request failed: {exc}")
        return False
    if response.status_code != 200:
        print(f"{path} returned status {response.status_code}")
        return False
    return True


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    args = p.parse_args()

    base_url = args.base_url.rstrip("/")
    wait_health(base_url)

    required_paths = [
        "/healthz/app",
        "/healthz/db",
        "/healthz/redis",
        "/openapi.json",
    ]
    for path in required_paths:
        if not check_endpoint(base_url, path):
            return 1

    print("Smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
