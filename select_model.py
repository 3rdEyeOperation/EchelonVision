from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request


def _request(url: str, method: str = "GET", payload: dict | None = None) -> dict:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise SystemExit(f"HTTP {exc.code}: {body}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="List or select /Yolo models through the app API")
    parser.add_argument("--host", default="http://127.0.0.1:3000", help="Base URL for the running /Yolo app")
    parser.add_argument("--select", help="Model name to select, e.g. yolov8s_performance.rknn")
    args = parser.parse_args()

    base = args.host.rstrip("/")
    if args.select:
        result = _request(f"{base}/api/models/select", method="POST", payload={"model": args.select})
        print(result.get("message", "model selected"))
        return 0

    data = _request(f"{base}/api/models")
    active = data.get("active_model")
    print(f"Active model: {active or 'none'}")
    for item in data.get("models", []):
        selectable = "yes" if item.get("selectable") else "no"
        note = item.get("note") or ""
        print(f"- {item.get('name')} [{item.get('format')}] selectable={selectable} {note}".rstrip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
