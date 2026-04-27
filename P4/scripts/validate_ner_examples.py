from __future__ import annotations

import argparse
import json
import sys
from urllib import error, request

DEFAULT_BASE_URL = "http://127.0.0.1:5000"
DEFAULT_TIMEOUT_SECONDS = 15
EXAMPLES = [
    "马云在杭州创办了阿里巴巴集团。",
    "腾讯公司在深圳发布了新产品。",
    "张三在上海交通大学参加学术会议。",
    "北京市政府与北京大学合作。",
]


def post_ner(base_url: str, text: str, timeout_seconds: int) -> dict:
    endpoint = base_url.rstrip("/") + "/api/ner"
    payload = json.dumps({"text": text}).encode("utf-8")
    req = request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )

    with request.urlopen(req, timeout=timeout_seconds) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Manual NER validation with real HTTP requests against local backend."
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Backend base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"HTTP timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS})",
    )

    args = parser.parse_args()

    print(f"NER validation base URL: {args.base_url}")
    print("=" * 72)

    any_error = False

    for idx, text in enumerate(EXAMPLES, start=1):
        print(f"[{idx}] 输入: {text}")
        try:
            result = post_ner(args.base_url, text, args.timeout)
        except error.HTTPError as exc:
            any_error = True
            raw = exc.read().decode("utf-8", errors="replace")
            print(f"    HTTP error: {exc.code}")
            print(f"    body: {raw}")
            print("-" * 72)
            continue
        except Exception as exc:  # pragma: no cover - manual script error path
            any_error = True
            print(f"    Request failed: {type(exc).__name__}: {exc}")
            print("-" * 72)
            continue

        success = result.get("success")
        entities = result.get("entities", [])
        print(f"    success: {success}")
        print(f"    entities: {json.dumps(entities, ensure_ascii=False)}")
        print("-" * 72)

    if any_error:
        print("Validation finished with request errors. Check backend status and base URL.")
        return 1

    print("Validation finished. Compare labels with README high-level expectations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
