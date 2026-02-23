from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = "AuthBenchSecondPhaseCrawler/1.0"


class JSONLWriter:
    def __init__(self, path: Path, *, append: bool = False, initial_count: int = 0):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        self._fh = self.path.open(mode, encoding="utf-8")
        self.count = initial_count

    def write(self, row: dict) -> None:
        self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.count += 1

    def close(self) -> None:
        self._fh.close()


def normalize_whitespace(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            yield json.loads(line)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def http_get_bytes(
    url: str,
    *,
    timeout: int = 60,
    retries: int = 3,
    retry_backoff_sec: float = 1.5,
    user_agent: str = DEFAULT_USER_AGENT,
) -> bytes:
    last_exc: Exception | None = None
    req = Request(url, headers={"User-Agent": user_agent, "Accept-Encoding": "identity"})
    for attempt in range(1, retries + 1):
        try:
            with urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (HTTPError, URLError, TimeoutError) as exc:
            last_exc = exc
            if isinstance(exc, HTTPError):
                # Most 4xx responses (e.g., 404/406) are not recoverable.
                if 400 <= exc.code < 500 and exc.code not in {408, 429}:
                    break
            if attempt == retries:
                break
            sleep_for = retry_backoff_sec * attempt
            logger.warning("GET %s failed (%s), retrying in %.1fs", url, exc, sleep_for)
            time.sleep(sleep_for)
    raise RuntimeError(f"Failed to fetch {url}: {last_exc}")


def http_get_text(
    url: str,
    *,
    timeout: int = 60,
    retries: int = 3,
    retry_backoff_sec: float = 1.5,
    user_agent: str = DEFAULT_USER_AGENT,
) -> str:
    raw = http_get_bytes(
        url,
        timeout=timeout,
        retries=retries,
        retry_backoff_sec=retry_backoff_sec,
        user_agent=user_agent,
    )
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace")


def download_file(
    url: str,
    destination: Path,
    *,
    timeout: int = 120,
    retries: int = 3,
    retry_backoff_sec: float = 1.5,
    overwrite: bool = False,
    user_agent: str = DEFAULT_USER_AGENT,
) -> Path:
    if destination.exists() and not overwrite:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": user_agent, "Accept-Encoding": "identity"})
    last_exc: Exception | None = None
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with urlopen(req, timeout=timeout) as resp, tmp_path.open("wb") as fh:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    fh.write(chunk)
            tmp_path.replace(destination)
            return destination
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_exc = exc
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            if isinstance(exc, HTTPError):
                # Most 4xx responses are not recoverable by retry.
                if 400 <= exc.code < 500 and exc.code not in {408, 429}:
                    break
            if attempt == retries:
                break
            sleep_for = retry_backoff_sec * attempt
            logger.warning("Download %s failed (%s), retrying in %.1fs", url, exc, sleep_for)
            time.sleep(sleep_for)
    raise RuntimeError(f"Failed to download {url}: {last_exc}")
