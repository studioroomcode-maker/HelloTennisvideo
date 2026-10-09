"""Minimal Gemini REST client (stdlib only).

Ported from TennisApp's services/gemini/videoProxy.ts + videoAnalysis.ts flow:
resumable File API upload -> wait ACTIVE -> generateContent -> delete file.
TennisApp goes through a server proxy to hide the key; this desktop tool runs
locally, so it talks to Google directly with the user's own key.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

API_BASE = os.environ.get("GEMINI_API_BASE", "https://generativelanguage.googleapis.com").rstrip("/")
DEFAULT_MODEL = "gemini-3.8-flash"


class GeminiError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, daily_quota: bool = False):
        super().__init__(message)
        self.status = status
        self.daily_quota = daily_quota  # a per-day quota is spent: retrying today is pointless

    @property
    def retryable(self) -> bool:
        return self.status is None or self.status == 429 or self.status >= 500


class _ProgressReader:
    """File wrapper so http.client streams the upload and we can report bytes sent."""

    def __init__(self, path: Path, callback=None):
        self._f = path.open("rb")
        self.total = path.stat().st_size
        self.sent = 0
        self._cb = callback

    def read(self, size=-1):
        chunk = self._f.read(size if size and size > 0 else 1024 * 1024)
        self.sent += len(chunk)
        if self._cb:
            self._cb(self.sent, self.total)
        return chunk

    def close(self):
        self._f.close()


def _request(method: str, url: str, api_key: str, body=None, headers=None, timeout=600):
    hdrs = {"x-goog-api-key": api_key}
    hdrs.update(headers or {})
    data = body
    if isinstance(body, (dict, list)):
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        hdrs.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=data, method=method, headers=hdrs)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return resp.status, dict(resp.headers), raw
    except urllib.error.HTTPError as exc:
        raw = exc.read() or b""
        detail = raw.decode("utf-8", errors="replace")
        daily = False
        try:
            err = json.loads(detail).get("error", {})
            detail = err.get("message", detail)
            daily = any("PerDay" in str(v.get("quotaId", ""))
                        for d in err.get("details") or [] for v in d.get("violations") or [])
        except Exception:
            pass
        if daily:
            detail = "오늘 쓸 수 있는 요청 수(무료 등급 하루 한도)를 다 썼어. 내일 다시 하거나 유료 등급으로 바꿔줘. " + str(detail)
        raise GeminiError(f"Gemini HTTP {exc.code}: {str(detail)[:400]}", exc.code, daily) from None
    except urllib.error.URLError as exc:
        raise GeminiError(f"Gemini 연결 실패: {exc.reason}") from None
    except TimeoutError:
        raise GeminiError("Gemini 응답 시간 초과") from None


def upload_file(api_key: str, path: str | Path, mime_type: str = "video/mp4", progress=None) -> dict:
    path = Path(path)
    size = path.stat().st_size
    _, headers, _ = _request(
        "POST",
        f"{API_BASE}/upload/v1beta/files",
        api_key,
        body={"file": {"display_name": path.name}},
        headers={
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(size),
            "X-Goog-Upload-Header-Content-Type": mime_type,
        },
        timeout=60,
    )
    upload_url = next((v for k, v in headers.items() if k.lower() == "x-goog-upload-url"), None)
    if not upload_url:
        raise GeminiError("업로드 URL을 받지 못했어 (X-Goog-Upload-URL 없음).")

    reader = _ProgressReader(path, progress)
    try:
        _, _, raw = _request(
            "POST",
            upload_url,
            api_key,
            body=reader,
            headers={
                "Content-Length": str(size),
                "X-Goog-Upload-Offset": "0",
                "X-Goog-Upload-Command": "upload, finalize",
            },
            timeout=1800,
        )
    finally:
        reader.close()
    try:
        return json.loads(raw)["file"]
    except Exception as exc:
        raise GeminiError(f"업로드 응답을 읽지 못했어: {raw[:200]!r}") from exc


def wait_active(api_key: str, file: dict, timeout: float = 600.0, cancelled=None) -> dict:
    deadline = time.monotonic() + timeout
    while file.get("state") != "ACTIVE":
        if file.get("state") == "FAILED":
            raise GeminiError("Gemini가 영상 처리에 실패했어 (state=FAILED).", 400)
        if time.monotonic() > deadline:
            raise GeminiError("Gemini 영상 처리 대기 시간 초과")
        if cancelled and cancelled():
            raise GeminiError("취소됨", 499)
        time.sleep(2.0)
        _, _, raw = _request("GET", f"{API_BASE}/v1beta/{file['name']}", api_key, timeout=60)
        file = json.loads(raw)
    return file


def delete_file(api_key: str, name: str):
    try:
        _request("DELETE", f"{API_BASE}/v1beta/{name}", api_key, timeout=30)
    except Exception:
        pass


def generate_json(api_key: str, model: str, parts: list, schema: dict, *,
                  media_resolution: str | None = None, max_output_tokens: int = 65536) -> dict:
    gen_cfg = {
        "responseMimeType": "application/json",
        "responseSchema": schema,
        "maxOutputTokens": max_output_tokens,
    }
    # Same thinking setting as TennisApp (gemini 3.x only; 2.x rejects thinkingLevel).
    if model.startswith("gemini-3"):
        gen_cfg["thinkingConfig"] = {"thinkingLevel": "high"}
    if media_resolution:
        gen_cfg["mediaResolution"] = media_resolution
    _, _, raw = _request(
        "POST",
        f"{API_BASE}/v1beta/models/{model}:generateContent",
        api_key,
        body={"contents": [{"role": "user", "parts": parts}], "generationConfig": gen_cfg},
        timeout=900,
    )
    data = json.loads(raw)
    cand = (data.get("candidates") or [{}])[0]
    text = next((p.get("text") for p in (cand.get("content") or {}).get("parts", [])
                 if p.get("text") and not p.get("thought")), None)
    if not text:
        reason = cand.get("finishReason") or (data.get("promptFeedback") or {}).get("blockReason")
        raise GeminiError(f"AI 응답이 비어 있어 (finishReason={reason}).")
    return {"json": parse_json_text(text), "usage": data.get("usageMetadata") or {},
            "finish_reason": cand.get("finishReason")}


def parse_json_text(text: str):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else ""
        cleaned = cleaned.rsplit("```", 1)[0]
    try:
        return json.loads(cleaned.strip())
    except json.JSONDecodeError as exc:
        raise GeminiError(f"AI 응답 JSON 파싱 실패 (길이 {len(text)}자). 응답이 잘렸을 수 있어.") from exc
