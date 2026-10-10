from __future__ import annotations

import json
import mimetypes
import os
import re
import sys
import importlib.util
import subprocess
import threading
import time
import traceback
import urllib.parse
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from core.project import new_project, file_entry, point_entry, recompute, save_project, load_project
from core.media import probe, create_proxy, export_gameplay

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
CACHE = ROOT / "cache"
PROJECTS = ROOT / "projects"
EXPORTS = ROOT / "exports"
CONFIG = ROOT / "config"
for d in (CACHE, PROJECTS, EXPORTS, CONFIG):
    d.mkdir(parents=True, exist_ok=True)

STATE_LOCK = threading.RLock()
PROJECT = new_project()
CURRENT_PROJECT_PATH = PROJECTS / "current_project.json"
APP_VERSION = "0.7.0-court-views"
PROGRESS_API_VERSION = 4


PROXY_JOBS_LOCK = threading.RLock()
PROXY_JOBS: dict[str, dict] = {}

AI_JOB_LOCK = threading.RLock()
AI_JOB: dict = {
    "status": "idle",
    "message": "AI 패키지 상태 확인 전",
    "percent": 0.0,
    "latest_line": "",
    "error": None,
}
AI_MODULES = {
    "numpy": "numpy",
    "opencv-python": "cv2",
    "ultralytics": "ultralytics",
    "ruptures": "ruptures",
}


SHOT_JOBS_LOCK = threading.RLock()
SHOT_JOBS: dict[str, dict] = {}
GEMINI_CONFIG_PATH = CONFIG / "gemini.json"


def _gemini_config() -> dict:
    cfg = {}
    if GEMINI_CONFIG_PATH.exists():
        try:
            cfg = json.loads(GEMINI_CONFIG_PATH.read_text(encoding="utf-8")) or {}
        except Exception:
            cfg = {}
    env_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    from analysis.gemini_client import DEFAULT_MODEL
    return {
        "api_key": env_key or cfg.get("api_key") or "",
        "key_source": "env" if env_key else ("file" if cfg.get("api_key") else None),
        "model": cfg.get("model") or DEFAULT_MODEL,
    }


def _public_gemini_config() -> dict:
    cfg = _gemini_config()
    key = cfg["api_key"]
    return {
        "has_key": bool(key),
        "key_hint": f"…{key[-4:]}" if len(key) >= 8 else "",
        "key_source": cfg["key_source"],
        "model": cfg["model"],
    }


def _shot_result_path(fid: str) -> Path:
    return CACHE / f"{fid}_shots.json"


def _public_shot_job(fid: str) -> dict:
    with SHOT_JOBS_LOCK:
        job = SHOT_JOBS.get(fid)
        if job:
            out = {k: v for k, v in job.items() if not k.startswith("_")}
            started = job.get("_started_monotonic")
            if started:
                out["elapsed_seconds"] = max(0.0, time.monotonic() - float(started))
            return out
    if _shot_result_path(fid).exists():
        return {"file_id": fid, "status": "done", "percent": 100.0, "message": "분석 결과 있음"}
    return {"file_id": fid, "status": "idle", "percent": 0.0}


def _update_shot_job(fid: str, **updates):
    with SHOT_JOBS_LOCK:
        SHOT_JOBS.setdefault(fid, {"file_id": fid}).update(updates)


def _save_shot_result(fid: str, result: dict):
    out = _shot_result_path(fid)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    with STATE_LOCK:
        live = get_file(fid)
        if live:
            live["shot_analysis"] = {"path": str(out), "analyzed_at": result["analyzed_at"],
                                     "counts": result["summary"]["counts"]}
            auto_save()
    msg = f"샷 분석 완료 · 타구 {result['summary']['total_shots']}개"
    if result.get("prompt_variant") == "hits":
        msg += " (공 추적 타구별 판정)"
    if result.get("failed_chunks"):
        msg += f" · 실패 구간 {result['failed_chunks']}개"
    _update_shot_job(fid, status="done", percent=100.0, message=msg)


def _run_and_free(fn, *args):
    """Run a worker, then hand the GPU memory its models used back to the system, so other
    programs on the PC (and the next job) are not squeezed into slow shared memory."""
    try:
        fn(*args)
    finally:
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _shot_worker(fid: str, options: dict):
    from analysis.shot_analysis import ShotAnalysisJob, Cancelled

    def cancelled():
        with SHOT_JOBS_LOCK:
            return bool(SHOT_JOBS.get(fid, {}).get("_cancel"))

    def report(info: dict):
        _update_shot_job(fid, status="running", **info)

    try:
        cfg = _gemini_config()
        with STATE_LOCK:
            snapshot = json.loads(json.dumps(PROJECT, ensure_ascii=False))
        f = next((x for x in snapshot.get("files", []) if x.get("id") == fid), None)
        if not f:
            raise RuntimeError("file not found")
        # v2 prompt (rally rules + measured serve impacts as anchors) scored better on hand labels
        # (3.8 Flash, points 1-6: 13-15/21 hits with 1-2 invented vs v1 10-11/21).
        if options.get("prompt") == "hits":
            from analysis.hit_shots import HitShotJob
            result = HitShotJob(project=snapshot, file_entry=f, api_key=cfg["api_key"], model=cfg["model"],
                                cache_dir=CACHE, report=report, cancelled=cancelled).run()
            _save_shot_result(fid, result)
            return
        variant = options.get("prompt") if options.get("prompt") in ("v1", "v2") else "v2"
        anchors = []
        measured = CACHE / f"{fid}_serves.json"
        if variant == "v2" and measured.exists():
            try:
                anchors = [{"t": s["contact_t"], "end": s.get("server_end"), "call": s.get("call")}
                           for s in json.loads(measured.read_text(encoding="utf-8")).get("serves", [])]
            except Exception:
                anchors = []
        job = ShotAnalysisJob(
            project=snapshot, file_entry=f, api_key=cfg["api_key"], model=cfg["model"],
            scope=options.get("scope") or "points", quality=options.get("quality") or "auto",
            high_res=bool(options.get("high_res")), cache_dir=CACHE,
            report=report, cancelled=cancelled, prompt_variant=variant, serve_anchors=anchors,
        )
        _save_shot_result(fid, job.run())
    except Cancelled:
        _update_shot_job(fid, status="cancelled", message="샷 분석 취소됨")
    except Exception as exc:
        traceback.print_exc()
        _update_shot_job(fid, status="error", message="샷 분석 실패", error=str(exc))


SERVE_JOBS_LOCK = threading.RLock()
SERVE_JOBS: dict[str, dict] = {}


def _serves_path(fid: str) -> Path:
    return CACHE / f"{fid}_serves.json"


def _public_serve_job(fid: str) -> dict:
    with SERVE_JOBS_LOCK:
        job = SERVE_JOBS.get(fid)
        if not job:
            return {"file_id": fid, "status": "done" if _serves_path(fid).exists() else "idle", "percent": 0.0}
        out = {k: v for k, v in job.items() if not k.startswith("_")}
        out["elapsed_seconds"] = max(0.0, time.monotonic() - float(job.get("_started_monotonic") or time.monotonic()))
        return out


def _update_serve_job(fid: str, **updates):
    with SERVE_JOBS_LOCK:
        SERVE_JOBS.setdefault(fid, {"file_id": fid}).update(updates)


def _serve_worker(fid: str):
    from analysis.serve_measure_job import measure_file, apply_measurements, summarize
    from analysis.shot_analysis import Cancelled

    def cancelled():
        with SERVE_JOBS_LOCK:
            return bool(SERVE_JOBS.get(fid, {}).get("_cancel"))

    try:
        with STATE_LOCK:
            snapshot = json.loads(json.dumps(PROJECT, ensure_ascii=False))
        f = next((x for x in snapshot.get("files", []) if x.get("id") == fid), None)
        if not f:
            raise RuntimeError("file not found")
        shots_fp = _shot_result_path(fid)
        shots = json.loads(shots_fp.read_text(encoding="utf-8")) if shots_fp.exists() else None
        measured = measure_file(snapshot, f, ROOT, shots,
                                report=lambda d: _update_serve_job(fid, status="running", **d),
                                cancelled=cancelled)
        rally = measured.pop("rally_hits", None)
        _serves_path(fid).write_text(json.dumps(measured, ensure_ascii=False, indent=1), encoding="utf-8")
        if rally is not None:
            from analysis.hit_shots import hits_path
            rally["measured_at"] = measured["measured_at"]
            hits_path(CACHE, fid).write_text(json.dumps(rally, ensure_ascii=False, indent=1), encoding="utf-8")
        if shots is not None:
            apply_measurements(shots, measured, snapshot.get("settings") or {})
            shots_fp.write_text(json.dumps(shots, ensure_ascii=False, indent=1), encoding="utf-8")
        sm = summarize(measured["serves"])
        msg = f"서브 실측 완료 · {sm['count']}개"
        if sm["count"]:
            msg += f" · 평균 {sm['avg']} · 최고 {sm['max']} km/h"
        if rally is not None:
            msg += f" · 랠리 타구 {rally['stats']['rally_hits']}개"
        _update_serve_job(fid, status="done", percent=100.0, message=msg)
    except Cancelled:
        _update_serve_job(fid, status="cancelled", message="서브 실측 취소됨")
    except Exception as exc:
        traceback.print_exc()
        _update_serve_job(fid, status="error", message="서브 실측 실패", error=str(exc))


DASH_JOBS_LOCK = threading.RLock()
DASH_JOBS: dict[str, dict] = {}


def _public_dash_job(fid: str) -> dict:
    with DASH_JOBS_LOCK:
        job = DASH_JOBS.get(fid)
        if not job:
            return {"file_id": fid, "status": "idle", "percent": 0.0}
        out = {k: v for k, v in job.items() if not k.startswith("_")}
        out["elapsed_seconds"] = max(0.0, time.monotonic() - float(job.get("_started_monotonic") or time.monotonic()))
        return out


def _update_dash_job(fid: str, **updates):
    with DASH_JOBS_LOCK:
        DASH_JOBS.setdefault(fid, {"file_id": fid}).update(updates)


def _dash_worker(fid: str, options: dict):
    from analysis.dashboard_video import export_dashboard
    from analysis.shot_analysis import Cancelled

    def cancelled():
        with DASH_JOBS_LOCK:
            return bool(DASH_JOBS.get(fid, {}).get("_cancel"))

    try:
        with STATE_LOCK:
            snapshot = json.loads(json.dumps(PROJECT, ensure_ascii=False))
        f = next((x for x in snapshot.get("files", []) if x.get("id") == fid), None)
        if not f:
            raise RuntimeError("file not found")
        if _shot_result_path(fid).exists():
            result = json.loads(_shot_result_path(fid).read_text(encoding="utf-8"))
        else:
            from analysis.serve_measure_job import serves_only_result
            result = serves_only_result(json.loads(_serves_path(fid).read_text(encoding="utf-8")),
                                        f, snapshot.get("settings") or {})
        stem = re.sub(r"[^0-9A-Za-z._-]+", "_", Path(f.get("name") or fid).stem) or fid
        name = f"{stem}_Dashboard.mp4"
        info = export_dashboard(
            snapshot, f, result, EXPORTS / name, CACHE / "dash_work" / fid,
            quality=options.get("quality") or "auto",
            report=lambda d: _update_dash_job(fid, status="running", **d),
            cancelled=cancelled, root=ROOT,
            ball=bool(options.get("ball")), pose=bool(options.get("pose")),
        )
        _update_dash_job(fid, status="done", percent=100.0, url=f"/download/{name}", path=info["path"],
                         message=f"대시보드 영상 완료 · {int(info['duration'] // 60)}분 {int(info['duration'] % 60)}초 · 타구 {info['events']}개"
                                 + "".join(f" · {n}" for n in info.get("notes") or []))
    except Cancelled:
        _update_dash_job(fid, status="cancelled", message="대시보드 영상 내보내기 취소됨")
    except Exception as exc:
        traceback.print_exc()
        _update_dash_job(fid, status="error", message="대시보드 영상 내보내기 실패", error=str(exc))


def _ai_dependency_status() -> dict:
    installed = []
    missing = []
    for package, module in AI_MODULES.items():
        try:
            ok = importlib.util.find_spec(module) is not None
        except Exception:
            ok = False
        (installed if ok else missing).append(package)
    return {
        "ready": len(missing) == 0,
        "installed": installed,
        "missing": missing,
        "python": sys.executable,
    }


def _public_ai_job() -> dict:
    dep = _ai_dependency_status()
    with AI_JOB_LOCK:
        job = {k: v for k, v in AI_JOB.items() if not k.startswith("_")}
        started = AI_JOB.get("_started_monotonic")
    job["dependencies"] = dep
    if started:
        elapsed = max(0.0, time.monotonic() - float(started))
    else:
        elapsed = 0.0
    job["elapsed_seconds"] = elapsed
    pct = float(job.get("percent") or 0.0)
    if 1.0 < pct < 100.0 and elapsed > 1.0:
        job["eta_seconds"] = elapsed * (100.0 - pct) / pct
    else:
        job["eta_seconds"] = 0.0 if pct >= 100.0 else None
    if dep["ready"] and job.get("status") not in ("running", "error"):
        job.update(status="done", percent=100.0, message="Formation AI 준비됨", eta_seconds=0.0)
    return job


def _set_ai_job(**updates):
    with AI_JOB_LOCK:
        AI_JOB.update(updates)
        AI_JOB["updated_at"] = time.time()


def _ai_install_worker():
    req = ROOT / "requirements-ai.txt"
    if not req.exists():
        _set_ai_job(status="error", error="requirements-ai.txt를 찾을 수 없어.", message="AI 설치 실패")
        return

    specs = []
    for raw in req.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        specs.append(line)

    # Map requirement spec -> import/module name using the package prefix.
    items = []
    for spec in specs:
        package = re.split(r"[<>=!~\[]", spec, 1)[0].strip()
        module = AI_MODULES.get(package, package.replace("-", "_"))
        items.append((spec, package, module))

    total = max(1, len(items))
    _set_ai_job(
        status="running",
        percent=0.0,
        message="AI 패키지 설치 준비 중...",
        latest_line="",
        error=None,
        package_index=0,
        package_count=total,
        completed_packages=0,
        current_package="",
        download_percent=None,
        _started_monotonic=time.monotonic(),
    )

    def module_ready(module: str) -> bool:
        try:
            return importlib.util.find_spec(module) is not None
        except Exception:
            return False

    completed = 0
    try:
        for idx, (spec, package, module) in enumerate(items, start=1):
            if module_ready(module):
                completed += 1
                pct = completed / total * 100.0
                _set_ai_job(
                    status="running",
                    percent=pct,
                    package_index=idx,
                    package_count=total,
                    completed_packages=completed,
                    current_package=package,
                    download_percent=100.0,
                    message=f"AI 패키지 확인 완료 · {package}",
                    latest_line="이미 설치되어 있어.",
                )
                continue

            base = (idx - 1) / total * 100.0
            span = 1.0 / total * 100.0
            _set_ai_job(
                status="running",
                percent=base,
                package_index=idx,
                package_count=total,
                completed_packages=completed,
                current_package=package,
                download_percent=0.0,
                message=f"AI 패키지 설치 중 · {package}",
                latest_line="pip 시작 중...",
            )

            cmd = [
                sys.executable, "-m", "pip", "install", spec,
                "--disable-pip-version-check", "--progress-bar", "raw"
            ]
            proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            stage_fraction = 0.05
            if proc.stdout is not None:
                for raw in proc.stdout:
                    line = raw.strip()
                    if not line:
                        continue
                    lower = line.lower()
                    dl_pct = None
                    # pip --progress-bar raw emits lines like: "Progress 12345 of 67890"
                    m = re.search(r"Progress\s+(\d+)\s+of\s+(\d+)", line, flags=re.I)
                    if m:
                        cur = int(m.group(1)); tot = max(1, int(m.group(2)))
                        dl_pct = min(100.0, cur / tot * 100.0)
                        stage_fraction = 0.10 + 0.65 * (dl_pct / 100.0)
                    elif lower.startswith("collecting "):
                        stage_fraction = max(stage_fraction, 0.08)
                    elif "downloading " in lower or "using cached " in lower:
                        stage_fraction = max(stage_fraction, 0.15)
                    elif "installing collected packages" in lower:
                        stage_fraction = max(stage_fraction, 0.82)
                    elif "successfully installed" in lower or "requirement already satisfied" in lower:
                        stage_fraction = 0.98
                    pct = min(base + span * stage_fraction, base + span * 0.99)
                    _set_ai_job(
                        status="running",
                        percent=pct,
                        package_index=idx,
                        package_count=total,
                        completed_packages=completed,
                        current_package=package,
                        download_percent=dl_pct,
                        message=f"AI 패키지 설치 중 · {package}",
                        latest_line=line[-300:],
                    )
            rc = proc.wait()
            if rc != 0 or not module_ready(module):
                raise RuntimeError(f"{package} 설치 실패 (pip 종료 코드 {rc})")
            completed += 1
            pct = completed / total * 100.0
            _set_ai_job(
                status="running",
                percent=pct,
                package_index=idx,
                package_count=total,
                completed_packages=completed,
                current_package=package,
                download_percent=100.0,
                message=f"AI 패키지 설치 완료 · {package}",
                latest_line=f"{package} 설치 완료",
            )

        dep = _ai_dependency_status()
        if dep["ready"]:
            _set_ai_job(
                status="done", percent=100.0, message="Formation AI 설치 완료",
                latest_line="모든 AI 패키지 설치가 완료됐어.", error=None,
                completed_packages=total, package_index=total, package_count=total,
                current_package="", download_percent=100.0,
            )
        else:
            missing = ", ".join(dep["missing"])
            _set_ai_job(status="error", percent=completed / total * 100.0,
                        message="AI 설치 실패", error=f"미설치: {missing or '확인 필요'}")
    except Exception as exc:
        _set_ai_job(status="error", message="AI 설치 실패", error=str(exc))

def _public_proxy_job(fid: str) -> dict:
    with PROXY_JOBS_LOCK:
        job = PROXY_JOBS.get(fid)
        if not job:
            f = get_file(fid)
            if f and f.get("proxy") and Path(f["proxy"]).exists():
                return {
                    "file_id": fid,
                    "status": "done",
                    "percent": 100.0,
                    "out_time": float(f.get("metadata", {}).get("duration") or 0.0),
                    "duration": float(f.get("metadata", {}).get("duration") or 0.0),
                    "frame": 0,
                    "fps": 0.0,
                    "speed": 0.0,
                    "eta_seconds": 0.0,
                    "elapsed_seconds": 0.0,
                    "attempt": "existing proxy",
                    "message": "프록시 준비됨",
                }
            return {"file_id": fid, "status": "idle", "percent": 0.0}
        return {k: v for k, v in job.items() if not k.startswith("_")}


def _update_proxy_job(fid: str, **updates):
    with PROXY_JOBS_LOCK:
        job = PROXY_JOBS.setdefault(fid, {"file_id": fid})
        job.update(updates)
        started = float(job.get("_started_monotonic") or time.monotonic())
        job["elapsed_seconds"] = max(0.0, time.monotonic() - started)
        job["updated_at"] = time.time()


def _proxy_worker(fid: str, src: str, dst: str, duration: float):
    _update_proxy_job(fid, status="running", message="FFmpeg 시작 중...", percent=0.0)

    def on_progress(info: dict):
        percent = float(info.get("percent") or 0.0)
        eta = info.get("eta_seconds")
        # If FFmpeg has not reported speed yet, derive a conservative ETA from
        # elapsed wall-clock time and actual completed media percentage.
        with PROXY_JOBS_LOCK:
            job = PROXY_JOBS.get(fid, {})
            elapsed = max(0.0, time.monotonic() - float(job.get("_started_monotonic") or time.monotonic()))
        if eta is None and 0.5 < percent < 100.0 and elapsed > 0:
            eta = elapsed * (100.0 - percent) / percent

        state = info.get("progress")
        attempt = info.get("attempt") or "FFmpeg"
        message = info.get("message")
        if not message:
            if state == "starting":
                message = f"프록시 생성 준비 중 · {attempt}"
            elif state == "fallback":
                message = f"{attempt} 실패 · 다른 방식으로 재시도 중"
            else:
                message = f"프록시 생성 중 · {attempt}"
        _update_proxy_job(
            fid,
            status="running",
            message=message,
            percent=percent,
            out_time=float(info.get("out_time") or 0.0),
            duration=float(info.get("duration") or duration or 0.0),
            frame=int(info.get("frame") or 0),
            fps=float(info.get("fps") or 0.0),
            speed=float(info.get("speed") or 0.0),
            eta_seconds=float(eta) if eta is not None else None,
            attempt=attempt,
            attempt_index=info.get("attempt_index"),
            attempt_count=info.get("attempt_count"),
        )

    try:
        result = create_proxy(src, dst, progress_callback=on_progress, duration=duration)
        with STATE_LOCK:
            f = get_file(fid)
            if f:
                f["proxy"] = str(dst)
                auto_save()
        _update_proxy_job(
            fid,
            status="done",
            message="프록시 생성 완료",
            percent=100.0,
            out_time=duration,
            duration=duration,
            eta_seconds=0.0,
            attempt=result.get("attempt"),
            command=result.get("command"),
        )
    except Exception as exc:
        _update_proxy_job(
            fid,
            status="error",
            message=str(exc),
            error=str(exc),
            eta_seconds=None,
        )


def jdump(obj):
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")


def get_file(fid):
    for f in PROJECT.get("files", []):
        if f.get("id") == fid:
            return f
    return None


def auto_save():
    with STATE_LOCK:
        recompute(PROJECT)
        save_project(PROJECT, CURRENT_PROJECT_PATH)


def native_select_files():
    if os.name == "nt":
        try:
            import subprocess
            ps = r"""[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; Add-Type -AssemblyName System.Windows.Forms; $d=New-Object System.Windows.Forms.OpenFileDialog; $d.Multiselect=$true; $d.Filter='Video files|*.mp4;*.mov;*.mkv;*.m4v;*.avi|All files|*.*'; if($d.ShowDialog() -eq 'OK'){ $d.FileNames | ForEach-Object { Write-Output $_ } }"""
            cp = subprocess.run(["powershell", "-NoProfile", "-STA", "-Command", ps], capture_output=True, text=True, encoding="utf-8", errors="replace")
            return [x.strip() for x in cp.stdout.splitlines() if x.strip()]
        except Exception:
            pass
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        paths = filedialog.askopenfilenames(
            title="테니스 영상 선택",
            filetypes=[("Video files", "*.mp4 *.mov *.mkv *.m4v *.avi"), ("All files", "*.*")],
        )
        root.destroy()
        return list(paths)
    except Exception:
        return []


def native_select_project():
    if os.name == "nt":
        try:
            import subprocess
            ps = r"""[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; Add-Type -AssemblyName System.Windows.Forms; $d=New-Object System.Windows.Forms.OpenFileDialog; $d.Multiselect=$false; $d.Filter='JSON project|*.json|All files|*.*'; if($d.ShowDialog() -eq 'OK'){ Write-Output $d.FileName }"""
            cp = subprocess.run(["powershell", "-NoProfile", "-STA", "-Command", ps], capture_output=True, text=True, encoding="utf-8", errors="replace")
            return cp.stdout.strip()
        except Exception:
            pass
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        path = filedialog.askopenfilename(title="프로젝트 JSON 열기", filetypes=[("JSON", "*.json")])
        root.destroy()
        return path
    except Exception:
        return ""



class Handler(BaseHTTPRequestHandler):
    server_version = f"HelloTennisVideoMVP/{APP_VERSION}"

    def log_message(self, fmt, *args):
        print("[HTTP]", fmt % args)

    def _json(self, obj, status=200):
        body = jdump(obj)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, exc, status=500):
        traceback.print_exc()
        self._json({"ok": False, "error": str(exc)}, status)

    def _read_json(self):
        n = int(self.headers.get("Content-Length", "0") or 0)
        if n <= 0:
            return {}
        raw_bytes = self.rfile.read(n)
        if not raw_bytes:
            return {}
        raw = raw_bytes.decode("utf-8", errors="replace").strip()
        if not raw or raw == "null":
            return {}
        try:
            data = json.loads(raw)
        except (TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"요청 JSON을 읽지 못했어: {exc}") from exc
        return data if isinstance(data, dict) else {}

    def do_GET(self):
        try:
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            if path == "/api/state":
                with STATE_LOCK:
                    recompute(PROJECT)
                    return self._json({"ok": True, "project": PROJECT})
            if path == "/api/health":
                return self._json({"ok": True, "version": APP_VERSION})
            if path == "/api/version":
                return self._json({"ok": True, "version": APP_VERSION, "progress_api": PROGRESS_API_VERSION, "shot_api": 1})
            if path == "/api/shot_config":
                return self._json({"ok": True, "config": _public_gemini_config()})
            if path.startswith("/api/shot_status/"):
                fid = path.rsplit("/", 1)[-1]
                return self._json({"ok": True, "job": _public_shot_job(fid)})
            if path == "/api/tracking_status":
                from analysis.serve_measure_job import tracking_ready
                return self._json({"ok": True, "tracking": tracking_ready()})
            if path.startswith("/api/serve_measure_status/"):
                return self._json({"ok": True, "job": _public_serve_job(path.rsplit("/", 1)[-1])})
            if path.startswith("/api/serves_result/"):
                fp = _serves_path(path.rsplit("/", 1)[-1])
                return self._json({"ok": True, "result": json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else None})
            if path.startswith("/api/export_dashboard_status/"):
                return self._json({"ok": True, "job": _public_dash_job(path.rsplit("/", 1)[-1])})
            if path.startswith("/api/shot_label_eval/"):
                fid = path.rsplit("/", 1)[-1]
                fp = _shot_result_path(fid)
                if not fp.exists():
                    return self._json({"ok": True, "eval": None, "reason": "AI 샷 분석 결과가 없어."})
                from analysis.label_eval import evaluate
                res = json.loads(fp.read_text(encoding="utf-8"))
                # Gemini is scored on its own events; in per-hit mode the tracked serves are part of the method
                own_only = res.get("prompt_variant") != "hits"
                events = [e for e in res.get("events", []) if e.get("file_id") == fid
                          and not (own_only and "added_by_tracking" in (e.get("fixes") or []))]
                with STATE_LOCK:
                    labels = [l for l in PROJECT.get("shot_labels", []) if l.get("file_id") == fid]
                    points = [x for x in PROJECT.get("points", []) if x.get("file_id") == fid]
                if not labels:
                    return self._json({"ok": True, "eval": None, "reason": "타구 라벨이 아직 없어."})
                return self._json({"ok": True, "eval": evaluate(labels, events, points)})
            if path.startswith("/api/shot_result/"):
                fid = path.rsplit("/", 1)[-1]
                fp = _shot_result_path(fid)
                if not fp.exists():
                    return self._json({"ok": True, "result": None})
                return self._json({"ok": True, "result": json.loads(fp.read_text(encoding="utf-8"))})
            if path == "/api/ai_status":
                return self._json({"ok": True, "job": _public_ai_job()})
            if path.startswith("/api/proxy_status/"):
                fid = path.rsplit("/", 1)[-1]
                if not get_file(fid):
                    return self._json({"ok": False, "error": "file not found"}, 404)
                return self._json({"ok": True, "job": _public_proxy_job(fid)})
            if path.startswith("/media/"):
                fid = path.split("/")[-1]
                f = get_file(fid)
                if not f or not f.get("proxy"):
                    return self._json({"ok": False, "error": "proxy not found"}, 404)
                return self._serve_file(Path(f["proxy"]), range_support=True)
            if path.startswith("/download/"):
                name = Path(path.split("/download/", 1)[1]).name
                fp = EXPORTS / name
                return self._serve_file(fp, download=True, range_support=True)
            if path == "/":
                return self._serve_file(WEB / "index.html")
            fp = (WEB / path.lstrip("/")).resolve()
            if WEB.resolve() not in fp.parents and fp != WEB.resolve():
                return self.send_error(403)
            if fp.exists() and fp.is_file():
                return self._serve_file(fp)
            self.send_error(404)
        except Exception as e:
            self._error(e)

    def do_POST(self):
        try:
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path

            if path == "/api/select_files":
                paths = native_select_files()
                if not paths:
                    return self._json({"ok": True, "added": [], "message": "No files selected"})
                added = self._add_paths(paths)
                return self._json({"ok": True, "added": added})

            if path == "/api/add_paths":
                data = self._read_json()
                paths = data.get("paths") or []
                added = self._add_paths(paths)
                return self._json({"ok": True, "added": added})

            if path.startswith("/api/proxy/"):
                fid = path.rsplit("/", 1)[-1]
                f = get_file(fid)
                if not f:
                    return self._json({"ok": False, "error": "file not found"}, 404)
                with PROXY_JOBS_LOCK:
                    existing = PROXY_JOBS.get(fid)
                    if existing and existing.get("status") == "running":
                        return self._json({"ok": True, "started": False, "job": _public_proxy_job(fid)})
                    PROXY_JOBS[fid] = {
                        "file_id": fid,
                        "status": "queued",
                        "percent": 0.0,
                        "out_time": 0.0,
                        "duration": float(f.get("metadata", {}).get("duration") or 0.0),
                        "frame": 0,
                        "fps": 0.0,
                        "speed": 0.0,
                        "eta_seconds": None,
                        "elapsed_seconds": 0.0,
                        "attempt": None,
                        "message": "프록시 작업 대기 중...",
                        "_started_monotonic": time.monotonic(),
                    }
                dst = CACHE / f"{fid}_proxy.mp4"
                thread = threading.Thread(
                    target=_proxy_worker,
                    args=(fid, f["path"], str(dst), float(f.get("metadata", {}).get("duration") or 0.0)),
                    daemon=True,
                    name=f"proxy-{fid}",
                )
                thread.start()
                return self._json({"ok": True, "started": True, "job": _public_proxy_job(fid)}, 202)


            if path == "/api/install_ai":
                dep = _ai_dependency_status()
                if dep["ready"]:
                    _set_ai_job(status="done", percent=100.0, message="Formation AI 준비됨", latest_line="이미 설치되어 있어.", error=None)
                    return self._json({"ok": True, "started": False, "job": _public_ai_job()})
                with AI_JOB_LOCK:
                    if AI_JOB.get("status") == "running":
                        return self._json({"ok": True, "started": False, "job": _public_ai_job()}, 202)
                    AI_JOB.update(status="queued", percent=0.0, message="AI 설치 작업 대기 중...", latest_line="", error=None,
                                  package_index=0, package_count=len(AI_MODULES), completed_packages=0,
                                  current_package="", download_percent=None, _started_monotonic=time.monotonic())
                thread = threading.Thread(target=_ai_install_worker, daemon=True, name="ai-installer")
                thread.start()
                return self._json({"ok": True, "started": True, "job": _public_ai_job()}, 202)

            if path.startswith("/api/analyze_formation/"):
                dep = _ai_dependency_status()
                if not dep["ready"]:
                    return self._json({
                        "ok": False,
                        "error": "Formation AI가 아직 설치되지 않았어. 먼저 ‘AI 기능 설치’를 눌러줘.",
                        "missing": dep["missing"],
                        "code": "AI_DEPENDENCIES_MISSING",
                    }, 409)
                fid = path.rsplit("/", 1)[-1]
                f = get_file(fid)
                if not f:
                    return self._json({"ok": False, "error": "file not found"}, 404)
                from analysis.formation_baseline import analyze
                out = CACHE / f"{fid}_formation.json"
                result = analyze(f, str(out))
                f["formation_analysis"] = str(out)
                auto_save()
                return self._json({
                    "ok": True,
                    "output": str(out),
                    "feature_count": len(result.get("features", [])),
                    "change_points": result.get("change_points", []),
                })

            if path == "/api/shot_config":
                data = self._read_json()
                cfg = {}
                if GEMINI_CONFIG_PATH.exists():
                    try:
                        cfg = json.loads(GEMINI_CONFIG_PATH.read_text(encoding="utf-8")) or {}
                    except Exception:
                        cfg = {}
                if "api_key" in data:
                    key = str(data.get("api_key") or "").strip()
                    if key:
                        cfg["api_key"] = key
                    else:
                        cfg.pop("api_key", None)
                if data.get("model"):
                    cfg["model"] = re.sub(r"[^0-9A-Za-z._-]", "", str(data["model"]))
                GEMINI_CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
                return self._json({"ok": True, "config": _public_gemini_config()})

            if path.startswith("/api/shot_analysis/"):
                fid = path.rsplit("/", 1)[-1]
                f = get_file(fid)
                if not f:
                    return self._json({"ok": False, "error": "file not found"}, 404)
                if not _gemini_config()["api_key"]:
                    return self._json({"ok": False, "code": "NO_API_KEY",
                                       "error": "Gemini API 키가 필요해. ‘API 키 설정’에서 입력해줘."}, 409)
                data = self._read_json()
                if data.get("scope") == "points" and not any(p.get("file_id") == fid for p in PROJECT.get("points", [])):
                    return self._json({"ok": False, "error": "이 영상에 라벨된 포인트가 없어. 범위를 ‘영상 전체’로 바꾸거나 포인트를 먼저 찍어줘."}, 400)
                with SHOT_JOBS_LOCK:
                    existing = SHOT_JOBS.get(fid)
                    if existing and existing.get("status") in ("queued", "running"):
                        return self._json({"ok": True, "started": False, "job": _public_shot_job(fid)})
                    SHOT_JOBS[fid] = {"file_id": fid, "status": "queued", "percent": 0.0,
                                      "message": "샷 분석 준비 중...", "_started_monotonic": time.monotonic()}
                threading.Thread(target=_run_and_free, args=(_shot_worker, fid, data), daemon=True, name=f"shots-{fid}").start()
                return self._json({"ok": True, "started": True, "job": _public_shot_job(fid)}, 202)

            if path.startswith("/api/serve_measure/"):
                fid = path.rsplit("/", 1)[-1]
                f = get_file(fid)
                if not f:
                    return self._json({"ok": False, "error": "file not found"}, 404)
                from analysis.serve_measure_job import tracking_ready
                tr = tracking_ready()
                if not tr["ready"]:
                    return self._json({"ok": False, "code": "AI_DEPENDENCIES_MISSING",
                                       "error": f"공 추적에 필요한 패키지가 없어 ({', '.join(tr['missing'])}). 먼저 ‘AI 기능 설치’를 눌러줘."}, 409)
                if len((f.get("calibration") or {}).get("points") or []) < 4:
                    return self._json({"ok": False, "error": "코트 4점 캘리브레이션을 먼저 해줘."}, 400)
                with SERVE_JOBS_LOCK:
                    job = SERVE_JOBS.get(fid)
                    if job and job.get("status") in ("queued", "running"):
                        return self._json({"ok": True, "started": False, "job": _public_serve_job(fid)})
                    SERVE_JOBS[fid] = {"file_id": fid, "status": "queued", "percent": 0.0,
                                       "message": "서브 실측 준비 중...", "_started_monotonic": time.monotonic()}
                threading.Thread(target=_run_and_free, args=(_serve_worker, fid), daemon=True, name=f"serve-{fid}").start()
                return self._json({"ok": True, "started": True, "job": _public_serve_job(fid)}, 202)

            if path.startswith("/api/serve_measure_cancel/"):
                fid = path.rsplit("/", 1)[-1]
                with SERVE_JOBS_LOCK:
                    job = SERVE_JOBS.get(fid)
                    if job and job.get("status") in ("queued", "running"):
                        job["_cancel"] = True
                        job["message"] = "취소 중..."
                return self._json({"ok": True, "job": _public_serve_job(fid)})

            if path.startswith("/api/export_dashboard/"):
                fid = path.rsplit("/", 1)[-1]
                if not get_file(fid):
                    return self._json({"ok": False, "error": "file not found"}, 404)
                if not _shot_result_path(fid).exists() and not _serves_path(fid).exists():
                    return self._json({"ok": False, "error": "먼저 AI 샷 분석이나 서브 실측을 실행해줘."}, 400)
                data = self._read_json()
                with DASH_JOBS_LOCK:
                    job = DASH_JOBS.get(fid)
                    if job and job.get("status") in ("queued", "running"):
                        return self._json({"ok": True, "started": False, "job": _public_dash_job(fid)})
                    DASH_JOBS[fid] = {"file_id": fid, "status": "queued", "percent": 0.0,
                                      "message": "대시보드 영상 준비 중...", "_started_monotonic": time.monotonic()}
                threading.Thread(target=_run_and_free, args=(_dash_worker, fid, data), daemon=True, name=f"dash-{fid}").start()
                return self._json({"ok": True, "started": True, "job": _public_dash_job(fid)}, 202)

            if path.startswith("/api/export_dashboard_cancel/"):
                fid = path.rsplit("/", 1)[-1]
                with DASH_JOBS_LOCK:
                    job = DASH_JOBS.get(fid)
                    if job and job.get("status") in ("queued", "running"):
                        job["_cancel"] = True
                        job["message"] = "취소 중..."
                return self._json({"ok": True, "job": _public_dash_job(fid)})

            if path.startswith("/api/shot_cancel/"):
                fid = path.rsplit("/", 1)[-1]
                with SHOT_JOBS_LOCK:
                    job = SHOT_JOBS.get(fid)
                    if job and job.get("status") in ("queued", "running"):
                        job["_cancel"] = True
                        job["message"] = "취소 중..."
                return self._json({"ok": True, "job": _public_shot_job(fid)})

            if path == "/api/settings":
                data = self._read_json()
                with STATE_LOCK:
                    PROJECT["settings"].update(data)
                    auto_save()
                return self._json({"ok": True, "project": PROJECT})

            if path.startswith("/api/calibration_refine/"):
                fid = path.rsplit("/", 1)[-1]
                f = get_file(fid)
                if not f:
                    return self._json({"ok": False, "error": "file not found"}, 404)
                from analysis.court_refine import refine_calibration
                cal = refine_calibration(f)
                with STATE_LOCK:
                    f["calibration"] = cal
                    auto_save()
                return self._json({"ok": True, "calibration": cal})

            if path.startswith("/api/calibration_revert/"):
                fid = path.rsplit("/", 1)[-1]
                f = get_file(fid)
                if not f:
                    return self._json({"ok": False, "error": "file not found"}, 404)
                cal = f.get("calibration") or {}
                if cal.get("manual_points"):
                    with STATE_LOCK:
                        f["calibration"] = {"points": cal["manual_points"], "order": cal.get("order")}
                        auto_save()
                return self._json({"ok": True, "calibration": f.get("calibration")})

            if path.startswith("/api/calibration/"):
                fid = path.rsplit("/", 1)[-1]
                data = self._read_json()
                points = data.get("points")
                if not isinstance(points, list) or len(points) < 4:
                    return self._json({"ok": False, "error": "At least 4 calibration points required"}, 400)
                f = get_file(fid)
                if not f:
                    return self._json({"ok": False, "error": "file not found"}, 404)
                f["calibration"] = {"points": points[:6], "order": "near-left,near-right,far-right,far-left"}
                auto_save()
                return self._json({"ok": True})

            if path == "/api/shot_labels":
                data = self._read_json()
                fid = data.get("file_id")
                shot = data.get("shot")
                if not get_file(fid):
                    return self._json({"ok": False, "error": "file_id invalid"}, 400)
                from analysis.label_eval import TYPES
                if shot not in TYPES:
                    return self._json({"ok": False, "error": f"unknown shot {shot}"}, 400)
                import uuid
                lab = {"id": uuid.uuid4().hex[:10], "file_id": fid, "t": round(float(data.get("t", 0)), 3),
                       "shot": shot, "side": data.get("side") if data.get("side") in ("near", "far") else None}
                if data.get("tol"):
                    lab["tol"] = min(3.0, max(0.2, float(data["tol"])))
                if shot == "serve" and data.get("spin") in ("flat", "slice", "kick"):
                    lab["spin"] = data["spin"]
                if data.get("note"):
                    lab["note"] = str(data["note"])[:80]
                with STATE_LOCK:
                    PROJECT.setdefault("shot_labels", []).append(lab)
                    PROJECT["shot_labels"].sort(key=lambda l: (l["file_id"], l["t"]))
                    auto_save()
                return self._json({"ok": True, "label": lab, "project": PROJECT})

            m = re.fullmatch(r"/api/shot_labels/([^/]+)/delete", path)
            if m:
                with STATE_LOCK:
                    PROJECT["shot_labels"] = [l for l in PROJECT.get("shot_labels", []) if l["id"] != m.group(1)]
                    auto_save()
                return self._json({"ok": True, "project": PROJECT})

            if path == "/api/points":
                data = self._read_json()
                fid = data.get("file_id")
                start, end = float(data.get("start", 0)), float(data.get("end", 0))
                if not get_file(fid):
                    return self._json({"ok": False, "error": "file_id invalid"}, 400)
                if end <= start:
                    return self._json({"ok": False, "error": "end must be later than start"}, 400)
                p = point_entry(fid, start, end)
                PROJECT["points"].append(p)
                auto_save()
                return self._json({"ok": True, "point": p, "project": PROJECT})

            m = re.fullmatch(r"/api/points/([^/]+)/(winner|start|end|delete)", path)
            if m:
                pid, action = m.group(1), m.group(2)
                data = self._read_json()
                point = next((p for p in PROJECT["points"] if p["id"] == pid), None)
                if not point:
                    return self._json({"ok": False, "error": "point not found"}, 404)
                if action == "winner":
                    val = data.get("winner")
                    point["winner"] = val if val in ("A", "B", None) else None
                elif action == "start":
                    point["start"] = round(float(data["time"]), 3)
                elif action == "end":
                    point["end"] = round(float(data["time"]), 3)
                elif action == "delete":
                    PROJECT["points"].remove(point)
                PROJECT["points"].sort(key=lambda p: (self._file_order(p["file_id"]), p["start"]))
                auto_save()
                return self._json({"ok": True, "project": PROJECT})

            if path == "/api/save":
                data = self._read_json()
                name = re.sub(r"[^0-9A-Za-z가-힣._-]+", "_", data.get("name") or "tennis_project.json")
                if not name.endswith(".json"):
                    name += ".json"
                dst = PROJECTS / name
                save_project(PROJECT, dst)
                return self._json({"ok": True, "path": str(dst)})

            if path == "/api/load":
                selected = native_select_project()
                if not selected:
                    return self._json({"ok": True, "loaded": False})
                loaded = load_project(selected)
                with STATE_LOCK:
                    PROJECT.clear(); PROJECT.update(loaded)
                    auto_save()
                return self._json({"ok": True, "loaded": True, "project": PROJECT})

            if path in ("/api/export_gameplay", "/api/export_broadcast"):
                with_score = path.endswith("broadcast")
                name = "Match_Broadcast.mp4" if with_score else "Match_GameplayOnly.mp4"
                out = EXPORTS / name
                export_gameplay(PROJECT, str(out), with_scoreboard=with_score)
                return self._json({"ok": True, "path": str(out), "url": f"/download/{name}"})

            self._json({"ok": False, "error": "unknown endpoint"}, 404)
        except Exception as e:
            self._error(e)

    def _file_order(self, fid):
        for i, f in enumerate(PROJECT.get("files", [])):
            if f.get("id") == fid:
                return i
        return 999999

    def _add_paths(self, paths):
        added = []
        with STATE_LOCK:
            existing = {f["path"] for f in PROJECT.get("files", [])}
            for raw in paths:
                path = str(Path(raw).expanduser().resolve())
                if path in existing or not Path(path).exists():
                    continue
                meta = probe(path)
                entry = file_entry(path, meta)
                PROJECT["files"].append(entry)
                existing.add(path)
                added.append(entry)
            auto_save()
        return added

    def _serve_file(self, path: Path, download=False, range_support=False):
        if not path.exists() or not path.is_file():
            return self.send_error(404)
        size = path.stat().st_size
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        start, end = 0, size - 1
        status = 200
        if range_support:
            rng = self.headers.get("Range")
            if rng:
                m = re.match(r"bytes=(\d*)-(\d*)", rng)
                if m:
                    if m.group(1): start = int(m.group(1))
                    if m.group(2): end = min(int(m.group(2)), size - 1)
                    if not m.group(1) and m.group(2):
                        length = int(m.group(2)); start = max(0, size - length); end = size - 1
                    status = 206
        length = max(0, end - start + 1)
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        if range_support:
            self.send_header("Accept-Ranges", "bytes")
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.end_headers()
        with path.open("rb") as f:
            f.seek(start)
            remain = length
            while remain > 0:
                chunk = f.read(min(1024 * 1024, remain))
                if not chunk: break
                self.wfile.write(chunk)
                remain -= len(chunk)


def main(host="127.0.0.1", port=8765, open_browser=True):
    if CURRENT_PROJECT_PATH.exists():
        try:
            loaded = load_project(CURRENT_PROJECT_PATH)
            PROJECT.clear(); PROJECT.update(loaded)
        except Exception as e:
            print("Could not restore current project:", e)
    url = f"http://{host}:{port}"
    print("\nHello Tennis Video MVP")
    print("Open:", url)
    print("Press Ctrl+C to stop.\n")
    server = ThreadingHTTPServer((host, port), Handler)
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        auto_save()
        server.server_close()


if __name__ == "__main__":
    main()
