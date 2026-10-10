from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import queue
import threading
import time
from collections import deque
from pathlib import Path


def exe(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise RuntimeError(f"{name} not found in PATH. Install FFmpeg and add it to PATH.")
    return found


_HW_DECODE = None


def hw_decode_args() -> list[str]:
    """["-hwaccel", "cuda"] when FFmpeg can decode on an NVIDIA GPU here, else []. Decoding the
    4K HEVC footage on the GPU gives identical frames 1.7-4x faster. HT_GPU_DECODE=0 turns it off."""
    global _HW_DECODE
    if _HW_DECODE is None:
        import os
        _HW_DECODE = False
        if os.environ.get("HT_GPU_DECODE", "1") != "0":
            try:
                import torch
                _HW_DECODE = bool(torch.cuda.is_available())
            except Exception:
                pass
    return ["-hwaccel", "cuda"] if _HW_DECODE else []


def decode_raw(args: list[str]) -> bytes:
    """Run `ffmpeg <hw decode> <args>` and return stdout; retries on the CPU if the GPU path gives nothing."""
    global _HW_DECODE
    hw = hw_decode_args()
    out = subprocess.run([exe("ffmpeg"), *hw, *args], capture_output=True).stdout
    if not out and hw:
        _HW_DECODE = False
        out = subprocess.run([exe("ffmpeg"), *args], capture_output=True).stdout
    return out


def run(cmd: list[str], check=True):
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def probe(path: str) -> dict:
    ffprobe = exe("ffprobe")
    cp = run([
        ffprobe, "-v", "error", "-show_streams", "-show_format",
        "-of", "json", path
    ])
    raw = cp.stdout
    if raw is None:
        raise RuntimeError("ffprobe가 JSON 출력을 반환하지 않았어. FFmpeg/ffprobe 설치 상태와 PATH를 확인해줘.")
    raw = str(raw).strip()
    if not raw:
        err = str(cp.stderr or "").strip()
        raise RuntimeError("ffprobe 출력이 비어 있어 영상 정보를 읽지 못했어." + (f"\n{err}" if err else ""))
    try:
        data = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        preview = raw[:300].replace("\n", " ")
        raise RuntimeError(f"ffprobe JSON 파싱에 실패했어: {exc}\n출력: {preview}") from exc
    if not isinstance(data, dict):
        raise RuntimeError("ffprobe가 올바른 JSON 객체를 반환하지 않았어.")
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    audio = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), {})
    fmt = data.get("format", {})
    return {
        "duration": float(fmt.get("duration") or video.get("duration") or 0),
        "size": int(fmt.get("size") or 0),
        "bit_rate": int(fmt.get("bit_rate") or 0),
        "codec": video.get("codec_name"),
        "profile": video.get("profile"),
        "pix_fmt": video.get("pix_fmt"),
        "width": int(video.get("width") or 0),
        "height": int(video.get("height") or 0),
        "fps": _fps(video.get("avg_frame_rate") or video.get("r_frame_rate")),
        "audio_codec": audio.get("codec_name"),
        "audio_channels": audio.get("channels"),
        "audio_rate": audio.get("sample_rate"),
    }


def _fps(rate):
    try:
        a, b = str(rate).split("/")
        return float(a) / float(b) if float(b) else 0
    except Exception:
        return 0


def supports_token(kind: str, token: str) -> bool:
    ffmpeg = exe("ffmpeg")
    arg = "-decoders" if kind == "decoder" else "-encoders"
    try:
        cp = subprocess.run(
            [ffmpeg, "-hide_banner", arg],
            capture_output=True, text=True, timeout=5,
            encoding="utf-8", errors="replace",
        )
        return token in ((cp.stdout or "") + (cp.stderr or ""))
    except (subprocess.TimeoutExpired, OSError):
        return False


def _parse_ffmpeg_clock(value: str) -> float:
    """Parse FFmpeg out_time (HH:MM:SS.microseconds) into seconds."""
    try:
        parts = str(value).strip().split(":")
        if len(parts) != 3:
            return 0.0
        h = float(parts[0]); m = float(parts[1]); sec = float(parts[2])
        return h * 3600.0 + m * 60.0 + sec
    except Exception:
        return 0.0


def _parse_speed(value: str) -> float:
    """Parse FFmpeg speed value such as '7.42x'."""
    try:
        text = str(value or "").strip().lower().rstrip("x")
        if text in ("", "n/a"):
            return 0.0
        return max(0.0, float(text))
    except Exception:
        return 0.0


def _run_ffmpeg_with_progress(
    cmd: list[str],
    duration: float,
    progress_callback=None,
    attempt_name: str = "",
    startup_timeout: float = 30.0,
    stall_timeout: float = 60.0,
    should_cancel=None,
):
    """Run FFmpeg and emit real progress without pipe deadlocks.

    FFmpeg stderr is merged into stdout so a noisy decoder cannot fill the
    stderr pipe and freeze the process. A reader thread feeds a Queue, allowing
    the main thread to emit heartbeats and enforce startup/stall watchdogs on
    Windows as well as POSIX.
    """
    # Make the command non-interactive and ask FFmpeg to publish progress often.
    cmd = list(cmd)
    if "-nostdin" not in cmd:
        cmd.insert(1, "-nostdin")
    if "-stats_period" not in cmd:
        # Must be a global option; inserting just before -progress is safe.
        try:
            idx = cmd.index("-progress")
            cmd[idx:idx] = ["-stats_period", "0.5"]
        except ValueError:
            pass

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    q: queue.Queue[str | None] = queue.Queue()
    tail = deque(maxlen=120)

    def _reader():
        try:
            if proc.stdout is not None:
                for line in proc.stdout:
                    q.put(line)
        finally:
            q.put(None)

    reader = threading.Thread(target=_reader, daemon=True, name=f"ffmpeg-reader-{attempt_name}")
    reader.start()

    started = time.monotonic()
    last_progress_wall = started
    first_media_progress = False
    stream_ended = False
    data = {}
    last_heartbeat = 0.0
    forced_error = None

    while True:
        now = time.monotonic()
        line = None
        try:
            line = q.get(timeout=0.35)
        except queue.Empty:
            pass

        if line is None:
            # None can mean queue timeout OR reader EOF. Distinguish with state.
            if not q.empty():
                continue
            if proc.poll() is not None:
                stream_ended = True
        else:
            stripped = line.strip()
            if stripped:
                tail.append(stripped)
            if "=" in stripped:
                key, value = stripped.split("=", 1)
                data[key] = value
                if key == "progress":
                    out_time = _parse_ffmpeg_clock(data.get("out_time", "0"))
                    try:
                        frame = int(float(data.get("frame", 0) or 0))
                    except Exception:
                        frame = 0
                    try:
                        fps = float(data.get("fps", 0) or 0)
                    except Exception:
                        fps = 0.0
                    speed = _parse_speed(data.get("speed", "0"))
                    percent = min(100.0, max(0.0, out_time / duration * 100.0)) if duration > 0 else 0.0
                    eta = ((duration - out_time) / speed) if duration > out_time and speed > 0 else None

                    if out_time > 0.001 or frame > 0:
                        first_media_progress = True
                        last_progress_wall = now

                    if progress_callback:
                        progress_callback({
                            "attempt": attempt_name,
                            "frame": frame,
                            "fps": fps,
                            "speed": speed,
                            "out_time": out_time,
                            "duration": duration,
                            "percent": percent,
                            "eta_seconds": max(0.0, eta) if eta is not None else None,
                            "progress": value,
                            "elapsed_seconds": now - started,
                        })
                    data = {}

        # Heartbeat keeps the UI alive even before FFmpeg produces its first frame.
        if progress_callback and now - last_heartbeat >= 1.0 and proc.poll() is None:
            last_heartbeat = now
            phase = "encoding" if first_media_progress else "initializing"
            progress_callback({
                "attempt": attempt_name,
                "frame": int(float(data.get("frame", 0) or 0)) if str(data.get("frame", "")).strip() else 0,
                "fps": 0.0,
                "speed": 0.0,
                "out_time": 0.0,
                "duration": duration,
                "percent": 0.0,
                "eta_seconds": None,
                "progress": "heartbeat",
                "phase": phase,
                "elapsed_seconds": now - started,
                "message": (
                    f"{attempt_name} 초기화 중 · {int(now-started)}초"
                    if not first_media_progress else
                    f"{attempt_name} 처리 중"
                ),
            })

        # Hardware backends can exist in FFmpeg's list but still hang on startup
        # because of driver/runtime/codec support. Do not wait forever.
        if proc.poll() is None and not first_media_progress and now - started > startup_timeout:
            forced_error = f"{attempt_name}가 {int(startup_timeout)}초 안에 첫 프레임을 처리하지 못해 자동 중단했어."
            try:
                proc.kill()
            except Exception:
                pass

        if proc.poll() is None and first_media_progress and now - last_progress_wall > stall_timeout:
            forced_error = f"{attempt_name} 진행이 {int(stall_timeout)}초 동안 멈춰 자동 중단했어."
            try:
                proc.kill()
            except Exception:
                pass

        if should_cancel and proc.poll() is None and should_cancel():
            forced_error = "취소됨"
            try:
                proc.kill()
            except Exception:
                pass

        if proc.poll() is not None and (stream_ended or q.empty()):
            break

    try:
        rc = proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        rc = proc.wait()

    output_tail = "\n".join(tail)
    if forced_error:
        output_tail = forced_error + ("\n" + output_tail if output_tail else "")
        if rc == 0:
            rc = -9
    return rc, output_tail

def create_proxy(src: str, dst: str, width=1280, progress_callback=None, duration: float | None = None) -> dict:
    """Create browser-friendly H.264 proxy with real FFmpeg progress.

    Always 8-bit 4:2:0: browsers can't play 10-bit H.264 (High 10), and h264_nvenc
    refuses 10-bit input outright, so a 10-bit phone video must be converted here.

    The callback receives actual encoded media time, frame count, speed, percent,
    and ETA. NVIDIA paths are tried first and CPU x264 is the fallback.
    """
    ffmpeg = exe("ffmpeg")
    dstp = Path(dst)
    dstp.parent.mkdir(parents=True, exist_ok=True)
    duration = float(duration or 0.0)
    if duration <= 0:
        try:
            duration = float(probe(src).get("duration") or 0.0)
        except Exception:
            duration = 0.0

    attempts = []
    if supports_token("decoder", "hevc_cuvid") and supports_token("encoder", "h264_nvenc"):
        attempts.append(("NVIDIA decode + NVENC", [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-c:v", "hevc_cuvid", "-resize", f"{width}x720", "-i", src,
            "-map", "0:v:0", "-map", "0:a?", "-pix_fmt", "yuv420p",
            "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "25",
            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
            "-progress", "pipe:1", "-nostats", str(dstp)
        ]))

    if supports_token("encoder", "h264_nvenc"):
        attempts.append(("NVENC", [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", src,
            "-vf", f"scale={width}:-2,format=yuv420p", "-map", "0:v:0", "-map", "0:a?",
            "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "25",
            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
            "-progress", "pipe:1", "-nostats", str(dstp)
        ]))

    attempts.append(("CPU x264", [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", src,
        "-vf", f"scale={width}:-2,format=yuv420p", "-map", "0:v:0", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "25",
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
        "-progress", "pipe:1", "-nostats", str(dstp)
    ]))

    errors = []
    for attempt_index, (attempt_name, cmd) in enumerate(attempts, start=1):
        try:
            if dstp.exists():
                dstp.unlink()
        except Exception:
            pass
        if progress_callback:
            progress_callback({
                "attempt": attempt_name,
                "attempt_index": attempt_index,
                "attempt_count": len(attempts),
                "frame": 0,
                "fps": 0.0,
                "speed": 0.0,
                "out_time": 0.0,
                "duration": duration,
                "percent": 0.0,
                "eta_seconds": None,
                "progress": "starting",
            })
        startup_timeout = 15.0 if ("NVIDIA" in attempt_name or "NVENC" in attempt_name) else 45.0
        rc, stderr = _run_ffmpeg_with_progress(
            cmd, duration, progress_callback, attempt_name,
            startup_timeout=startup_timeout, stall_timeout=90.0
        )
        if rc == 0 and dstp.exists() and dstp.stat().st_size > 0:
            return {"path": str(dstp), "command": cmd, "attempt": attempt_name}
        errors.append(f"[{attempt_name}]\n{stderr[-2000:]}")
        if progress_callback:
            progress_callback({
                "attempt": attempt_name,
                "attempt_index": attempt_index,
                "attempt_count": len(attempts),
                "percent": 0.0,
                "out_time": 0.0,
                "duration": duration,
                "eta_seconds": None,
                "progress": "fallback",
                "message": f"{attempt_name} 실패, 다음 방식으로 재시도",
            })

    raise RuntimeError("Proxy generation failed:\n" + "\n---\n".join(errors))

def _ass_time(sec: float) -> str:
    sec = max(0.0, sec)
    h = int(sec // 3600)
    sec -= h * 3600
    m = int(sec // 60)
    sec -= m * 60
    return f"{h}:{m:02d}:{sec:05.2f}"


def make_score_ass(points: list[dict], settings: dict, path: str):
    team_a = " / ".join(settings.get("team_a") or ["Team A"])
    team_b = " / ".join(settings.get("team_b") or ["Team B"])
    position_mode = settings.get("scoreboard_position_mode", "court")

    header = """[Script Info]\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\nScaledBorderAndShadow: yes\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Score,Malgun Gothic,36,&H00FFFFFF,&H000000FF,&H00151515,&HC0000000,-1,0,0,0,100,100,0,0,3,2,0,7,48,48,48,1\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"""
    lines = [header]
    edited_cursor = 0.0
    for p in points:
        dur = max(0.01, float(p["end"]) - float(p["start"]))
        st = p.get("score_before") or {}
        team_a_near = st.get("team_a_end", "near") == "near"
        server = st.get("server_id", "")
        sa = "● " if str(server).startswith("A") else "  "
        sb = "● " if str(server).startswith("B") else "  "
        a = f"{sa}{team_a}    {st.get('sets_a',0)}  {st.get('set_games_a',0)}  {st.get('point_a','0')}"
        b = f"{sb}{team_b}    {st.get('sets_b',0)}  {st.get('set_games_b',0)}  {st.get('point_b','0')}"
        if position_mode == "court" and not team_a_near:
            a, b = b, a
        text = (a + r"\N" + b).replace("{", "(").replace("}", ")")
        lines.append(f"Dialogue: 0,{_ass_time(edited_cursor)},{_ass_time(edited_cursor+dur)},Score,,0,0,0,,{text}\n")
        edited_cursor += dur
    Path(path).write_text("".join(lines), encoding="utf-8-sig")


def export_gameplay(project: dict, out_path: str, with_scoreboard=False) -> str:
    ffmpeg = exe("ffmpeg")
    points = [p for p in project.get("points", []) if float(p.get("end",0)) > float(p.get("start",0))]
    if not points:
        raise RuntimeError("No valid point segments to export.")
    file_map = {f["id"]: f for f in project.get("files", [])}
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="hello_tennis_") as td:
        segs = []
        use_nvenc = supports_token("encoder", "h264_nvenc")
        for i, p in enumerate(points):
            src = file_map[p["file_id"]]["path"]
            seg = Path(td) / f"seg_{i:05d}.mp4"
            dur = float(p["end"]) - float(p["start"])
            base = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                    "-ss", f"{float(p['start']):.3f}", "-i", src, "-t", f"{dur:.3f}"]
            attempts = []
            hw = hw_decode_args()
            if hw:  # decode on the GPU first (same frames, faster on 4K HEVC)
                hb = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", *hw] + base[5:]
                if use_nvenc:
                    attempts.append(hb + ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "20",
                                          "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(seg)])
                attempts.append(hb + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                                      "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(seg)])
            if use_nvenc:
                attempts.append(base + ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "20",
                                        "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(seg)])
            attempts.append(base + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                                    "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(seg)])
            errors = []
            ok = False
            for cmd in attempts:
                cp = subprocess.run(cmd, capture_output=True, text=True)
                if cp.returncode == 0 and seg.exists() and seg.stat().st_size > 0:
                    ok = True
                    break
                errors.append(cp.stderr[-2000:])
            if not ok:
                raise RuntimeError("Segment export failed:\n" + "\n---\n".join(errors))
            segs.append(seg)

        concat_list = Path(td) / "concat.txt"
        concat_list.write_text("".join(f"file '{s.as_posix()}'\n" for s in segs), encoding="utf-8")
        gameplay_tmp = Path(td) / "gameplay.mp4"
        cp = subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                             "-f", "concat", "-safe", "0", "-i", str(concat_list),
                             "-c", "copy", str(gameplay_tmp)], capture_output=True, text=True)
        if cp.returncode != 0:
            raise RuntimeError(f"Concat failed: {cp.stderr[-2000:]}")

        if not with_scoreboard:
            shutil.copy2(gameplay_tmp, outp)
            return str(outp)

        attempts = []
        try:  # scoreboard drawn per point (analysis/scoreboard.py), overlaid as an image sequence
            from PIL import Image  # noqa: F401
            from analysis import scoreboard
            from analysis.render_style import Fonts
            meta = file_map[points[0]["file_id"]].get("metadata") or {}
            size = (int(meta.get("width") or 1920), int(meta.get("height") or 1080))
            fonts = Fonts()
            lines = ["ffconcat version 1.0"]
            for i, p in enumerate(points):
                name = f"score_{i:05d}.png"
                scoreboard.render(p, project.get("settings", {}), size, fonts).save(Path(td) / name)
                lines += [f"file '{name}'", f"duration {max(0.01, float(p['end']) - float(p['start'])):.4f}"]
            lines.append(f"file 'score_{len(points) - 1:05d}.png'")
            (Path(td) / "scores.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            base = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(gameplay_tmp),
                    "-f", "concat", "-safe", "0", "-i", str(Path(td) / "scores.txt"),
                    "-filter_complex", "[1:v]format=rgba[s];[0:v][s]overlay=0:0:eof_action=pass:shortest=0[v]",
                    "-map", "[v]", "-map", "0:a?"]
        except ImportError:  # no Pillow: plain ASS subtitles (needs FFmpeg with libass)
            ass = Path(td) / "score.ass"
            make_score_ass(points, project.get("settings", {}), str(ass))
            vf = f"subtitles='{str(ass).replace(chr(92), '/')}'"
            base = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(gameplay_tmp), "-vf", vf]
        if use_nvenc:
            attempts.append(base + ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "20", "-c:a", "copy", str(outp)])
        attempts.append(base + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "copy", str(outp)])
        errors = []
        for cmd in attempts:
            cp = subprocess.run(cmd, capture_output=True, text=True)
            if cp.returncode == 0 and outp.exists() and outp.stat().st_size > 0:
                return str(outp)
            errors.append(cp.stderr[-2000:])
        raise RuntimeError("Broadcast render failed.\n" + "\n---\n".join(errors))
