"""AI shot analysis for match videos (forehand/backhand counts, serve type, speed).

Ported from TennisApp (services/gemini/videoAnalysis.ts + utils/videoSanityCheck.ts)
and adapted to long match footage:

* TennisApp sends one short phone clip and asks Gemini for aggregate counts.
  Match files here are 4K and run for many minutes, so the video is cut into
  chunks (labeled points only when available) and Gemini returns an *event list*
  (one row per racket contact). Counts, serve-type breakdowns and speed stats are
  then computed deterministically here, which is easier to verify and to merge.
* Labeled points carry score state (server_id, which end Team A is on), so serves
  are attributed to the actual server and strokes to Team A/B.

Speeds are visual AI estimates, not radar measurements.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from core.media import exe, _run_ffmpeg_with_progress
from analysis import gemini_client as gemini
from analysis.rally_rules import apply_rally_rules

PROMPT_VERSION = 1
FALLBACK_MODEL = "gemini-3.5-flash"  # used for a chunk only when the chosen model stays overloaded
VIDEO_FPS = 5            # same sampling density TennisApp settled on (VIDEO_ANALYSIS_FPS)
CHUNK_MAX_SEC = 240.0    # ~100k video tokens per request at 5fps
CHUNK_MAX_SEGMENTS = 30
WINDOW_OVERLAP_SEC = 2.0
POINT_PAD_BEFORE = 2.0   # catch the toss before START
POINT_PAD_AFTER = 1.0
AUTO_ORIGINAL_MAX_SEC = 15 * 60  # 4K decode is ~1x realtime; beyond this use the proxy
MAX_WORKERS = 3

SHOTS = ["serve", "forehand", "backhand", "volley", "smash", "lob", "dropshot", "other"]
SPINS = ["topspin", "slice", "flat", "kick", "unknown"]
SERVE_TYPES = ["flat", "slice", "kick", "unknown"]
SERVE_DIRS = ["T", "body", "wide", "unknown"]
RESULTS = ["in", "fault", "ace", "winner", "error", "unknown"]

SHOT_LABELS = {
    "serve": "서브", "forehand": "포핸드", "backhand": "백핸드", "volley": "발리",
    "smash": "스매시", "lob": "로브", "dropshot": "드롭샷", "other": "기타",
}

# Plausible radar-speed ranges (km/h). Outside → speed discarded with a warning.
SPEED_RANGE = {"serve": (50, 240), "default": (15, 200)}


class Cancelled(RuntimeError):
    pass


# ── Segments / chunks ─────────────────────────────────────────────

def build_segments(file_entry: dict, points: list[dict], scope: str) -> list[dict]:
    """Source-time ranges to analyze. `own_*` is the non-overlapping core used for dedupe."""
    duration = float((file_entry.get("metadata") or {}).get("duration") or 0.0)
    fid = file_entry["id"]
    pts = sorted((p for p in points if p.get("file_id") == fid), key=lambda p: p["start"])
    segs: list[dict] = []
    if scope == "points" and pts:
        for idx, p in enumerate(pts):
            st = p.get("score_before") or {}
            raw_start, raw_end = float(p["start"]), float(p["end"])
            start = max(0.0, raw_start - POINT_PAD_BEFORE)
            end = raw_end + POINT_PAD_AFTER
            if duration > 0:
                end = min(duration, end)
            if segs and start < segs[-1]["end"]:
                # Padding overlaps the previous point: split the gap so each second belongs to one point.
                prev = segs[-1]
                cut = max(prev["start"] + 0.5, (prev["raw_end"] + raw_start) / 2)
                prev["end"] = prev["own_end"] = round(cut, 3)
                start = cut
            segs.append({
                "start": round(start, 3), "end": round(end, 3),
                "own_start": round(start, 3), "own_end": round(end, 3),
                "raw_end": raw_end,
                "point_id": p.get("id"), "point_no": idx + 1,
                "server_id": st.get("server_id"), "team_a_end": st.get("team_a_end"),
                "winner": None if p.get("winner_auto") else p.get("winner"),  # a guessed winner is no evidence for the AI
            })
        return [s for s in segs if s["end"] - s["start"] >= 0.5]

    if duration <= 0:
        raise RuntimeError("영상 길이를 알 수 없어. 파일을 다시 불러와줘.")
    step = CHUNK_MAX_SEC - WINDOW_OVERLAP_SEC
    t = 0.0
    while True:
        last = t + CHUNK_MAX_SEC >= duration
        segs.append({
            "start": round(t, 3), "end": round(min(duration, t + CHUNK_MAX_SEC), 3),
            "own_start": round(t, 3), "own_end": round(duration if last else t + step, 3),
            "point_id": None, "point_no": None, "server_id": None, "team_a_end": None,
        })
        if last:
            return segs
        t += step


def group_chunks(segments: list[dict]) -> list[dict]:
    chunks: list[dict] = []
    cur: list[dict] = []
    cur_len = 0.0
    for s in segments:
        length = s["end"] - s["start"]
        if cur and (cur_len + length > CHUNK_MAX_SEC or len(cur) >= CHUNK_MAX_SEGMENTS):
            chunks.append(_make_chunk(cur))
            cur, cur_len = [], 0.0
        cur.append(s)
        cur_len += length
    if cur:
        chunks.append(_make_chunk(cur))
    return chunks


def _make_chunk(segs: list[dict]) -> dict:
    offset = 0.0
    out = []
    for s in segs:
        s = dict(s)
        s["clip_start"] = round(offset, 3)
        offset += s["end"] - s["start"]
        s["clip_end"] = round(offset, 3)
        out.append(s)
    return {"segments": out, "clip_duration": round(offset, 3)}


def clip_to_source(chunk: dict, t_clip: float):
    """Map a clip timestamp back to (source_time, segment) through the concat table."""
    segs = chunk["segments"]
    for s in segs:
        if t_clip < s["clip_end"] or s is segs[-1]:
            local = min(max(0.0, t_clip - s["clip_start"]), s["end"] - s["start"])
            return s["start"] + local, s
    return None, None


# ── Clip extraction ───────────────────────────────────────────────

def pick_source(file_entry: dict, quality: str, analyzed_seconds: float) -> tuple[str, str]:
    original = file_entry.get("path")
    proxy = file_entry.get("proxy")
    has_orig = bool(original and Path(original).exists())
    has_proxy = bool(proxy and Path(proxy).exists())
    if quality == "proxy" and has_proxy:
        return proxy, "proxy"
    if quality == "original" and has_orig:
        return original, "original"
    if has_orig and (analyzed_seconds <= AUTO_ORIGINAL_MAX_SEC or not has_proxy):
        return original, "original"
    if has_proxy:
        return proxy, "proxy"
    raise RuntimeError("원본 영상과 프록시를 모두 찾을 수 없어.")


def extract_clip(src: str, chunk: dict, out: Path, has_audio: bool, progress=None, cancelled=None, crf: int = 23):
    """Cut each segment to its own 720p H.264 file, then join them with the concat demuxer.

    Segments are encoded one at a time (instead of one filter graph with N inputs) so a
    chunk of 30 points never holds 30 4K HEVC decoders in memory. The real duration of
    every piece is measured afterwards and written back into the chunk's clip time map.
    """
    ffmpeg = exe("ffmpeg")
    segs = chunk["segments"]
    total = sum(s["end"] - s["start"] for s in segs) or 1.0
    parts_dir = out.with_suffix("")
    parts_dir.mkdir(parents=True, exist_ok=True)
    done = 0.0
    files = []
    try:
        for k, s in enumerate(segs):
            length = s["end"] - s["start"]
            part = parts_dir / f"p{k:03d}.mp4"
            from core.media import hw_decode_args
            hw = hw_decode_args()
            cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", *hw,
                   "-ss", f"{s['start']:.3f}", "-t", f"{length:.3f}", "-i", src,
                   "-map", "0:v:0", "-vf", "fps=30,scale=-2:720,setsar=1",
                   "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p"]
            if has_audio:
                # Racket-impact sound is a strong cue for counting shots, so audio is kept.
                cmd += ["-map", "0:a:0", "-ac", "1", "-ar", "48000", "-c:a", "aac", "-b:a", "64k"]
            cmd += ["-progress", "pipe:1", str(part)]
            base = done

            def on_progress(info, base=base):
                if progress and info.get("progress") != "heartbeat":
                    progress(min(100.0, (base + float(info.get("out_time") or 0.0)) / total * 100.0))

            rc, tail = _run_ffmpeg_with_progress(
                cmd, length, progress_callback=on_progress, attempt_name="분석 클립 추출",
                startup_timeout=180.0, stall_timeout=180.0, should_cancel=cancelled,
            )
            if rc != 0 and hw and not (cancelled and cancelled()):  # GPU decode failed: once more on the CPU
                cmd = [c for c in cmd if c not in hw]
                rc, tail = _run_ffmpeg_with_progress(
                    cmd, length, progress_callback=on_progress, attempt_name="분석 클립 추출",
                    startup_timeout=180.0, stall_timeout=180.0, should_cancel=cancelled,
                )
            if cancelled and cancelled():
                raise Cancelled()
            if rc != 0 or not part.exists():
                raise RuntimeError(f"분석 클립 추출 실패 (FFmpeg {rc}): {tail[-600:]}")
            files.append(part)
            done += length

        offset = 0.0
        for s, part in zip(segs, files):
            real = _media_duration(part) or (s["end"] - s["start"])
            s["clip_start"] = round(offset, 3)
            offset += real
            s["clip_end"] = round(offset, 3)
            s["end"] = round(min(s["end"], s["start"] + real), 3)
        chunk["clip_duration"] = round(offset, 3)

        if len(files) == 1:
            shutil.move(str(files[0]), str(out))
        else:
            listing = parts_dir / "list.txt"
            listing.write_text("".join(f"file '{f.name}'\n" for f in files), encoding="utf-8")
            cp = subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-f", "concat",
                                 "-safe", "0", "-i", str(listing), "-c", "copy", "-movflags", "+faststart",
                                 str(out)], capture_output=True, text=True, encoding="utf-8", errors="replace")
            if cp.returncode != 0 or not out.exists():
                raise RuntimeError(f"분석 클립 합치기 실패: {(cp.stderr or '')[-600:]}")
    finally:
        shutil.rmtree(parts_dir, ignore_errors=True)


def _media_duration(path) -> float:
    try:
        cp = subprocess.run([exe("ffprobe"), "-v", "error", "-show_entries", "format=duration",
                             "-of", "csv=p=0", str(path)], capture_output=True, text=True, timeout=30)
        return float(cp.stdout.strip() or 0)
    except Exception:
        return 0.0


# ── Prompt / schema (adapted from TennisApp ANALYSIS_PROMPT) ──────

def _mmss(t: float) -> str:
    m = int(t // 60)
    return f"{m:02d}:{t - m * 60:04.1f}"


def player_name(settings: dict, pid: str | None) -> str | None:
    if not pid or len(pid) < 2:
        return None
    team = settings.get("team_a" if pid[0].upper() == "A" else "team_b") or []
    try:
        name = team[int(pid[1:]) - 1]
    except (ValueError, IndexError):
        name = ""
    return f"{pid}({name})" if name else pid


def _clip_time(chunk: dict, t_src: float):
    for s in chunk["segments"]:
        if s["start"] - 0.05 <= t_src <= s["end"] + 0.05:
            return s["clip_start"] + min(max(0.0, t_src - s["start"]), s["end"] - s["start"])
    return None


V2_RULES = [
    "",
    "## 랠리 구조 (반드시 지킬 것)",
    "- 랠리 중 타구는 두 팀이 번갈아 칩니다 (A → B → A ...). 같은 팀이 연속으로 두 번 칠 수 없습니다.",
    "- 서브 임팩트 후 리턴까지 최소 약 0.6초, 그 밖의 타구도 직전 타구 후 0.4초 안에는 나올 수 없습니다.",
    "- 마지막 타구 뒤의 '아웃'·'네트'는 그 타구의 result입니다. 별도 타구로 기록하지 마세요.",
    "- 다음 타구가 있었다는 것은 직전 타구가 들어갔다는 뜻입니다.",
    "",
    "## 복식에서 놓치기 쉬운 타구",
    "- 네트 앞 전위의 발리: 동작이 짧고 작습니다. 네트 근처 선수의 라켓과 공의 방향 전환, 짧고 둔탁한 타구음을 함께 확인하세요.",
    "- 먼 쪽 코트 선수는 화면에서 작습니다. 공의 진행 방향이 바뀌는 순간과 타구음으로 판단하세요.",
    "- 리턴 이후 네트 앞 발리 교환은 0.5~1초 간격으로 빠르게 이어질 수 있습니다.",
    "",
    "## 서브 구질 판단 근거",
    "- slice: 토스가 서버의 라켓 쥔 손 쪽 앞으로 가고, 라켓이 공의 옆면을 깎으며, 공이 옆으로 휘어 바운스 후 옆으로 빠집니다.",
    "- kick: 토스가 머리 위나 뒤쪽, 아래에서 위로 긁어 올리는 스윙, 바운스 후 높게 튀어 오릅니다.",
    "- flat: 회전이 거의 없이 곧게 날아가고 바운스가 낮고 빠릅니다.",
    "- 근거가 보이지 않으면 unknown.",
    "",
    "## 기록 원칙",
    "- 위 구조와 타구음으로 확인되는 타구는 작게 보여도 기록하세요. 공이 오갔는데 중간 타구가 빠지면 번갈아 치기 규칙에 어긋납니다.",
    "- 포인트 사이에 공을 넘겨주는 타구, 연습 스윙, 공 튀기기는 제외. 같은 타구를 두 번 기록하지 마세요.",
]


def build_prompt(chunk: dict, settings: dict, file_entry: dict, variant: str = "v1",
                 serve_anchors: list | None = None, hit_candidates: list | None = None) -> str:
    doubles = (settings.get("mode") or "doubles") == "doubles"
    team_a = " / ".join(x for x in settings.get("team_a") or [] if x) or "Team A"
    team_b = " / ".join(x for x in settings.get("team_b") or [] if x) or "Team B"

    lines = [
        "당신은 SwingVision·Hawk-Eye 수준의 프로 테니스 영상 분석 AI입니다.",
        "이 영상은 실제 경기 촬영본에서 잘라낸 클립입니다. 클립 안에서 일어난 모든 타구(라켓이 공을 친 순간)를",
        "시간 순서대로 한 줄씩 events 배열에 기록하세요. 집계는 하지 마세요 — 집계는 프로그램이 합니다.",
        "",
        "## 경기 정보",
        f"- 경기 유형: {'복식 (코트에 4명)' if doubles else '단식 (코트에 2명)'}",
        f"- Team A: {team_a}",
        f"- Team B: {team_b}",
        "- 코트 규격: 길이 23.77m, 폭 10.97m(복식)/8.23m(단식), 네트~서비스라인 6.40m, 베이스라인~반대편 서비스라인 18.29m",
    ]
    cal = (file_entry.get("calibration") or {}).get("points") or []
    if len(cal) >= 4:
        names = ["가까운 쪽 왼쪽", "가까운 쪽 오른쪽", "먼 쪽 오른쪽", "먼 쪽 왼쪽"]
        corners = ", ".join(f"{n} ({p['x']:.2f}, {p['y']:.2f})" for n, p in zip(names, cal[:4]))
        lines.append(f"- 코트 바깥 코너의 화면 좌표(0~1 정규화, 좌상단 원점): {corners}")

    segs = chunk["segments"]
    lines += ["", "## 클립 구성 (클립 시간 기준)"]
    if len(segs) > 1 or segs[0].get("point_no"):
        lines.append("여러 포인트 구간을 이어 붙인 편집본입니다. 구간 경계에서 장면이 바뀌는 것은 정상입니다.")
    for i, s in enumerate(segs, 1):
        desc = f"{i}. {_mmss(s['clip_start'])}–{_mmss(s['clip_end'])}"
        if s.get("point_no"):
            desc += f" · 포인트 #{s['point_no']}"
        server = player_name(settings, s.get("server_id"))
        if server:
            team = "Team A" if s["server_id"][0].upper() == "A" else "Team B"
            desc += f" · 서버 {server} ({team})"
        if s.get("team_a_end") in ("near", "far"):
            other = "far" if s["team_a_end"] == "near" else "near"
            desc += f" · Team A={s['team_a_end']}, Team B={other}"
        lines.append(desc)

    lines += [
        "",
        "## 각 event 필드",
        "- t: 라켓-공 임팩트 순간의 클립 시간(초, 소수점 1자리 이상). 타구음(오디오)을 함께 활용해 정확히.",
        "- side: 친 선수가 카메라 기준 가까운 쪽 코트(near)인지 먼 쪽(far)인지",
        "- lane: 친 선수의 화면상 좌우 위치 left / center / right",
        "- shot: serve / forehand / backhand / volley / smash / lob / dropshot / other",
        "  · serve: 토스 후 오버헤드 서브 동작 (언더핸드 서브 포함)",
        "  · forehand / backhand: 바운스 후 타구. 라켓을 쥔 손 쪽이면 포핸드, 반대쪽이면 백핸드(양손 백핸드 주의).",
        "    서브 리턴도 포핸드/백핸드로 기록.",
        "  · volley: 바운스 전에 친 공 (주로 네트 근처). smash: 머리 위 강타. lob: 높게 띄운 공. dropshot: 네트 바로 뒤에 짧게 떨어뜨린 공",
        "- spin: topspin / slice / flat / kick / unknown (구질)",
        "- serve_number: 서브일 때 1(첫 서브) 또는 2(폴트 후 두 번째 서브), 서브가 아니면 0",
        "- serve_type: 서브 구질 flat(빠르고 낮은 직선) / slice(옆으로 휘어짐) / kick(바운스 후 높게 튐) / unknown. 서브가 아니면 unknown",
        "- serve_direction: 서브 방향 T(센터) / body(몸쪽) / wide(사이드라인쪽) / unknown",
        "- result: in / fault(서브 폴트·네트) / ace(리시버가 못 건드린 서브) / winner(상대가 못 받은 샷) / error(아웃·네트로 실점) / unknown",
        "- speed_kmh: 임팩트 직후 공 속도 추정치(km/h, 레이더건 초속 기준). 근거가 부족하면 0",
        "- confidence: 이 event 판정 신뢰도 0~100",
        "",
        "## 속도 추정 방법",
        "- 공이 이동한 코트상 거리 ÷ 비행 시간으로 계산하고, 평균속도의 약 1.2배를 초속으로 기록.",
        "  예) 서브: 임팩트(베이스라인 위 약 2.5m 높이) → 상대 서비스박스 바운스 ≈ 18~20m. 0.55초 비행 → 평균 약 125km/h → 초속 약 150km/h",
        "- 동호인 일반 범위(참고용, 실제 관찰을 우선): 1st 플랫 서브 110~170, 슬라이스 서브 100~150, 킥 서브 90~130,",
        "  2nd 서브 80~120, 그라운드스트로크 60~120, 발리 40~90",
        "",
    ]
    if variant in ("v2", "v3"):
        lines += V2_RULES
        anchors = []
        for a in serve_anchors or []:
            tc = _clip_time(chunk, a["t"])
            if tc is not None:
                anchors.append(f"- {tc:.2f}초: {'먼 쪽' if a.get('end') == 'far' else '가까운 쪽'} 서버의 서브 임팩트"
                               + (f" → {'인' if a['call'] == 'in' else '폴트(' + a['call'] + ')'}" if a.get("call") in ("in", "out", "net") else ""))
        if anchors:
            lines += ["", "## 실측된 서브 임팩트 (공 추적·타구음 기준, 오차 약 0.05초)",
                      "아래 서브는 실제로 있었습니다. 서브 event의 t는 이 시각을 그대로 쓰고, 이 목록에 없는 서브는 신중히 판단하세요.",
                      *anchors]
        cands = []
        for c in (hit_candidates or []) if variant == "v3" else []:
            tc = _clip_time(chunk, c["t"])
            if tc is not None:
                cands.append(f"{tc:.2f}")
        if cands:
            lines += ["", "## 타구음 후보 시각 (오디오에서 자동 검출, 클립 시간 초)",
                      "라켓에 공이 맞는 소리처럼 들린 순간들입니다. 옆 코트 소리·발소리·바운스 소리도 섞여 있어 대부분은 타구가 아닙니다.",
                      "각 후보 시각의 영상을 확인해 이 코트 선수의 실제 타구인 것만 event로 기록하고, t는 해당 후보 시각을 쓰세요.",
                      "특히 네트 앞 발리는 이 목록으로 찾으세요. 후보에 없어도 영상에서 확실한 타구는 기록하세요.",
                      ", ".join(cands)]
    else:
        lines += [
            "",
            "## 보수적 원칙",
            "- 불명확한 타구는 기록하지 않음. 0개가 정답일 수 있음.",
            "- 포인트 사이에 공을 넘겨주는 타구, 연습 스윙, 공 튀기기는 제외.",
            "- 같은 타구를 두 번 기록하지 말 것.",
        ]
    lines += [
        "",
        "## observations",
        "- 코치 관점의 구체적 관찰 1~3개 (냉정·직접적, 칭찬으로 시작 금지). 어느 선수(side·lane 또는 팀)인지 명시.",
        "",
        "순수 JSON만 출력하세요.",
    ]
    return "\n".join(lines)


EVENT_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "t": {"type": "NUMBER"},
        "side": {"type": "STRING", "enum": ["near", "far", "unknown"]},
        "lane": {"type": "STRING", "enum": ["left", "center", "right", "unknown"]},
        "shot": {"type": "STRING", "enum": SHOTS},
        "spin": {"type": "STRING", "enum": SPINS},
        "serve_number": {"type": "INTEGER"},
        "serve_type": {"type": "STRING", "enum": SERVE_TYPES},
        "serve_direction": {"type": "STRING", "enum": SERVE_DIRS},
        "result": {"type": "STRING", "enum": RESULTS},
        "speed_kmh": {"type": "NUMBER"},
        "confidence": {"type": "INTEGER"},
    },
    "required": ["t", "side", "shot", "speed_kmh", "confidence"],
}

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "videoContext": {
            "type": "OBJECT",
            "properties": {
                "cameraAngle": {"type": "STRING"},
                "courtSurface": {"type": "STRING"},
                "visibility": {"type": "STRING"},
            },
        },
        "events": {"type": "ARRAY", "items": EVENT_SCHEMA},
        "observations": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["events"],
}


# ── Normalize / attribute / aggregate ─────────────────────────────

def _enum(v, allowed, default="unknown"):
    return v if v in allowed else default


def _clamp(v, lo=0, hi=100):
    try:
        return max(lo, min(hi, round(float(v))))
    except (TypeError, ValueError):
        return lo


def normalize_events(raw_events, chunk: dict, file_id: str, settings: dict, warnings: list) -> list[dict]:
    out = []
    for e in raw_events or []:
        if not isinstance(e, dict):
            continue
        try:
            t_clip = float(e.get("t"))
        except (TypeError, ValueError):
            continue
        if t_clip < 0 or t_clip > chunk["clip_duration"] + 1.0:
            continue
        t_src, seg = clip_to_source(chunk, t_clip)
        if seg is None or not (seg["own_start"] - 0.05 <= t_src < seg["own_end"] + 0.05):
            continue  # overlap region owned by the neighbouring window
        shot = _enum(e.get("shot"), SHOTS, "other")
        is_serve = shot == "serve"
        ev = {
            "t": round(t_src, 2),
            "file_id": file_id,
            "point_id": seg.get("point_id"),
            "point_no": seg.get("point_no"),
            "shot": shot,
            "side": _enum(e.get("side"), ["near", "far"]),
            "lane": _enum(e.get("lane"), ["left", "center", "right"]),
            "spin": _enum(e.get("spin"), SPINS),
            "serve_number": int(e.get("serve_number") or 0) if is_serve else 0,
            "serve_type": _enum(e.get("serve_type"), SERVE_TYPES) if is_serve else None,
            "serve_direction": _enum(e.get("serve_direction"), SERVE_DIRS) if is_serve else None,
            "result": _enum(e.get("result"), RESULTS),
            "speed_kmh": None,
            "confidence": _clamp(e.get("confidence")),
            "team": None,
            "player": None,
            "fixes": [],
            "_side_raw": _enum(e.get("side"), ["near", "far"]),
            "_ctx": {"a_end": seg.get("team_a_end"), "server": seg.get("server_id"),
                     "winner": seg.get("winner")},
        }
        if ev["serve_number"] not in (0, 1, 2):
            ev["serve_number"] = 0
        try:
            spd = float(e.get("speed_kmh") or 0)
        except (TypeError, ValueError):
            spd = 0.0
        if spd > 0:
            lo, hi = SPEED_RANGE.get(shot, SPEED_RANGE["default"])
            if lo <= spd <= hi:
                ev["speed_kmh"] = round(spd)
            else:
                warnings.append(f"{_mmss(t_src)} {SHOT_LABELS[shot]} 속도 {round(spd)}km/h 비현실적 → 제외")
        _attribute(ev, seg, settings)
        out.append(ev)
    return out


def _attribute(ev: dict, seg: dict, settings: dict):
    a_end = seg.get("team_a_end")
    server = seg.get("server_id")
    if ev["shot"] == "serve" and server:
        ev["team"] = server[0].upper()
        ev["player"] = server.upper()
        if a_end in ("near", "far"):
            # Serve side is known from the score state; trust it over the model.
            ev["side"] = a_end if ev["team"] == "A" else ("far" if a_end == "near" else "near")
        return
    if a_end in ("near", "far") and ev["side"] in ("near", "far"):
        ev["team"] = "A" if ev["side"] == a_end else "B"
        if (settings.get("mode") or "doubles") == "singles":
            ev["player"] = f"{ev['team']}1"


def dedupe_events(events: list[dict]) -> list[dict]:
    events = sorted(events, key=lambda e: e["t"])
    kept: list[dict] = []
    for e in events:
        dup = next((k for k in reversed(kept[-4:])
                    if abs(k["t"] - e["t"]) < 0.4 and k["shot"] == e["shot"] and k["side"] == e["side"]), None)
        if dup is None:
            kept.append(e)
        elif e["confidence"] > dup["confidence"]:
            kept[kept.index(dup)] = e
    return kept


def fix_serve_numbers(events: list[dict], warnings: list):
    """Within one labeled point, serves are 1st then 2nd regardless of what the model said."""
    by_point: dict[str, list[dict]] = {}
    for e in events:
        if e["shot"] == "serve" and e.get("point_id"):
            by_point.setdefault(e["point_id"], []).append(e)
    for serves in by_point.values():
        serves.sort(key=lambda e: e["t"])
        if len(serves) > 2:
            warnings.append(f"포인트 #{serves[0]['point_no']}에 서브 {len(serves)}개 감지 (렛 또는 오탐 가능)")
        for i, s in enumerate(serves[:2]):
            s["serve_number"] = i + 1
            if i == 0 and len(serves) > 1 and s["result"] in ("in", "unknown"):
                s["result"] = "fault"


def _speed_stats(values):
    vals = [v for v in values if v]
    if not vals:
        return {"avg": None, "max": None, "n": 0}
    return {"avg": round(sum(vals) / len(vals)), "max": max(vals), "n": len(vals)}


def _rally_block(events: list[dict]) -> dict:
    """Line calls on rally shots (point_winner.call_shot): how many landed in, out, in the net,
    too close to call, and the in-rate of those called."""
    calls = {c: 0 for c in ("in", "out", "net", "close")}
    shots = [e for e in events if e["shot"] != "serve"]
    for e in shots:
        c = (e.get("landing") or {}).get("call")
        if c in calls:
            calls[c] += 1
    judged = calls["in"] + calls["out"] + calls["net"]
    return {"shots": len(shots), "calls": calls, "unjudged": len(shots) - sum(calls.values()),
            "in_rate": round(calls["in"] / judged * 100) if judged else None,
            "winners": sum(1 for e in shots if e["result"] == "winner"),
            "errors": sum(1 for e in shots if e["result"] == "error")}


def aggregate(events: list[dict], settings: dict) -> dict:
    counts = {s: 0 for s in SHOTS}
    for e in events:
        counts[e["shot"]] += 1

    def stroke_block(shot):
        evs = [e for e in events if e["shot"] == shot]
        spins = {s: 0 for s in SPINS}
        for e in evs:
            spins[e["spin"]] += 1
        return {"count": len(evs), "spin": spins, "speed": _speed_stats(e["speed_kmh"] for e in evs)}

    serves = [e for e in events if e["shot"] == "serve"]
    by_type = {}
    for st in SERVE_TYPES:
        evs = [e for e in serves if e["serve_type"] == st]
        by_type[st] = {"count": len(evs), "speed": _speed_stats(e["speed_kmh"] for e in evs)}
    by_number = {}
    for n in (1, 2):
        evs = [e for e in serves if e["serve_number"] == n]
        judged = [e for e in evs if e["result"] in ("in", "ace", "fault")]
        good = [e for e in judged if e["result"] in ("in", "ace")]
        by_number[str(n)] = {
            "count": len(evs),
            "in_rate": round(len(good) / len(judged) * 100) if judged else None,
            "speed": _speed_stats(e["speed_kmh"] for e in evs),
        }
    dirs = {d: sum(1 for e in serves if e["serve_direction"] == d) for d in SERVE_DIRS}

    # Who hit it: Team A/B when the labeled score says which end each team is on,
    # otherwise only the near/far half of the court is known.
    team_names = {t: " / ".join(x for x in settings.get(f"team_{t.lower()}") or [] if x) for t in "AB"}
    groups: dict[str, dict] = {}
    for e in events:
        if e["team"]:
            key = f"team{e['team']}"
            label = f"Team {e['team']}" + (f" · {team_names[e['team']]}" if team_names[e["team"]] else "")
        else:
            key = f"side_{e['side']}"
            label = {"near": "가까운 쪽", "far": "먼 쪽"}.get(e["side"], "미확인")
        g = groups.setdefault(key, {"label": label, "counts": {s: 0 for s in SHOTS}, "rally": _rally_block([])})
        g["counts"][e["shot"]] += 1
        g.setdefault("_evs", []).append(e)
    for g in groups.values():
        g["rally"] = _rally_block(g.pop("_evs"))

    servers: dict[str, dict] = {}
    for e in serves:
        key = e["player"] or f"side_{e['side']}"
        label = player_name(settings, e["player"]) if e["player"] else \
            {"near": "가까운 쪽 서버", "far": "먼 쪽 서버"}.get(e["side"], "미확인 서버")
        sv = servers.setdefault(key, {"label": label, "count": 0, "first": 0, "second": 0,
                                      "types": {t: 0 for t in SERVE_TYPES}, "_speeds": []})
        sv["count"] += 1
        if e["serve_number"] == 1:
            sv["first"] += 1
        elif e["serve_number"] == 2:
            sv["second"] += 1
        sv["types"][e["serve_type"]] += 1
        if e["speed_kmh"]:
            sv["_speeds"].append(e["speed_kmh"])
    for sv in servers.values():
        sv["speed"] = _speed_stats(sv.pop("_speeds"))

    return {
        "counts": counts,
        "total_shots": len(events),
        "forehand": stroke_block("forehand"),
        "backhand": stroke_block("backhand"),
        "volley": stroke_block("volley"),
        "serve": {
            "count": len(serves),
            "by_type": by_type,
            "by_number": by_number,
            "directions": dirs,
            "speed": _speed_stats(e["speed_kmh"] for e in serves),
            "aces": sum(1 for e in serves if e["result"] == "ace"),
        },
        "rally": _rally_block(events),
        "winners": sum(1 for e in events if e["result"] in ("winner", "ace")),
        "errors": sum(1 for e in events if e["result"] == "error"),
        "groups": dict(sorted(groups.items())),
        "servers": dict(sorted(servers.items())),
    }


def sanity_warnings(summary: dict, events: list[dict], analyzed_seconds: float,
                    point_count: int = 0) -> tuple[list[str], int]:
    """Subset of TennisApp utils/videoSanityCheck.ts rules that apply to event data."""
    w, penalty = [], 0
    s1 = summary["serve"]["by_number"]["1"]["count"]
    s2 = summary["serve"]["by_number"]["2"]["count"]
    if s2 > s1:
        w.append(f"2nd 서브({s2})가 1st 서브({s1})보다 많음")
        penalty = max(penalty, 30)
    n_serves = summary["serve"]["count"]
    if point_count and n_serves > point_count * 2.5:
        w.append(f"서브 수({n_serves})가 포인트 수({point_count})×2.5 초과 — 중복/오탐 의심")
        penalty = max(penalty, 20)
    elif not point_count and analyzed_seconds > 0 and n_serves > analyzed_seconds / 8:
        w.append(f"서브 수({n_serves})가 분석 길이({round(analyzed_seconds)}s) 대비 과다")
        penalty = max(penalty, 20)
    ground = summary["counts"]["forehand"] + summary["counts"]["backhand"]
    if summary["counts"]["volley"] > ground > 0:
        w.append(f"발리({summary['counts']['volley']}) > 포핸드+백핸드({ground})")
        penalty = max(penalty, 25)
    if summary["serve"]["count"] > 5 and summary["counts"]["smash"] > summary["serve"]["count"] * 0.5:
        w.append(f"스매시({summary['counts']['smash']})가 서브 수의 50% 초과 — 오탐 의심")
        penalty = max(penalty, 15)
    sd = summary["serve"]["by_type"]
    if summary["serve"]["count"] >= 6 and sd["unknown"]["count"] > summary["serve"]["count"] * 0.6:
        w.append("서브 구질 판정 불가가 60% 이상 — 카메라 각도/화질 한계")
        penalty = max(penalty, 10)
    if analyzed_seconds > 60 and not events:
        w.append("타구가 하나도 감지되지 않음 — 영상 분석 실패 의심")
        penalty = max(penalty, 40)
    return w, penalty


# ── Runner ────────────────────────────────────────────────────────

class ShotAnalysisJob:
    """Runs the whole pipeline; `report(dict)` receives progress updates."""

    def __init__(self, *, project: dict, file_entry: dict, api_key: str, model: str,
                 scope: str, quality: str, high_res: bool, cache_dir: Path,
                 report=None, cancelled=None, prompt_variant: str = "v1", serve_anchors: list | None = None,
                 hit_candidates: list | None = None):
        self.project = project
        self.f = file_entry
        self.api_key = api_key
        self.model = model or gemini.DEFAULT_MODEL
        self.scope = scope
        self.quality = quality
        self.high_res = high_res
        self.prompt_variant = prompt_variant
        self.serve_anchors = serve_anchors or []
        self.hit_candidates = hit_candidates or []
        self.cache_dir = Path(cache_dir)
        self.report = report or (lambda d: None)
        self.cancelled = cancelled or (lambda: False)
        self.chunk_progress: list[float] = []
        self.chunk_stage: list[str] = []

    def _check(self):
        if self.cancelled():
            raise Cancelled()

    def _emit(self, message):
        n = max(1, len(self.chunk_progress))
        pct = sum(self.chunk_progress) / n
        done = sum(1 for p in self.chunk_progress if p >= 100)
        self.report({"percent": round(pct, 1), "message": message,
                     "chunks_done": done, "chunk_count": len(self.chunk_progress),
                     "stages": list(self.chunk_stage)})

    def _set(self, i, pct, stage, message=None):
        self.chunk_progress[i] = max(self.chunk_progress[i], min(100.0, pct))
        self.chunk_stage[i] = stage
        self._emit(message or f"구간 {i + 1}/{len(self.chunk_progress)} · {stage}")

    def run(self) -> dict:
        settings = self.project.get("settings") or {}
        segs = build_segments(self.f, self.project.get("points") or [], self.scope)
        if not segs:
            raise RuntimeError("분석할 구간이 없어.")
        chunks = group_chunks(segs)
        analyzed = sum(s["end"] - s["start"] for s in segs)
        src, src_kind = pick_source(self.f, self.quality, analyzed)
        has_audio = bool((self.f.get("metadata") or {}).get("audio_codec"))
        if src_kind == "proxy":
            has_audio = has_audio and _has_audio(src)
        self.chunk_progress = [0.0] * len(chunks)
        self.chunk_stage = ["대기"] * len(chunks)
        self._emit(f"{len(chunks)}개 구간 분석 시작 · 소스 {'원본' if src_kind == 'original' else '프록시'}")

        work = self.cache_dir / "shots_work" / self.f["id"]
        work.mkdir(parents=True, exist_ok=True)
        results: list[dict | None] = [None] * len(chunks)
        errors: list[str] = []

        def run_chunk(i):
            try:
                results[i] = self._run_chunk(i, chunks[i], src, has_audio, settings, work)
            except Cancelled:
                raise
            except Exception as exc:
                errors.append(f"구간 {i + 1}: {exc}")
                self._set(i, 100, "실패")

        try:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                futures = [pool.submit(run_chunk, i) for i in range(len(chunks))]
                for fu in futures:
                    fu.result()
        finally:
            shutil.rmtree(work, ignore_errors=True)
        self._check()

        ok = [r for r in results if r]
        if not ok:
            raise RuntimeError("모든 구간 분석이 실패했어.\n" + "\n".join(errors[:5]))

        warnings: list[str] = list(errors)
        events: list[dict] = []
        observations: list[str] = []
        usage = {"prompt": 0, "output": 0}
        context = None
        for chunk, r in zip(chunks, results):
            if not r:
                continue
            events += normalize_events(r["json"].get("events"), chunk, self.f["id"], settings, warnings)
            observations += [str(o) for o in (r["json"].get("observations") or []) if o][:3]
            context = context or r["json"].get("videoContext")
            usage["prompt"] += int(r.get("usage", {}).get("promptTokenCount") or 0)
            usage["output"] += int(r.get("usage", {}).get("candidatesTokenCount") or 0)
        fell_back = [i + 1 for i, r in enumerate(results) if r and r.get("model_used", self.model) != self.model]
        if fell_back:
            warnings.append(f"{self.model} 과부하·하루 한도로 구간 {', '.join(map(str, fell_back))}은 {FALLBACK_MODEL}로 분석했어.")
        events = dedupe_events(events)
        events, rule_fixes = apply_rally_rules(events, settings, fix_serve_numbers, warnings)
        summary = aggregate(events, settings)
        point_count = sum(1 for s in segs if s.get("point_id"))
        sanity, penalty = sanity_warnings(summary, events, analyzed, point_count)
        confs = [e["confidence"] for e in events]
        overall = round(sum(confs) / len(confs)) if confs else 0

        result = {
            "version": PROMPT_VERSION,
            "prompt_variant": self.prompt_variant,
            "serve_anchors": len(self.serve_anchors),
            "file_id": self.f["id"],
            "file_name": self.f.get("name"),
            "analyzed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": self.model,
            "scope": "points" if segs[0].get("point_id") else "full",
            "source": src_kind,
            "analyzed_seconds": round(analyzed, 1),
            "segment_count": len(segs),
            "chunk_count": len(chunks),
            "failed_chunks": len(errors),
            "video_context": context,
            "overall_confidence": max(0, overall - penalty),
            "summary": summary,
            "events": events,
            "rule_fixes": rule_fixes,
            "observations": observations[:8],
            "warnings": sanity + warnings[:30],
            "usage": usage,
        }
        measured = self.cache_dir / f"{self.f['id']}_serves.json"
        if measured.exists():
            from analysis.serve_measure_job import apply_measurements
            try:
                apply_measurements(result, json.loads(measured.read_text(encoding="utf-8")), settings)
            except Exception as exc:
                result["warnings"].append(f"서브 실측 결과 병합 실패: {exc}")
        return result

    def _run_chunk(self, i, chunk, src, has_audio, settings, work: Path) -> dict:
        self._check()
        variant = {} if self.prompt_variant == "v1" else {  # v1 keys stay identical to older caches
            "variant": self.prompt_variant,
            "anchors": [(round(a["t"], 2), a.get("end"), a.get("call")) for a in self.serve_anchors],
            **({"cands": [c["t"] for c in self.hit_candidates]} if self.prompt_variant == "v3" else {})}
        key = hashlib.sha1(json.dumps({
            "v": PROMPT_VERSION, **variant, "src": src, "model": self.model, "hr": self.high_res,
            "segs": [(s["start"], s["end"], s.get("server_id"), s.get("team_a_end")) for s in chunk["segments"]],
            "players": [settings.get("mode"), settings.get("team_a"), settings.get("team_b")],
            "cal": (self.f.get("calibration") or {}).get("points"),
        }, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
        cached = self.cache_dir / "shots" / f"{key}.json"
        if cached.exists():
            try:
                data = json.loads(cached.read_text(encoding="utf-8"))
                chunk["segments"] = data["segments"]
                chunk["clip_duration"] = data["clip_duration"]
                self._set(i, 100, "완료(캐시)")
                return data["response"]
            except Exception:
                pass

        clip = work / f"{i:03d}_{key}.mp4"
        self._set(i, 1, "클립 추출")
        extract_clip(src, chunk, clip, has_audio,
                     progress=lambda p: self._set(i, p * 0.35, "클립 추출"), cancelled=self.cancelled)
        # Built after extraction so the segment table uses the measured clip times.
        prompt = build_prompt(chunk, settings, self.f, self.prompt_variant, self.serve_anchors, self.hit_candidates)

        attempt = 0
        model = self.model
        while True:
            attempt += 1
            name = None
            try:
                self._check()
                self._set(i, 35, "업로드")
                up = gemini.upload_file(self.api_key, clip,
                                        progress=lambda s, t: self._set(i, 35 + 15 * s / max(1, t), "업로드"))
                name = up.get("name")
                self._set(i, 50, "Gemini 영상 처리")
                up = gemini.wait_active(self.api_key, up, cancelled=self.cancelled)
                self._set(i, 60, "AI 분석")
                parts = [
                    {"fileData": {"mimeType": "video/mp4", "fileUri": up["uri"]},
                     "videoMetadata": {"fps": VIDEO_FPS}},
                    {"text": prompt},
                ]
                res = gemini.generate_json(
                    self.api_key, model, parts, RESPONSE_SCHEMA,
                    media_resolution="MEDIA_RESOLUTION_HIGH" if self.high_res else None,
                )
                if not isinstance(res["json"], dict):
                    raise gemini.GeminiError("AI 응답 형식이 올바르지 않아.")
                break
            except gemini.GeminiError as exc:
                if exc.status == 499:
                    raise Cancelled()
                if exc.status in (400, 401, 403, 404):
                    raise
                if attempt >= 3 or getattr(exc, "daily_quota", False):
                    if exc.status in (429, 500, 503) and model != FALLBACK_MODEL:
                        model, attempt = FALLBACK_MODEL, 0
                        why = "하루 한도 소진" if getattr(exc, "daily_quota", False) else "과부하"
                        self._set(i, 35, f"{self.model} {why} → {FALLBACK_MODEL}로 분석")
                        continue
                    raise
                self._set(i, 35, f"재시도 대기 ({attempt}/3)")
                for _ in range(int(10 * attempt * 2)):
                    self._check()
                    time.sleep(0.5)
            finally:
                if name:
                    gemini.delete_file(self.api_key, name)

        res["model_used"] = model
        if model == self.model:  # a fallback answer must not be cached under the chosen model's key
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_text(json.dumps({"segments": chunk["segments"], "clip_duration": chunk["clip_duration"],
                                          "response": res}, ensure_ascii=False), encoding="utf-8")
        try:
            clip.unlink()
        except OSError:
            pass
        self._set(i, 100, "완료")
        return res


def _has_audio(path: str) -> bool:
    try:
        cp = subprocess.run([exe("ffprobe"), "-v", "error", "-select_streams", "a", "-show_entries",
                             "stream=index", "-of", "csv=p=0", path],
                            capture_output=True, text=True, timeout=20)
        return bool(cp.stdout.strip())
    except Exception:
        return False
