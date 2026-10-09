"""Experiment: if the hit moment is known, how well does Gemini name the stroke?

For every hand-labelled non-serve hit, a 1.3 s clip around the label time is cut from the
original video, cropped to the hitter's half of the court at native resolution. All clips go
inline in ONE request (the free tier allows ~20 requests per model per day), each with a narrow
question about its single hit. Scored against the labels.

    python tools/hit_type_probe.py --model gemini-3.8-flash --out <dir> [--fps 10] [--hits detected.json]

With --hits the clips are cut at automatically detected hit moments ([{"t", "side", "hole"}],
analysis.hit_detect) instead of the label times, and each label is scored through the
detected hit matched to it -- the whole pipeline, not just the classifier.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis import gemini_client as gemini  # noqa: E402
from analysis.hit_shots import FAR_SOUND_S, SCHEMA, clip_line, crop_box, cut, hand_note, prompt  # noqa: E402
from analysis.shot_analysis import build_segments  # noqa: E402
from analysis.serve_speed import Court  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="2d9348b629")
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--fps", type=float, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hits", help="detected hits json; default: cut at the label times")
    ap.add_argument("--hands", action="store_true", help="tell which hand each side's players hold the racket with")
    args = ap.parse_args()
    project = json.loads((ROOT / "projects" / "current_project.json").read_text(encoding="utf-8"))
    cfg = json.loads((ROOT / "config" / "gemini.json").read_text(encoding="utf-8"))
    f = next(x for x in project["files"] if x["id"] == args.file)
    W, H = int(f["metadata"]["width"]), int(f["metadata"]["height"])
    court = Court(f["calibration"]["points"])
    doubles = (project["settings"].get("mode") or "doubles") == "doubles"
    labels = [l for l in project["shot_labels"] if l["file_id"] == args.file and l["shot"] != "serve"]
    if args.hits:
        hits = json.loads(Path(args.hits).read_text(encoding="utf-8"))
        moments = [{"t": h["t"] - (FAR_SOUND_S if h["side"] == "far" and h.get("audio") else 0.0),
                    "side": h["side"], "detected": h} for h in hits]
    else:
        moments = [{"t": l["t"], "side": l["side"], "label": l} for l in labels]
    if args.hands:
        segs = build_segments(f, project.get("points") or [], "points")
        for m in moments:
            seg = next((s for s in segs if s["start"] - 0.5 <= m["t"] <= s["end"] + 0.5), None) or {}
            m["note"] = hand_note(project["settings"], seg, m["side"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def make(m):
        clip = out / f"hit_{m['t']:.2f}.mp4"
        cut(f["path"], m["t"], crop_box(court, m["side"], W, H), clip)
        return clip

    with ThreadPoolExecutor(4) as pool:
        clips = list(pool.map(make, moments))
    parts = [{"text": prompt(doubles)}]
    for k, (m, clip) in enumerate(zip(moments, clips), 1):
        parts.append({"text": clip_line(k, m["side"], m.get("note") or "")})
        parts.append({"inlineData": {"mimeType": "video/mp4", "data": base64.b64encode(clip.read_bytes()).decode()},
                      "videoMetadata": {"fps": args.fps}})
    print(f"{len(clips)} clips, {sum(c.stat().st_size for c in clips) / 1e6:.1f} MB inline", flush=True)
    r = gemini.generate_json(cfg["api_key"], args.model, parts, SCHEMA, media_resolution="MEDIA_RESOLUTION_HIGH")
    by_clip = {h.get("clip"): h for h in r["json"].get("hits", [])}
    for k, m in enumerate(moments, 1):
        m["answer"] = by_clip.get(k)
    if args.hits:  # pair each label with the closest unused detected hit on the same side
        used = set()
        for lab in labels:
            best = None
            for k, m in enumerate(moments):
                dt = abs(m["detected"]["t"] - lab["t"])
                if k in used or m["side"] != lab["side"] or dt > (lab.get("tol") or 1.2):
                    continue
                if best is None or dt < best[0]:
                    best = (dt, k)
            if best:
                used.add(best[1])
                moments[best[1]]["label"] = lab
    ok, n, rows = 0, 0, []
    for m in moments:
        j, lab = m.get("answer"), m.get("label")
        if lab and not j:
            print(f"{lab['t']:7.2f} {lab['shot']:8s} 응답 없음")
            continue
        if not lab:
            print(f"{m['t']:7.2f} {m['side']:4s} (정답 없음) AI {(j or {}).get('shot')} 타구 {(j or {}).get('found_hit')} "
                  f"· {((j or {}).get('reason') or '')[:60]}")
            continue
        n += 1
        hit = j.get("shot") == lab["shot"]
        ok += hit
        rows.append({"label": lab, "answer": j, "t": m["t"]})
        print(f"{lab['t']:7.2f}→{m['t']:7.2f} {lab['side']:4s} 정답 {lab['shot']:8s} AI {j.get('shot'):8s} {'O' if hit else 'X'} "
              f"바운스 {j.get('bounced_before_hit'):7s} 라켓 {j.get('racket_side'):8s} 위치 {j.get('hitter_position')} "
              f"신뢰 {j.get('confidence')} · {(j.get('reason') or '')[:60]}")
    missed = [l for l in labels if not any(m.get("label") is l for m in moments)]
    for lab in missed:
        print(f"{lab['t']:7.2f} {lab['shot']:8s} 타구 검출 못 함")
    u = r["usage"]
    print(f"맞힘 {ok}/{n} (정답 {len(labels)}개 중) · tokens in {u.get('promptTokenCount')} out "
          f"{int(u.get('candidatesTokenCount') or 0) + int(u.get('thoughtsTokenCount') or 0)}")
    (out / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
