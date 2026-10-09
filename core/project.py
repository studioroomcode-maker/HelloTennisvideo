from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from .scoring import build_score_rows


def new_project() -> dict:
    return {
        "version": "0.1.0-mvp",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "settings": {
            "mode": "doubles",
            "team_a": ["선수A1", "선수A2"],
            "team_b": ["선수B1", "선수B2"],
            "server_rotation": ["A1", "B1", "A2", "B2"],
            "no_ad": False,
            "tiebreak_at_6_all": True,
            "scoreboard_position_mode": "court",
        },
        "files": [],
        "points": [],
        "shot_labels": [],
        "notes": [],
    }


def file_entry(path: str, metadata: dict) -> dict:
    return {
        "id": uuid.uuid4().hex[:10],
        "path": str(Path(path).resolve()),
        "name": Path(path).name,
        "metadata": metadata,
        "proxy": None,
        "calibration": None,
    }


def point_entry(file_id: str, start: float, end: float) -> dict:
    return {
        "id": uuid.uuid4().hex[:10],
        "file_id": file_id,
        "start": round(float(start), 3),
        "end": round(float(end), 3),
        "winner": None,
        "status": "manual",
        "review_reason": None,
    }


def recompute(project: dict) -> dict:
    rows = build_score_rows(project.get("points", []), project.get("settings", {}))
    by_id = {r["point_id"]: r for r in rows}
    for p in project.get("points", []):
        row = by_id.get(p.get("id"))
        if row:
            p["score_before"] = row["before"]
            p["score_after"] = row["after"]
    return project


def save_project(project: dict, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    recompute(project)
    path.write_text(json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8")


def load_project(path: str | Path) -> dict:
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    if raw is None or not str(raw).strip():
        raise RuntimeError(f"프로젝트 파일이 비어 있어: {path}")
    try:
        data = json.loads(str(raw))
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"프로젝트 JSON을 읽지 못했어: {path.name} · {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"프로젝트 JSON의 최상위 값은 객체여야 해: {path.name}")
    defaults = new_project()
    data.setdefault("settings", defaults["settings"])
    data.setdefault("files", [])
    data.setdefault("points", [])
    data.setdefault("notes", [])
    data.setdefault("shot_labels", [])
    return recompute(data)
