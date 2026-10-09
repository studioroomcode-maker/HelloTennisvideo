"""Score AI shot events against shots the user labelled by hand while watching.

Only points that contain at least one label are scored, so an unlabelled point never
counts as "the AI made things up". Within those points an AI event matches a label of
the same shot type within the label's tolerance (MATCH_S by default; labels typed from
memory can carry a wider "tol"); same time but a different type is a confusion.

A "stroke" label is a groundstroke whose forehand/backhand side the labeller didn't
note: it matches any groundstroke the AI reports, so it scores detection but not type.
A volley and a smash are the same shot for scoring: both are taken out of the air, and a
high volley vs a smash is a judgement call the user chose not to make.
A serve label may carry "spin" (flat/slice/kick); matched serves then also score the
AI's serve type.
"""
from __future__ import annotations

MATCH_S = 0.7
TYPES = ["serve", "forehand", "backhand", "volley", "smash", "lob", "dropshot", "other", "stroke"]
STROKE_OK = {"forehand", "backhand", "lob", "dropshot", "other"}
OUT_OF_THE_AIR = {"volley", "smash"}


def _point_of(t, points):
    for p in points:
        if p["start"] - 2.5 <= t <= p["end"] + 1.0:
            return p["id"]
    return None


def _same(label_shot, ai_shot):
    return (ai_shot == label_shot or (label_shot == "stroke" and ai_shot in STROKE_OK)
            or (label_shot in OUT_OF_THE_AIR and ai_shot in OUT_OF_THE_AIR))


def evaluate(labels: list[dict], events: list[dict], points: list[dict]) -> dict:
    labelled_points = {_point_of(l["t"], points) for l in labels} - {None}
    ai = [e for e in events if _point_of(e["t"], points) in labelled_points]
    lab = [l for l in labels if _point_of(l["t"], points) in labelled_points]
    tol = [float(l.get("tol") or MATCH_S) for l in lab]
    used_ai, used_lab = set(), set()
    pairs = []
    # exact type matches first, nearest in time
    cand = sorted(((abs(e["t"] - l["t"]), i, j) for i, l in enumerate(lab) for j, e in enumerate(ai)
                   if _same(l["shot"], e["shot"]) and abs(e["t"] - l["t"]) <= tol[i]))
    for _, i, j in cand:
        if i in used_lab or j in used_ai:
            continue
        used_lab.add(i)
        used_ai.add(j)
        pairs.append((i, j, True))
    # then same moment, wrong type
    cand = sorted(((abs(e["t"] - l["t"]), i, j) for i, l in enumerate(lab) for j, e in enumerate(ai)
                   if abs(e["t"] - l["t"]) <= tol[i]))
    confusion = {}
    for _, i, j in cand:
        if i in used_lab or j in used_ai:
            continue
        used_lab.add(i)
        used_ai.add(j)
        pairs.append((i, j, False))
        key = f"{lab[i]['shot']}->{ai[j]['shot']}"
        confusion[key] = confusion.get(key, 0) + 1
    per = {}
    for ty in TYPES:
        n_lab = sum(1 for l in lab if l["shot"] == ty)
        n_ai = sum(1 for e in ai if e["shot"] == ty)
        if not n_lab and not n_ai:
            continue
        hit = sum(1 for i, j, ok in pairs if ok and lab[i]["shot"] == ty)
        per[ty] = {"labelled": n_lab, "ai": n_ai, "correct": hit,
                   "recall": round(hit / n_lab * 100) if n_lab else None,
                   "precision": round(hit / n_ai * 100) if n_ai else None}
    spin = [(lab[i]["spin"], ai[j].get("serve_type") or "unknown") for i, j, ok in pairs
            if ok and lab[i]["shot"] == "serve" and lab[i].get("spin")]
    sides = [lab[i]["side"] == ai[j].get("side") for i, j, ok in pairs if ok and lab[i].get("side")]
    dts = [ai[j]["t"] - lab[i]["t"] for i, j, _ in pairs]
    missed = [l for i, l in enumerate(lab) if i not in used_lab]
    extra = [e for j, e in enumerate(ai) if j not in used_ai]
    total_ok = sum(1 for *_, ok in pairs if ok)
    return {
        "points_scored": len(labelled_points),
        "labels": len(lab), "ai_events": len(ai), "correct": total_ok,
        "recall": round(total_ok / len(lab) * 100) if lab else None,
        "precision": round(total_ok / len(ai) * 100) if ai else None,
        "per_type": per, "confusion": confusion,
        "serve_spin": {"checked": len(spin), "correct": sum(1 for a, b in spin if a == b),
                       "pairs": [f"{a}->{b}" for a, b in spin]},
        "side": {"checked": len(sides), "correct": sum(sides)},
        "timing": {"n": len(dts), "mean_s": round(sum(dts) / len(dts), 2) if dts else None,
                   "mean_abs_s": round(sum(abs(d) for d in dts) / len(dts), 2) if dts else None},
        "missed": [{"t": l["t"], "shot": l["shot"]} for l in missed],
        "extra": [{"t": round(e["t"], 2), "shot": e["shot"]} for e in extra],
    }
