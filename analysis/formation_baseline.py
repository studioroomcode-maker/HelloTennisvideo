"""Optional Formation-first baseline for Hello Tennis Video.

Requires: numpy, opencv-python, ultralytics, ruptures
Runs on the browser-friendly proxy, not the original 4K HEVC file.
The output is intentionally simple and inspectable: a feature stream plus
unsupervised change-point proposals. It is a baseline/data tool, not the final
point detector.
"""
from __future__ import annotations

import json
from pathlib import Path


def analyze(file_entry: dict, output_path: str, sample_fps: float = 5.0, model_name: str = "yolo11n.pt") -> dict:
    try:
        import cv2
        import numpy as np
        import ruptures as rpt
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError(
            "Formation AI dependencies are not installed. Run: "
            "pip install -r requirements-ai.txt"
        ) from e

    proxy = file_entry.get("proxy")
    calibration = file_entry.get("calibration") or {}
    points = calibration.get("points") or []
    if not proxy or not Path(proxy).exists():
        raise RuntimeError("Generate the proxy first.")
    if len(points) < 4:
        raise RuntimeError("Save 4-point court calibration first.")

    cap = cv2.VideoCapture(proxy)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open proxy: {proxy}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, round(src_fps / sample_fps))

    # User order: near-left, near-right, far-right, far-left.
    src = np.float32([[p["x"], p["y"]] for p in points[:4]])
    dst = np.float32([[0, 1], [1, 1], [1, 0], [0, 0]])
    H = cv2.getPerspectiveTransform(src, dst)

    model = YOLO(model_name)
    features = []
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % step != 0:
            frame_index += 1
            continue
        h, w = frame.shape[:2]
        result = model.predict(frame, verbose=False, classes=[0], conf=0.2, imgsz=640)[0]
        court_people = []
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            x1,y1,x2,y2 = xyxy
            # Foot point is much more stable for court coordinates than box center.
            px, py = ((x1+x2)/2.0)/w, y2/h
            inp = np.array([[[px,py]]], dtype=np.float32)
            out = cv2.perspectiveTransform(inp, H)[0,0]
            cx, cy = float(out[0]), float(out[1])
            if -0.08 <= cx <= 1.08 and -0.10 <= cy <= 1.10:
                court_people.append((cx,cy,float(box.conf[0].cpu().item())))

        near = [(x,y,c) for x,y,c in court_people if y >= 0.5]
        far = [(x,y,c) for x,y,c in court_people if y < 0.5]
        def spread(group, axis):
            vals=[g[axis] for g in group]
            return float(max(vals)-min(vals)) if len(vals)>=2 else 0.0
        def mean_y(group):
            return float(sum(g[1] for g in group)/len(group)) if group else -1.0

        features.append({
            "time": frame_index/src_fps,
            "frame": frame_index,
            "people": len(court_people),
            "near_count": len(near),
            "far_count": len(far),
            "near_x_spread": spread(near,0),
            "far_x_spread": spread(far,0),
            "near_mean_y": mean_y(near),
            "far_mean_y": mean_y(far),
            "formation_2v2": int(len(near)>=2 and len(far)>=2),
        })
        frame_index += 1
    cap.release()

    change_times = []
    if len(features) >= 20:
        X = np.array([[f["people"],f["near_count"],f["far_count"],f["near_x_spread"],f["far_x_spread"],f["near_mean_y"],f["far_mean_y"],f["formation_2v2"]] for f in features], dtype=np.float64)
        # Robust enough for an unsupervised baseline: standardize dimensions.
        std = X.std(axis=0); std[std < 1e-6] = 1.0
        X = (X - X.mean(axis=0)) / std
        algo = rpt.Pelt(model="rbf", min_size=max(3, int(sample_fps)), jump=1).fit(X)
        # Conservative penalty; user labels will later determine optimal value.
        bkps = algo.predict(pen=8.0)
        for idx in bkps[:-1]:
            idx = min(max(idx,1),len(features)-1)
            change_times.append(features[idx]["time"])

    result = {
        "file_id": file_entry.get("id"),
        "proxy": proxy,
        "sample_fps": sample_fps,
        "model": model_name,
        "features": features,
        "change_points": change_times,
        "warning": "Unsupervised boundary proposals only; not POINT_START/END labels.",
    }
    Path(output_path).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("project_json")
    ap.add_argument("file_id")
    ap.add_argument("--output", default="formation_analysis.json")
    ap.add_argument("--fps", type=float, default=5.0)
    args=ap.parse_args()
    project=json.loads(Path(args.project_json).read_text(encoding="utf-8"))
    f=next(x for x in project["files"] if x["id"]==args.file_id)
    res=analyze(f,args.output,args.fps)
    print(f"features={len(res['features'])}, change_points={len(res['change_points'])}")
