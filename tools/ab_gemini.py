"""A/B Gemini settings on the labelled points and score each run against the shot labels.

Each run gets its own empty cache directory, so nothing is reused between runs (Gemini
answers differ run to run; repeats show how much) and the app's own results are untouched.
Serve measurements are not merged in, so the score is Gemini + rally rules only.

    python tools/ab_gemini.py --configs gemini-3.5-flash gemini-3.8-flash gemini-3.8-flash:hr --repeats 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis import gemini_client as gemini  # noqa: E402
from analysis.label_eval import evaluate  # noqa: E402
from analysis.shot_analysis import ShotAnalysisJob  # noqa: E402

# Standard-tier USD per 1M tokens (input, output), ai.google.dev pricing page, 2026-10-07.
PRICE = {"gemini-3.5-flash": (1.50, 9.00), "gemini-3.6-flash": (0.75, 3.75), "gemini-3.7-flash": (0.75, 3.75),
         "gemini-3.8-flash": (0.75, 3.75), "gemini-3.5-flash-lite": (0.30, 2.50)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="2d9348b629")
    ap.add_argument("--configs", nargs="+", required=True, help="model[:hr][:v2]")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    project = json.loads((ROOT / "projects" / "current_project.json").read_text(encoding="utf-8"))
    cfg = json.loads((ROOT / "config" / "gemini.json").read_text(encoding="utf-8"))
    f = next(x for x in project["files"] if x["id"] == args.file)
    labels = [l for l in project.get("shot_labels", []) if l["file_id"] == args.file]
    all_points = [p for p in project["points"] if p["file_id"] == args.file]
    labelled = {p["id"] for p in all_points
                if any(p["start"] - 2.5 <= l["t"] <= p["end"] + 1.0 for l in labels)}
    project["points"] = [p for p in all_points if p["id"] in labelled]
    print(f"points {len(project['points'])}, labels {len(labels)}", flush=True)

    measured = ROOT / "cache" / f"{args.file}_serves.json"
    anchors = [{"t": s["contact_t"], "end": s.get("server_end"), "call": s.get("call")}
               for s in json.loads(measured.read_text(encoding="utf-8"))["serves"]] if measured.exists() else []

    from analysis.audio_onset import hit_candidates
    cands = hit_candidates(f["path"], [(p["start"] - 1.0, p["end"] + 0.5) for p in project["points"]])
    print(f"hit candidates {len(cands)}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in args.configs:
        model, *opts = spec.split(":")
        hr = "hr" in opts
        variant = next((o for o in opts if o in ("v2", "v3")), "v1")
        for rep in range(args.repeats):
            name = f"{model}{'_hr' if hr else ''}{'_' + variant if variant != 'v1' else ''}_r{rep + 1}"
            t0 = time.time()
            job = ShotAnalysisJob(project=project, file_entry=f, api_key=cfg["api_key"], model=model,
                                  scope="points", quality="auto", high_res=hr,
                                  cache_dir=out / name, prompt_variant=variant,
                                  serve_anchors=anchors if variant != "v1" else None,
                                  hit_candidates=cands if variant == "v3" else None)
            try:
                res = job.run()
            except Exception as exc:  # keep going with the other configs
                print(f"{name}: FAILED {exc}", flush=True)
                rows.append({"run": name, "error": str(exc)})
                continue
            ev = evaluate(labels, res["events"], project["points"])
            pin, pout = PRICE.get(model, (0, 0))
            cost = res["usage"]["prompt"] / 1e6 * pin + res["usage"]["output"] / 1e6 * pout
            row = {"run": name, "secs": round(time.time() - t0), "usage": res["usage"], "usd": round(cost, 4),
                   "eval": ev, "events": [{k: e.get(k) for k in ("t", "shot", "side", "serve_type", "fixes")}
                                          for e in res["events"]]}
            rows.append(row)
            (out / f"{name}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"{name}: found {ev['correct']}/{ev['labels']} · AI {ev['ai_events']} · confusions {ev['confusion']} "
                  f"· spin {ev['serve_spin']['correct']}/{ev['serve_spin']['checked']} "
                  f"· side {ev['side']['correct']}/{ev['side']['checked']} · dt {ev['timing']['mean_s']}s "
                  f"· tokens {res['usage']} · ${cost:.4f} · {row['secs']}s", flush=True)
    (out / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
