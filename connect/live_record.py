"""
live_record.py — 실시간 서버(server.py)의 시행 기록  [3-1조 추가]

서버 터미널에서:
  시행ID 입력 후 Enter → 기록 시작   예) 1019_front_01
  그냥 Enter          → 기록 끝
결과: runs/<RUNNER>/<시행ID>_<MMDD>_<HHMMSS>/
  run_meta.json, frames.csv(녹화 영상 run_model.py와 같은 형식), summary.json, latency.csv(구간별 지연)
RUNNER(돌린 사람 이름)는 서버 실행할 때 지정: RUNNER=지은 uvicorn server:app --host 0.0.0.0 --port 8001
"""

import csv
import json
import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import judge

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.environ.get("RUNNER", "live")
LATENCY_COLUMNS = ["frame_idx", "t_ms", "recv_to_start_ms", "decode_ms", "yolo_ms", "depth_ms", "server_ms"]

_lock = threading.Lock()
_cur = {"rec": None}
_meta_extra = {}


def _write_json(path, data):
    with open(path, "x", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else "unknown"
    except Exception:
        return "unknown"


def _versions():
    v = {"opencv": cv2.__version__}
    for name in ("torch", "ultralytics", "transformers"):
        try:
            v[name] = __import__(name).__version__
        except Exception:
            pass
    return v


class _Recorder:
    def __init__(self, trial):
        self.trial = trial
        self.t0_ms = time.time() * 1000
        self.started = datetime.now()
        self.dir = Path(HERE) / judge.RUNS_DIR / RUNNER / f"{trial}_{self.started:%m%d_%H%M%S}"
        self.dir.mkdir(parents=True, exist_ok=False)
        self.rows, self.lat, self.idx, self.frame_size = [], [], 0, None
        self.f = open(self.dir / "frames.csv", "x", newline="", encoding="utf-8-sig")
        self.w = csv.DictWriter(self.f, fieldnames=judge.CSV_COLUMNS)
        self.w.writeheader()
        self.lf = open(self.dir / "latency.csv", "x", newline="", encoding="utf-8-sig")
        self.lw = csv.DictWriter(self.lf, fieldnames=LATENCY_COLUMNS)
        self.lw.writeheader()
        _write_json(self.dir / "run_meta.json", {
            "run_at": self.started.isoformat(timespec="seconds"),
            "source": "live",
            "args": {"trial": trial, **_meta_extra},
            "pipeline": {"yolo_weights": judge.MODEL_PATH, "depth_model": judge.DEPTH_MODEL,
                         "conf": judge.CONF_THRES, "imgsz": judge.IMG_SIZE, **_versions(), **_meta_extra},
            "alarm_constants": judge.alarm_constants(),
            "runner": RUNNER,
            "git_commit": _git_commit(),
            "video_file": None,
            "schema_version": judge.SCHEMA_VERSION,
        })

    def add(self, recv_ms, orig_wh, sx, j, yolo_ms, depth_ms, lat):
        t_ms = round(recv_ms - self.t0_ms)   # 시행 시작부터 서버가 프레임을 받은 시각 (ms)
        self.frame_size = list(orig_wh)
        row = judge.frame_row(self.trial, self.idx, t_ms, sx, j["box_results"],
                              (j["obs_m"], j["obs_src"], j["obs_warn"], j["obs_target"]),
                              j["direction"], (j["motor"], j["vib_count"]), yolo_ms, depth_ms)
        self.rows.append(row)
        self.w.writerow(row)
        self.f.flush()
        lat_row = {"frame_idx": self.idx, "t_ms": t_ms, **lat}
        self.lat.append(lat_row)
        self.lw.writerow(lat_row)
        self.lf.flush()
        self.idx += 1

    def close(self):
        self.f.close()
        self.lf.close()
        rows = self.rows
        label_counts = {}
        for r in rows:
            for label in filter(None, r["labels"].split(";")):
                label_counts[label] = label_counts.get(label, 0) + 1
        first2 = next((r for r in rows if r["level"] == 2), None)
        first_vib = next((r for r in rows if r["motor"] != ""), None)
        med = lambda v: round(float(np.median(v)), 1) if v else None
        dur = rows[-1]["t_ms"] / 1000 if rows else 0
        _write_json(self.dir / "summary.json", {
            "trial": self.trial,
            "source": "live",
            "video": None,
            "frames": len(rows),
            "processed": len(rows),
            "frame_size": self.frame_size,
            "video_fps": round(len(rows) / dur, 2) if dur > 0 else None,   # 실시간: 서버가 처리한 프레임 기준
            "duration_sec": round(dur, 3),
            "frames_with_det": sum(1 for r in rows if r["n_dets"] > 0),
            "frames_with_target": sum(1 for r in rows if r["level"] == 2),
            "label_counts": label_counts,
            "first_level1": None,
            "first_level2": None if first2 is None else {
                "t_sec": first2["t_ms"] / 1000, "dist": first2["dist"], "direction": first2["direction"] or None},
            "first_level3": None,
            "max_level": max((r["level"] for r in rows), default=0),
            "yolo_ms_median": med([r["yolo_ms"] for r in rows]),
            "depth_ms_median": med([r["depth_ms"] for r in rows if r["depth_ms"] != ""]),
            "server_ms_median": med([l["server_ms"] for l in self.lat]),
            "wall_sec": round(time.time() - self.t0_ms / 1000, 2),
            "expected_dir": None,
            "gt_2m_sec": None,
            "note": None,
            "first_vibration": None if first_vib is None else {
                "t_sec": first_vib["t_ms"] / 1000, "dist": first_vib["dist"],
                "direction": first_vib["direction"] or None, "target_label": first_vib["target_label"]},
        })


def start_trial(trial):
    with _lock:
        if _cur["rec"] is not None:
            return f"이미 기록 중: {_cur['rec'].trial} (먼저 그냥 Enter로 끝내세요)"
        if not judge.check_trial(trial):
            print(f"[주의] 시행ID '{trial}'가 <MMDD>_<front|left|right|none>_<2자리> 형식이 아니에요. 그대로 진행합니다.")
        _cur["rec"] = _Recorder(trial)
        return f"● 기록 시작: {trial} → {_cur['rec'].dir}"


def stop_trial():
    with _lock:
        rec, _cur["rec"] = _cur["rec"], None
        if rec is None:
            return "기록 중인 시행이 없어요."
        rec.close()
    msg = f"■ 기록 끝: {rec.trial} ({rec.idx}프레임) → {rec.dir}"
    print("\n" + msg)
    return msg


def add(*a, **k):
    """server.py가 매 프레임 부름. 기록 중일 때만 씀"""
    with _lock:
        if _cur["rec"] is not None:
            _cur["rec"].add(*a, **k)


def draw_rec(img, H):
    rec = _cur["rec"]
    if rec is not None:
        judge.put_text(img, f"REC {rec.trial}", (8, H - 12), (0, 0, 255), 0.6)


def _console():
    print(f"\n[시행 기록] 시행ID 입력 후 Enter = 시작 (예: 1019_front_01) / 그냥 Enter = 끝  (RUNNER={RUNNER})\n")
    while True:
        try:
            line = input().strip()
        except (EOFError, OSError):
            return
        if line:
            print("\n" + start_trial(line))
        else:
            stop_trial()


def start_console(meta_extra=None):
    _meta_extra.update(meta_extra or {})
    threading.Thread(target=_console, daemon=True).start()
