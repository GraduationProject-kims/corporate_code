"""
live_record.py — 실시간 서버(server.py)의 시행 기록  [3-1조 추가]

서버 터미널에서:
  시행ID 입력 후 Enter → 기록 시작   예) 1019_front_01
  그냥 Enter          → 기록 끝
결과: runs/<RUNNER>/<시행ID>_<MMDD>_<HHMMSS>/
  run_meta.json, frames.csv, summary.json (녹화 영상 run_model.py와 같은 형식, judge.py 공통 함수),
  latency.csv(구간별 지연)
RUNNER(돌린 사람 이름)는 서버 실행할 때 지정: RUNNER=지은 uvicorn server:app --host 0.0.0.0 --port 8001

[10/26 통합]
  - 기록을 시작하면 판정 상태를 처음으로 되돌림 (server.py가 넘겨준 on_start → judge.FramePipeline.reset)
    녹화 영상처럼 frame_idx 0, 방향 기억 없음, depth 새로 계산에서 출발. 그 전 상태로 판정한 프레임은 기록 안 함
  - frame_idx는 판정에 쓴 프레임 번호 그대로 (녹화 영상과 같은 뜻)
  - run_meta.json, summary.json은 judge.run_meta(), judge.summary()로 만듦 (video_source: "live")
  - 판정은 하지 않음. 오차를 고칠 땐 이 파일이 아니라 judge.py
"""

import csv
import os
import threading
import time
from datetime import datetime

import judge

RUNNER = os.environ.get("RUNNER", "live")
LATENCY_COLUMNS = ["frame_idx", "t_ms", "recv_to_start_ms", "decode_ms", "yolo_ms", "depth_ms", "server_ms"]

_lock = threading.Lock()
_cur = {"rec": None}
_cfg = {"args": {}, "pipeline": {}, "on_start": None}   # start_console()에서 server.py가 넘겨줌


class _Recorder:
    def __init__(self, trial):
        self.trial = trial
        self.t0_ms = time.time() * 1000
        self.started = datetime.now()
        self.epoch = None    # on_start가 돌려준 판정 상태 번호. 이 번호로 처리한 프레임만 기록 (None이면 전부)
        self.dir = judge.make_run_dir(RUNNER, trial, self.started)
        self.rows, self.lat, self.frame_size = [], [], None
        self.f = open(self.dir / "frames.csv", "x", newline="", encoding="utf-8-sig")
        self.w = csv.DictWriter(self.f, fieldnames=judge.CSV_COLUMNS)
        self.w.writeheader()
        self.lf = open(self.dir / "latency.csv", "x", newline="", encoding="utf-8-sig")
        self.lw = csv.DictWriter(self.lf, fieldnames=LATENCY_COLUMNS)
        self.lw.writeheader()
        judge.write_json(self.dir / "run_meta.json", judge.run_meta(
            "live", {"trial": trial, **_cfg["args"]}, _cfg["pipeline"], RUNNER, None, run_at=self.started))

    def add(self, recv_ms, out, lat):
        """out: judge.FramePipeline.step() 결과, lat: 구간별 지연(ms)"""
        if self.epoch is not None and out["epoch"] != self.epoch:
            return   # 시행 시작 전 상태로 판정한 프레임
        if recv_ms < self.t0_ms:
            return   # 시행 시작(Enter) 전에 받은 프레임
        t_ms = round(recv_ms - self.t0_ms)   # 시행 시작부터 서버가 프레임을 받은 시각 (ms)
        self.frame_size = list(out["orig_wh"])
        j = out["j"]
        row = judge.frame_row(self.trial, out["idx"], t_ms, out["sx"], j["box_results"],
                              (j["obs_m"], j["obs_src"], j["obs_warn"], j["obs_target"]),
                              j["direction"], (j["motor"], j["vib_count"]), out["yolo_ms"], out["depth_ms"])
        self.rows.append(row)
        self.w.writerow(row)
        self.f.flush()
        lat_row = {"frame_idx": out["idx"], "t_ms": t_ms, **lat}
        self.lat.append(lat_row)
        self.lw.writerow(lat_row)
        self.lf.flush()

    def close(self):
        self.f.close()
        self.lf.close()
        rows = self.rows
        dur = rows[-1]["t_ms"] / 1000 if rows else 0
        judge.write_json(self.dir / "summary.json", judge.summary(
            rows, self.trial, "live", None, self.frame_size,
            round(len(rows) / dur, 2) if dur > 0 else None,   # 실시간: 서버가 처리한 프레임 기준
            round(dur, 3), time.time() - self.t0_ms / 1000,
            server_ms=[l["server_ms"] for l in self.lat]))


def start_trial(trial):
    with _lock:
        if _cur["rec"] is not None:
            return f"이미 기록 중: {_cur['rec'].trial} (먼저 그냥 Enter로 끝내세요)"
        if not judge.check_trial(trial):
            print(f"[주의] 시행ID '{trial}'가 <MMDD>_<front|left|right|none>_<2자리> 형식이 아니에요. 그대로 진행합니다.")
        rec = _Recorder(trial)
        if _cfg["on_start"] is not None:
            rec.epoch = _cfg["on_start"]()   # 판정 상태를 처음으로 (녹화 영상을 처음부터 트는 것과 같게)
        _cur["rec"] = rec
        return f"● 기록 시작: {trial} → {rec.dir}"


def stop_trial():
    with _lock:
        rec, _cur["rec"] = _cur["rec"], None
        if rec is None:
            return "기록 중인 시행이 없어요."
        rec.close()
    msg = f"■ 기록 끝: {rec.trial} ({len(rec.rows)}프레임) → {rec.dir}"
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


def start_console(args=None, pipeline=None, on_start=None):
    """args: run_meta.json args에 더할 값, pipeline: judge.pipeline_info() 결과,
    on_start: 기록 시작 때 부를 함수 (판정 상태를 처음으로 되돌리고 새 epoch 번호를 돌려줌)"""
    _cfg.update(args=dict(args or {}), pipeline=dict(pipeline or {}), on_start=on_start)
    threading.Thread(target=_console, daemon=True).start()
