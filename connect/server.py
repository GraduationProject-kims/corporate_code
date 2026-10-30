"""
Blind Helper - 경로 B 서버 (YOLO + Metric Depth, 박스별 절대거리 + 폰 회신)

YOLO로 객체 탐지 → 각 박스 중심의 metric depth 값(미터)을 읽어
"라벨 + 거리(m)"를 폰으로 회신. 폰이 프리뷰 위에 오버레이.

Metric Depth 모델: 절대 거리(미터) 추정.
  실내: depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf
  실외: depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf

종료: Ctrl+C

[3-1조 수정 10/16] 판정을 judge.py로 (녹화 영상 run_model.py와 같은 코드)
  - 박스 거리·경로 장애물·경고·방향·모터를 judge.py로 계산 (경로 depth < 3 m면 경고)
  - 폰 회신 형식은 그대로 + "judge" 항목 추가 (dets는 맨 끝 유지) → 앱 수정 불필요
  - 시연 사이트 전송(live_mirror)은 원래 맥북 버전 그대로 유지
  - 시행 기록: 서버 터미널에 시행ID 입력 + Enter → 시작, 그냥 Enter → 끝 (live_record.py)
    실행할 때 이름 지정: RUNNER=내이름 uvicorn server:app --host 0.0.0.0 --port 8001

[3-1조 수정 10/26] 처리 순서도 judge.py로 (녹화 영상 run_model.py와 완전히 같은 코드)
  - 축소 → YOLO → depth(N프레임마다) → 판정 → 방향 유지는 judge.FramePipeline.step() 한 줄
  - 장치 자동 선택(DEVICE), 모델 불러오기, depth 몇 프레임마다(DEPTH_EVERY_N) 기본값도 judge.py
  - 시행 기록을 시작하면 판정 상태를 처음으로 되돌림 (녹화 영상을 처음부터 트는 것과 같게)
  - 이 파일에는 폰 사진 받기·풀기, 폰 회신, 사이트 전송, 화면 창만 남음
  - 오차를 고칠 땐 이 파일이 아니라 judge.py를 고치고 서버를 다시 켤 것
"""

import asyncio
import json
import os
import threading
import time
from collections import deque

# 맥(MPS)에서 지원 안 되는 연산이 나오면 오류 대신 CPU로 대신 계산 (torch import 전에 설정해야 함)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
import torch
from live_mirror import LiveMirror
live_mirror = LiveMirror(wait_for_phone=False)
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

import judge          # [3-1조] 처리·판정 (녹화 영상 평가와 같은 코드). 모델·판정 설정값도 여기
import live_record    # [3-1조] 시행 기록

# 설정 — 서버에서만 쓰는 값 (판정에는 영향 없음)
SHOW_WINDOW = True
SITE_JUDGE = os.environ.get("SITE_JUDGE") == "1"   # [3-1조] 1이면 사이트에도 판정(경고·방향·모터·거리)을 record 칸으로 보냄 (기본: 원래 형식 그대로)
_site_state = {"sent": False, "traced": False}
TORCH_THREADS = 4
USE_OPENVINO = False          # Intel CPU 전용 가속. 장치가 cpu일 때만 쓰임 (맥 M1에서는 안 씀). 켜면 결과가 조금 다를 수 있음
# [3-1조] N프레임마다 depth 1번. 기본값은 judge.py의 DEPTH_EVERY_N
#   한 번만 바꿔 볼 때: DEPTH_EVERY_N=3 uvicorn server:app ... (녹화 평가 run_model.py --depth_every_n과 같은 값이어야 결과가 같음)
DEPTH_EVERY_N = max(1, int(os.environ.get("DEPTH_EVERY_N", judge.DEPTH_EVERY_N)))

# 연산 장치: judge.py의 DEVICE ("auto" = NVIDIA GPU(cuda) → 맥 Apple Silicon(mps) → cpu)
DEVICE = judge.resolve_device(judge.DEVICE)

torch.set_num_threads(TORCH_THREADS)
cv2.setNumThreads(TORCH_THREADS)

app = FastAPI()

print(f"연산 장치: {DEVICE}")
model, depth_pipe, using_openvino = judge.load_models(DEVICE, use_openvino=USE_OPENVINO, depth_optional=True)
pipe = judge.FramePipeline(model, depth_pipe, DEVICE, DEPTH_EVERY_N, using_openvino=using_openvino,
                           keep_going_on_depth_error=True)

print(f"모델 로드 완료 (device={DEVICE}, openvino={using_openvino}, depth={depth_pipe is not None}). "
      f"폰 연결 대기 중...")
print(f"[3-1조] DEPTH_EVERY_N={DEPTH_EVERY_N} (depth {DEPTH_EVERY_N}프레임마다 1번), "
      f"사이트 판정 전송 {'켜짐' if SITE_JUDGE else '꺼짐 (켜려면 SITE_JUDGE=1)'}")

_lock = threading.Lock()
_latest = {"data": None, "recv_ms": 0.0}
_stop = threading.Event()
_result_lock = threading.Lock()
_pending_result = {"json": None}
latest_view = {"annotated": None}

proc_times = deque(maxlen=30)


def on_trial_start():
    """[3-1조] 시행 기록 시작 때 live_record가 부름: 기다리던 프레임은 버리고 판정 상태를 처음으로
    (녹화 영상을 처음부터 트는 것과 같게 frame_idx 0, 방향 기억 없음, depth 새로 계산). 새 epoch 번호를 돌려줌"""
    with _lock:
        _latest["data"] = None
    return pipe.reset()


def inference_worker():
    sent_count = 0            # 서버를 켠 뒤 처리한 프레임 수 (사이트 전송 번호, 시행과 상관없이 계속 늘어남)
    last_depth_ms = 0.0
    last_depth_color = None   # [3-1조] depth를 건너뛴 프레임에도 오른쪽 depth 화면을 유지 (사이트 화면 폭이 들쭉날쭉하지 않게)
    while not _stop.is_set():
        with _lock:
            data = _latest["data"]
            recv_ms = _latest["recv_ms"]
            _latest["data"] = None
        if data is None:
            time.sleep(0.002)
            continue

        server_t0 = time.time()

        d0 = time.time()
        arr = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        decode_ms = (time.time() - d0) * 1000
        if frame is None:
            continue
        h0, w0 = frame.shape[:2]   # 폰이 보낸 원본 크기 (폰 회신 좌표 기준)

        # [3-1조] 축소 → YOLO → depth → 판정 (judge.py — 녹화 영상 run_model.py와 같은 코드)
        try:
            out = pipe.step(frame)
        except Exception as error:   # [10/30] 프레임 하나 에러로 처리 스레드가 죽어 폰 회신이 끊기지 않게
            print(f"\n[처리 에러] {type(error).__name__}: {error} (이 프레임 건너뜀)")
            if not _site_state.get("step_traced"):
                _site_state["step_traced"] = True
                import traceback
                traceback.print_exc()
            continue
        frame, W, H, sx, j = out["frame"], out["W"], out["H"], out["sx"], out["j"]
        yolo_ms, depth_ms_now = out["yolo_ms"], out["depth_ms"]
        depth_color = None
        if depth_ms_now is not None:
            last_depth_ms = depth_ms_now
            depth_color = last_depth_color = judge.depth_colormap(out["depth_m"])
        sent_count += 1

        def dets_at(scale):   # scale=sx → 원본 좌표(폰), scale=1 → 줄인 화면 좌표(시연 사이트)
            return [{
                "label": label,
                "conf": round(conf, 2),
                "box": [round(v * scale, 1) for v in box],
                "dist": round(dist, 2) if dist is not None else None,   # 박스 거리 (judge.box_distance)
            } for box, dist, _, label, conf in j["box_results"]]
        dets = dets_at(sx)
        judge_info = {   # 이 서버의 판정
            "warn": bool(j["obs_warn"]),
            "obstacle_m": round(j["obs_m"], 2) if j["obs_m"] is not None else None,
            "source": j["obs_src"],
            "direction": j["direction"],
            "motor": j["motor"] or None,
            "vib_count": j["vib_count"] or None,
        }

        # 뷰어: run_model.py와 같은 표시 (박스·경로·장애물 거리·모터·방향 칸)
        annotated = frame.copy()
        judge.draw_boxes(annotated, j["box_results"])
        judge.draw_path(annotated, H, W, j["obs_m"], j["obs_src"], j["obs_warn"], j["near_pt"], j["motor"])
        if judge.ZONE_MODE != "off":
            judge.draw_zones(annotated, W, H)
        live_record.draw_rec(annotated, H)

        server_ms = (time.time() - server_t0) * 1000

        result_json = json.dumps({
            "server_ms": round(server_ms, 1),
            "yolo_ms": round(yolo_ms, 1),
            "depth_ms": round(last_depth_ms, 1),
            "frame_w": w0, "frame_h": h0,
            "judge": judge_info,   # [3-1조] 폰 앱 파싱 때문에 dets보다 앞에 둠
            "dets": dets,          # 반드시 맨 끝 (폰 앱이 "dets": [ ... ] 를 끝까지 읽음)
        }, ensure_ascii=False)
        with _result_lock:
            _pending_result["json"] = result_json

        # [3-1조] 시행 기록 중이면 frames.csv(run_model.py와 같은 형식) + latency.csv에 한 줄
        live_record.add(recv_ms, out, {
            "recv_to_start_ms": round(server_t0 * 1000 - recv_ms, 1), "decode_ms": round(decode_ms, 1),
            "yolo_ms": round(yolo_ms, 1), "depth_ms": "" if depth_ms_now is None else round(depth_ms_now, 1),
            "server_ms": round(server_ms, 1)})

        proc_times.append(time.time() * 1000)
        fps = 0.0
        if len(proc_times) >= 2:
            span = (proc_times[-1] - proc_times[0]) / 1000.0
            if span > 0:
                fps = (len(proc_times) - 1) / span

        lag_ms = time.time() * 1000 - recv_ms
        warn_txt = f" ⚠ {j['direction'] or ''}→{j['motor']}" if j["obs_warn"] else ""
        obs_txt = f"{j['obs_m']:.2f}m" if j["obs_m"] is not None else "-"
        print(f"fps={fps:4.1f} server={server_ms:6.1f}ms (yolo={yolo_ms:5.1f} depth={last_depth_ms:6.1f}) "
              f"lag={lag_ms:6.1f}ms objs={len(dets)} 장애물={obs_txt}{warn_txt}" + " " * 8, end="\r")

        if depth_color is None and last_depth_color is not None and last_depth_color.shape[:2] == annotated.shape[:2]:
            depth_color = last_depth_color   # [3-1조] 건너뛴 프레임은 직전 depth 화면
        if depth_color is not None:
            combined = np.hstack([annotated, depth_color])
        else:
            combined = annotated

        latest_view["annotated"] = combined

        # [3-1조] 사이트로는 원래 맥북 버전과 같은 형식으로 보냄
        #   - 이미지: 원래처럼 폰이 보낸 원본 크기 (판정은 640 폭에서 했으므로 다시 키움)
        #   - 데이터: 원래 항목(server_ms, yolo_ms, depth_ms, frame_w, frame_h, dets)만, 좌표도 원본 기준
        #     SITE_JUDGE=1 로 실행하면 judge 항목(경고·방향·모터)도 같이 보냄
        site_img = combined
        if sx != 1.0:
            site_img = cv2.resize(combined, (int(round(combined.shape[1] * sx)), h0))
        site_data = json.loads(result_json)
        judge_row = None
        if SITE_JUDGE:
            # [3-1조] frames.csv 한 줄과 같은 함수·같은 값으로 만들어 사이트 record에 넣음 (좌표는 원본 픽셀)
            judge_row = judge.frame_row("", out["idx"], 0, sx, j["box_results"],
                                        (j["obs_m"], j["obs_src"], j["obs_warn"], j["obs_target"]),
                                        j["direction"], (j["motor"], j["vib_count"]), yolo_ms, depth_ms_now)
        else:
            site_data.pop("judge", None)
        try:
            live_mirror.remember_bgr(site_img, site_data, sent_count, judge_row)
            if not _site_state["sent"]:
                _site_state["sent"] = True
                print(f"\n[사이트] 첫 프레임 넘김 (이미지 {site_img.shape[1]}x{site_img.shape[0]}, "
                      f"judge 항목 {'포함' if SITE_JUDGE else '없음'})")
        except Exception as error:
            print("사이트 영상 전송 실패:", type(error).__name__, error)
            if not _site_state["traced"]:
                _site_state["traced"] = True
                import traceback
                traceback.print_exc()

@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    print("\n폰 연결됨.")
    try:
        while True:
            data = await ws.receive_bytes()
            with _lock:
                _latest["data"] = data
                _latest["recv_ms"] = time.time() * 1000
            with _result_lock:
                rj = _pending_result["json"]
                _pending_result["json"] = None
            if rj is not None:
                await ws.send_text(rj)
    except WebSocketDisconnect:
        print("\n폰 연결 끊김.")
    except Exception as e:
        print(f"\n에러: {e}")


async def viewer():
    if not SHOW_WINDOW:
        return
    while True:
        img = latest_view["annotated"]
        if img is not None:
            cv2.imshow("Blind Helper - YOLO + Metric Depth", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
        await asyncio.sleep(0.01)


@app.on_event("startup")
async def on_startup():
    threading.Thread(target=inference_worker, daemon=True).start()
    # [3-1조] 시행 기록: 시작할 때 on_trial_start로 판정 상태를 처음으로
    live_record.start_console(args={"device": DEVICE, "depth_every_n": DEPTH_EVERY_N},
                              pipeline=judge.pipeline_info(DEVICE, DEPTH_EVERY_N, using_openvino),
                              on_start=on_trial_start)
    asyncio.create_task(viewer())


@app.on_event("shutdown")
async def on_shutdown():
    live_record.stop_trial()   # [3-1조] 기록 중이면 저장하고 끝냄
    _stop.set()
