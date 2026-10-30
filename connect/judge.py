"""
judge.py — 장애물 경고 판정 (녹화 영상 run_model.py와 실시간 server.py가 함께 쓰는 코드)

녹화 영상으로 오차를 보고 고칠 곳은 이 파일 하나뿐.
  고친 뒤: run_model.py로 같은 영상을 다시 돌려 확인 → 커밋 → (서버 컴퓨터에서 받기) → 서버 재시작
  run_model.py·server.py·live_record.py에는 판정·처리 순서가 없음 (입력 받기와 결과 내보내기만)

이 파일에 있는 것
  설정값            맨 위 (판정 규칙, 장치, depth 몇 프레임마다)
  판정 함수         4.1 ~ 4.11 (원래 run_model.py에서 옮김), judge_frame()이 한 프레임 판정
  FramePipeline     프레임 한 장 → 축소 → YOLO → depth → judge_frame (녹화·실시간이 같은 순서로 처리)
  기록 공통 함수     run_meta.json, summary.json 만들기 (녹화·실시간이 같은 형식으로 저장)

[10/26 통합] 축소·YOLO 호출·depth 건너뛰기·장치 선택·모델 불러오기·방향 유지를 FramePipeline으로,
            run_meta/summary 만들기를 공통 함수로 옮김. 판정 규칙(계산 결과)은 바꾸지 않음
            RUN_FORMAT schema_version 2: video_source(live/file), server_ms_median 칸 추가
[10/30 YOLO26] 탐지 yolo11n → yolo26s, depth Depth Anything V2 → YOLO26-depth (DEPTH_BACKEND).
            판정 규칙·상수·기록 형식은 그대로. 바뀐 건 모델 불러오기(load_models)와 depth_map뿐
            DEPTH_BACKEND=dav2 로 켜면 예전 모델로 똑같이 돌아감 (베이스라인 비교용)
"""

import json
import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

# 맥(MPS)에서 지원 안 되는 연산은 CPU로 대신 계산 (torch를 불러오기 전에 설정해야 함)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))   # 모델 파일·runs 폴더 기준 (실행 위치와 상관없이 같게)

# ===== 설정 (run_model.py에서 옮김) =====
FRAME_WIDTH = 640      # 프레임 축소 폭 (저장 좌표는 원본 픽셀로 되돌림)
# [10/30 YOLO26] 탐지 모델 yolo11n → yolo26s (ultralytics 8.4.104 이상 필요, 처음 실행 때 자동 다운로드)
#   한 번만 바꿔 볼 때: YOLO_MODEL=yolo26n.pt python run_model.py ...  (예전 것: YOLO_MODEL=yolo11n.pt)
MODEL_PATH = os.environ.get("YOLO_MODEL", "yolo26s.pt")   # 2-4
CONF_THRES = 0.4
IMG_SIZE = 416         # YOLO imgsz (프레임 축소 폭 640과 별개)
USE_DEPTH = True

# [10/30 YOLO26] depth 모델: Depth Anything V2 → YOLO26-depth (미터 단위 거리, ultralytics 안에서 바로 돌아감)
#   DEPTH_BACKEND "yolo26"(기본) / "dav2"(예전 Depth Anything V2, 비교·베이스라인 재현용)
#   한 번만 바꿔 볼 때: DEPTH_BACKEND=dav2 python run_model.py ...  /  DEPTH_MODEL=yolo26s-depth.pt ...
DEPTH_BACKEND = os.environ.get("DEPTH_BACKEND", "yolo26")
DEPTH_MODELS = {
    "yolo26": "yolo26n-depth.pt",                                        # n/s/m/l/x. 클수록 정확·느림
    "dav2": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",   # 실내 (실외: ...-Outdoor-Small-hf)
}
if DEPTH_BACKEND not in DEPTH_MODELS:
    raise SystemExit(f"DEPTH_BACKEND는 {list(DEPTH_MODELS)} 중 하나여야 해요 (지금: {DEPTH_BACKEND})")
DEPTH_MODEL = os.environ.get("DEPTH_MODEL", DEPTH_MODELS[DEPTH_BACKEND])
# YOLO26-depth 입력 크기. 공개 가중치는 768로 학습됨 → 768이 가장 정확. 줄이면 빨라지지만 거리 오차가 커질 수 있음
#   (32의 배수. 바꿔 볼 때: DEPTH_IMG_SIZE=640 ...)  dav2에는 안 쓰임 (dav2는 518 고정)
DEPTH_IMG_SIZE = int(os.environ.get("DEPTH_IMG_SIZE", 768))
# 맥 전용: 1이면 YOLO26-depth를 CoreML(.mlpackage)로 바꿔서 Neural Engine으로 돌림 (처음 한 번 변환에 1~2분)
#   PyTorch(mps)와 소수점 결과가 조금 다를 수 있음 → run_meta.json에 기록됨
DEPTH_COREML = os.environ.get("DEPTH_COREML") == "1"
DEPTH_SCALE = 1.0      # 거리 보정 계수: 모델 출력에 곱함 (예: 실제 2m인데 4m로 나오면 0.5). 모델 바꾸면 다시 잴 것

# ===== 실행 환경 (녹화·실시간 공통) =====
DEVICE = "auto"        # 연산 장치: "auto"(cuda → mps → cpu 순서로 자동) / "cuda" / "mps" / "cpu". 장치가 다르면 소수점 끝자리가 다를 수 있음
DEPTH_EVERY_N = 1      # N프레임마다 depth 1번 (1 = 매 프레임). 결과가 달라지니 녹화·실시간 같은 값으로
                       # 한 번만 바꿔 볼 때: run_model.py --depth_every_n 3 / 서버 DEPTH_EVERY_N=3 uvicorn ...

# ===== 경로 영역 (2-1 A) =====
PATH_WIDTH_M = 0.8     # 경로 폭(m): 몸 중심에서 좌우 0.4m
HFOV_DEG = 70.0        # 카메라 가로 화각, PATH_ROI 계산에만 쓰임
WARN_M = 3             # 2-9 B: 장애물 거리가 이보다 가까우면 경고 (전에는 2)
PATH_Y = (0.25, 0.75)  # 경로 세로 범위 (천장·바닥 제외)
# x 폭 = PATH_WIDTH_M / (2 × WARN_M × tan(HFOV/2)) → 3m, 70°면 약 0.19 (x 0.40~0.60)
PATH_ROI = (0.40, 0.60, 0.25, 0.75)   # (x 시작, x 끝, y 시작, y 끝) 화면 비율. 2-9 WARN_M에 맞춰 다시 계산 (전에는 0.36~0.64)
NEAR_PERCENTILE = 5    # 2-1 A: 경로 영역 안 depth 하위 몇 % 백분위를 경로 거리로 쓸지
# DEPTH 경고 위치 (2-11 B, 4.3): 가장 가까운 한 점 대신, 경로 거리에서 NEAR_BAND_M 안에 있는 픽셀들의 중심.
# 장애물이 경로를 다 덮으면 가운데(정면)가 되어 한 점처럼 프레임마다 흔들리지 않음. None이면 한 점
NEAR_BAND_M = 0.3

# ===== [3-1조] 클래스 모드 (시험·비교용) =====
# 실행할 때 TARGET_LABELS=bottle 처럼 주면 그 클래스의 YOLO 박스만 장애물로 봄 (깊이 통로 판정은 안 씀)
#   예) TARGET_LABELS=bottle RUNNER=지은 uvicorn server:app --host 0.0.0.0 --port 8001
#       TARGET_LABELS=bottle python run_model.py --video 1019_front_01.mp4 --runner 지은
#   여러 개면 쉼표: TARGET_LABELS=bottle,cup
# 안 주면(None) 원래대로 깊이 기반 판정. 거리는 어느 모드든 깊이(박스 영역) 값
TARGET_LABELS = frozenset(s.strip() for s in os.environ["TARGET_LABELS"].split(",") if s.strip()) \
    if os.environ.get("TARGET_LABELS") else None
print(f"판정 모드: {'클래스 ' + ','.join(sorted(TARGET_LABELS)) + ' 박스만' if TARGET_LABELS else '깊이 기반 (기본)'}")

# ===== YOLO 박스 거리 (클래스는 판정에 쓰지 않음) =====
BOX_SHRINK = 0.2       # 2-2 B: 박스를 상하좌우 이 비율만큼 줄인 영역에서 거리를 잼
BOX_PERCENTILE = 30    # 2-2 B: 줄인 영역 안 depth의 하위 몇 % 백분위를 박스 거리로 쓸지
BOX_OVERLAP_MIN = 0.3  # 2-7 B: 박스 폭 중 경로 x 범위 안에 들어온 비율이 이 이상이면 경로 안

# ===== 방향 칸 (2-5) =====
ZONE_MODE = "path"     # 2-5 B(ZONE_PATH_SPLIT 비율), 판정용 (test_model2.py, test_model.py는 A): "off"(사용 안 함) / "screen"(A, 화면 전체 3등분) / "path"(B, 경로 영역 3등분)
ZONE_DRAW = "screen"   # 화면에 그리는 칸 (보기 편하게 화면 전체 3등분). direction·motor 판정은 ZONE_MODE로만 함
ZONE_PATH_SPLIT = (1 / 3, 2 / 3)  # "path" 칸 경계: PATH_ROI x 범위의 비율 (3등분). 0.40~0.60이면 0.467, 0.533
HOLD_DIRECTION = True  # 경고가 이어지는 동안 첫 프레임의 direction(→ motor)을 유지. 경고가 끊기면 다음 경고에서 새로 정함 (2-12 B, 4.8)
ZONE_KO = {"LEFT": "왼쪽", "FRONT": "정면", "RIGHT": "오른쪽"}   # RUN_FORMAT 값

# ===== 진동 명령 (2-10, 4.11) =====
MOTOR_MAP = {"왼쪽": "right", "오른쪽": "left", "정면": "neck"}   # 2-10: 피할 방향 모터 (score.py와 같은 값)
VIB_COUNT = {1: 1, 2: 2}              # 2-10: 레벨 → 진동 횟수 (1 주의 = 1회, 2 경고 = 2회)
MOTOR_TEXT_SCALE = 1.6                # 화면의 "-> NECK" 글자 크기 (장애물 거리 글자는 0.6)

# ===== run 결과 저장 (4.9) =====
RUNS_DIR = "runs"
SCHEMA_VERSION = 2     # RUN_FORMAT과 맞춤 (2: video_source, server_ms_median 추가)
CSV_COLUMNS = [
    "trial", "frame_idx", "t_ms", "processed", "n_dets", "labels", "dets_json",
    "target_label", "target_conf", "target_cx", "dist", "ttc", "level", "level_name",
    "direction", "beep", "vibrate_ms", "tts", "yolo_ms", "depth_ms", "target_box",
    "motor", "vib_count",
]
CHOICES = {"2-1": "A", "2-2": "B", "2-3": "경고하지 않음", "2-5": ZONE_MODE, "2-7": "B",
           "2-9": "B (PATH_ROI 다시 계산)", "2-10": "A (정면 → neck)",
           "2-11": "B" if NEAR_BAND_M is not None else "A", "2-12": "B" if HOLD_DIRECTION else "A"}


# ----- 4.1 Depth 거리 지도 -----
class Yolo26Depth:
    """YOLO26-depth. 프레임(BGR) → 거리 지도 (m, float32). ultralytics가 BGR numpy를 바로 받음"""

    def __init__(self, model, device, using_coreml=False):
        self.model = model
        self.kwargs = dict(imgsz=DEPTH_IMG_SIZE, verbose=False)
        if not using_coreml:   # CoreML은 장치를 스스로 고름
            self.kwargs["device"] = {"cuda": 0, "mps": "mps", "cpu": "cpu"}[device]

    def __call__(self, frame):
        r = self.model.predict(frame, **self.kwargs)[0]
        return r.depth.data.squeeze().float().cpu().numpy()


class DAV2Depth:
    """예전 Depth Anything V2 (transformers pipeline). 프레임(BGR) → 거리 지도 (m)"""

    def __init__(self, pipe):
        self.pipe = pipe

    def __call__(self, frame):
        out = self.pipe(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        return out["predicted_depth"].squeeze().float().cpu().numpy()


def depth_map(frame, W, H, depth_pipe):
    """모델 출력 → 프레임 크기로 resize → DEPTH_SCALE 곱하기. 결과 shape (H, W), 단위 m
    depth_pipe: Yolo26Depth 또는 DAV2Depth (load_models가 DEPTH_BACKEND에 맞춰 만듦)"""
    d = depth_pipe(frame).astype(np.float32)
    if d.shape != (H, W):
        d = cv2.resize(d, (W, H))
    return d * DEPTH_SCALE


# ----- 4.2 YOLO 박스 거리 -----
def box_distance(dm, box):
    """박스를 상하좌우 BOX_SHRINK씩 줄인 영역에서 depth 하위 BOX_PERCENTILE 백분위(m)"""
    H, W = dm.shape
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * BOX_SHRINK, (y2 - y1) * BOX_SHRINK
    sx1, sy1 = max(0, int(round(x1 + dx))), max(0, int(round(y1 + dy)))
    sx2, sy2 = min(W, int(round(x2 - dx))), min(H, int(round(y2 - dy)))
    if sx2 - sx1 < 1 or sy2 - sy1 < 1:   # 줄인 영역이 1픽셀보다 작으면 원래 박스 영역
        sx1, sy1 = max(0, int(x1)), max(0, int(y1))
        sx2, sy2 = min(W, int(x2)), min(H, int(y2))
    region = dm[sy1:sy2, sx1:sx2]
    if region.size == 0:
        return None
    return float(np.percentile(region, BOX_PERCENTILE))


def box_in_path(box, W, H):
    """2-7 B: 박스 폭 중 경로 x 범위 안 비율 ≥ BOX_OVERLAP_MIN 이고 세로로 조금이라도 겹치면 경로 안"""
    x1, y1, x2, y2 = box
    if x2 - x1 <= 0:
        return False
    rx0, rx1 = PATH_ROI[0] * W, PATH_ROI[1] * W
    ry0, ry1 = PATH_ROI[2] * H, PATH_ROI[3] * H
    x_overlap = max(0, min(x2, rx1) - max(x1, rx0)) / (x2 - x1)
    y_overlap = min(y2, ry1) > max(y1, ry0)
    return x_overlap >= BOX_OVERLAP_MIN and y_overlap


def box_results_of(r, dm, W, H):
    """YOLO 결과 → [(box, 거리, in_path, label, conf), ...]
    label·conf는 표시(4.5)·기록(4.9)용으로만 들고 다님 (판정에 쓰지 않음). depth가 아직 없으면 거리 None"""
    results = []
    for b in r.boxes:
        box = tuple(map(int, b.xyxy[0].tolist()))
        dist = box_distance(dm, box) if dm is not None else None
        results.append((box, dist, box_in_path(box, W, H),
                        r.names[int(b.cls)], float(b.conf)))
    return results


# ----- 4.3 경로 장애물 (depth만, YOLO와 무관) -----
def roi_pixels(H, W):
    x0, x1, y0, y1 = PATH_ROI
    return int(W * x0), int(W * x1), int(H * y0), int(H * y1)


def path_obstacle(dm):
    """경로 영역 depth 하위 NEAR_PERCENTILE 백분위(m)와 장애물 위치 (x, y)
    위치: 백분위 값 + NEAR_BAND_M 안 픽셀들의 중심 (NEAR_BAND_M이 None이면 백분위 값에 가장 가까운 픽셀)"""
    H, W = dm.shape
    px0, px1, py0, py1 = roi_pixels(H, W)
    smooth = cv2.medianBlur(dm.astype(np.float32), 5)         # 튀는 픽셀 하나에 끌려가지 않게 먼저 적용
    roi = smooth[py0:py1, px0:px1]
    if roi.size == 0:
        return None, None
    near = float(np.percentile(roi, NEAR_PERCENTILE))
    if NEAR_BAND_M is not None:   # 가까운 픽셀들의 중심 (백분위 값 이하 픽셀이 항상 있어서 비지 않음)
        ys, xs = np.nonzero(roi <= near + NEAR_BAND_M)
        return near, (px0 + int(round(xs.mean())), py0 + int(round(ys.mean())))
    ry, rx = np.unravel_index(np.argmin(np.abs(roi - near)), roi.shape)   # 백분위 값에 가장 가까운 픽셀
    return near, (px0 + int(rx), py0 + int(ry))


# ----- 4.4 결과 합치기 -----
def merge_result(path_res, boxes):
    """4.3 거리 + 경로 안 YOLO 박스 거리 중 최솟값 → (거리, 출처, 경고, 대상)
    대상: 경고일 때만. YOLO면 그 박스 항목, DEPTH면 4.3의 장애물 위치 (x, y)"""
    path_m, near_pt = path_res
    cands = []
    if path_m is not None and not TARGET_LABELS:   # [3-1조] 클래스 모드면 깊이 통로는 후보 아님
        cands.append((path_m, "DEPTH", near_pt))
    for item in boxes:
        _, dist, in_path, label, _ = item
        if TARGET_LABELS and label not in TARGET_LABELS:   # [3-1조] 클래스 모드: 지정 클래스만
            continue
        if in_path and dist is not None:      # 2-3: 경로 밖 박스는 후보 아님
            cands.append((dist, "YOLO", item))
    if not cands:
        return None, None, False, None
    dist, source, target = min(cands, key=lambda c: c[0])
    warn = dist < WARN_M
    return dist, source, warn, (target if warn else None)


# ----- 4.5 화면 표시 -----
BOX_COLOR = (255, 200, 0)   # 모든 클래스 같은 색


def put_text(img, text, org, color, scale=0.6, thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def draw_boxes(img, boxes):
    """경로 안 박스는 굵게, 경로 밖은 얇게. 모든 박스에 '클래스 확률 거리' 글자 (예: chair 0.82 1.85m)"""
    for (x1, y1, x2, y2), dist, in_path, label, conf in boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, 4 if in_path else 1)
        text = f"{label} {conf:.2f}" + (f" {dist:.2f}m" if dist is not None else "")
        put_text(img, text, (x1, max(15, y1 - 6)), BOX_COLOR, 0.5)


def draw_path(img, H, W, obs_m, obs_src, obs_warn, near_pt, motor=""):
    """경로 영역 테두리 (정상 초록 / 경고 빨강) + 상단 장애물 거리 글자 + 경고면 그 아래 큰 모터 글자 + 경로 안 가장 가까운 점"""
    px0, px1, py0, py1 = roi_pixels(H, W)
    color = (0, 0, 255) if obs_warn else (0, 200, 0)
    cv2.rectangle(img, (px0, py0), (px1, py1), color, 2)
    if obs_m is not None:   # cv2.putText는 한글을 못 그려서 영어로 표시
        put_text(img, f"OBSTACLE {obs_m:.2f}m ({obs_src})", (8, 52), color)   # 위쪽 칸 이름(y 25)과 안 겹치게
    if motor:   # 4.11 진동 명령: 멀리서도 보이게 크게
        put_text(img, f"-> {motor.upper()}", (8, 52 + int(40 * MOTOR_TEXT_SCALE)), color,
                 MOTOR_TEXT_SCALE, max(2, int(2 * MOTOR_TEXT_SCALE)))
    if near_pt is not None:
        cv2.circle(img, near_pt, 6, color, 2)


def depth_colormap(dm):
    """depth 지도 → 보기용 컬러 화면 (밝을수록 멂)"""
    dnorm = cv2.normalize(dm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(dnorm, cv2.COLORMAP_INFERNO)


# ----- 4.8 방향 칸 (dividebox.py에서 옮김) -----
def zone_limits(W, mode=ZONE_MODE):
    """칸 경계 x좌표 2개. "screen"(2-5 A): 화면 전체 3등분 / "path"(B): PATH_ROI x 범위를 ZONE_PATH_SPLIT 비율로 나눔"""
    if mode == "path":
        x0, x1 = PATH_ROI[0] * W, PATH_ROI[1] * W
        return int(x0 + (x1 - x0) * ZONE_PATH_SPLIT[0]), int(x0 + (x1 - x0) * ZONE_PATH_SPLIT[1])
    return W // 3, 2 * W // 3


def get_zone(center_x, W):
    """x좌표가 어느 칸인지: LEFT / FRONT / RIGHT"""
    left, right = zone_limits(W)
    if center_x < left:
        return "LEFT"
    elif center_x > right:
        return "RIGHT"
    return "FRONT"


def draw_zones(img, W, H, color=(255, 255, 0)):
    """세로선 2개 + 칸 이름 표시 (ZONE_DRAW 기준, 판정 경계와 다를 수 있음)"""
    left, right = zone_limits(W, ZONE_DRAW)
    cv2.line(img, (left, 0), (left, H), color, 2)
    cv2.line(img, (right, 0), (right, H), color, 2)
    for text, x in (("LEFT", 10), ("FRONT", left + 10), ("RIGHT", right + 10)):
        cv2.putText(img, text, (x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def target_direction(obs_src, obs_target, W):
    """경고 대상의 방향 (왼쪽/정면/오른쪽). 경고가 없으면 None
    YOLO면 박스 가로 중앙, DEPTH면 4.3의 가장 가까운 점 x로 판정 (경고 여부는 바꾸지 않음)
    2-5 사용 안 함("off")이면 PATH_ROI x 범위 3등분 기준 (4.9 기본 규칙)"""
    if obs_target is None:
        return None
    if obs_src == "YOLO":
        x1, _, x2, _ = obs_target[0]
        x = (x1 + x2) / 2
    else:
        x = obs_target[0]
    if ZONE_MODE == "off":
        x0, x1 = PATH_ROI[0] * W, PATH_ROI[1] * W
        left, right = x0 + (x1 - x0) / 3, x0 + 2 * (x1 - x0) / 3
        return ZONE_KO["LEFT" if x < left else "RIGHT" if x > right else "FRONT"]
    return ZONE_KO[get_zone(x, W)]


# ----- 4.11 진동 명령 -----
def vibration(level, direction):
    """경고(level ≥ 2)이고 방향이 있을 때만 (motor, vib_count). 아니면 ("", "")
    motor는 피할 방향 모터 (장애물 왼쪽 → right, 오른쪽 → left, 정면 → neck)"""
    if level < 2 or not direction:
        return "", ""
    return MOTOR_MAP[direction], VIB_COUNT[level]


def check_trial(trial):
    """<MMDD>_<front|left|right|none>_<2자리> 형식인지"""
    p = trial.split("_")
    return (len(p) == 3 and len(p[0]) == 4 and p[0].isdigit()
            and p[1] in ("front", "left", "right", "none") and len(p[2]) == 2 and p[2].isdigit())


def frame_row(trial, idx, t_ms, sx, boxes, obs, direction, vib, yolo_ms, depth_ms):
    """frames.csv 한 줄 (dict). t_ms는 영상 시각(ms, frame_time_ms 결과)
    좌표는 640 축소 화면 기준이므로 sx(원본 폭 / 축소 폭)를 곱해 원본 픽셀로 저장"""
    def scale(b):
        return [round(v * sx) for v in b]

    obs_m, obs_src, obs_warn, obs_target = obs
    row = dict.fromkeys(CSV_COLUMNS, "")   # ttc, beep, vibrate_ms, tts는 빈칸
    row.update(
        trial=trial,
        frame_idx=idx,
        t_ms=t_ms,
        processed=1,
        n_dets=len(boxes),
        labels=";".join(label for _, _, _, label, _ in boxes),
        dets_json=json.dumps([{"label": label, "conf": round(conf, 3), "box": scale(box),
                               "dist": None if dist is None else round(dist, 3)}
                              for box, dist, _, label, conf in boxes], ensure_ascii=False),
        dist="" if obs_m is None else round(obs_m, 3),
        level=2 if obs_warn else 0,
        level_name="경고" if obs_warn else "없음",
        yolo_ms=round(yolo_ms, 1),
        depth_ms="" if depth_ms is None else round(depth_ms, 1),
    )
    if obs_warn:
        if obs_src == "YOLO":
            box, _, _, label, conf = obs_target
            row.update(target_label=label, target_conf=round(conf, 3),
                       target_cx=round((box[0] + box[2]) / 2 * sx), target_box=" ".join(map(str, scale(box))))
        else:
            row.update(target_label="DEPTH", target_cx=round(obs_target[0] * sx))
        row["direction"] = direction or ""
        row["motor"], row["vib_count"] = vib   # 4.11 (경고 아니면 빈칸 그대로)
    return row


# ----- 한 프레임 판정 (원래 run_model.py main 루프 4.2~4.11) -----
def judge_frame(r, depth_m, W, H, held_dir):
    """YOLO 결과 r + depth 지도 → 판정 결과 dict, 다음 프레임에 넘길 held_dir
    (HOLD_DIRECTION: 경고가 이어지는 동안 첫 방향 유지, 끊기면 초기화)"""
    # 4.2 YOLO 박스마다 거리 + 경로 안/밖
    box_results = box_results_of(r, depth_m, W, H)
    # 4.3 경로 장애물 (depth가 있을 때만)
    path_m, near_pt = path_obstacle(depth_m) if depth_m is not None else (None, None)
    # 4.4 결과 합치기: 경로 depth vs 경로 안 YOLO 박스 중 가까운 쪽
    obs_m, obs_src, obs_warn, obs_target = merge_result((path_m, near_pt), box_results)
    # 4.8 경고 대상의 방향 칸
    direction = target_direction(obs_src, obs_target, W)
    if HOLD_DIRECTION:
        if obs_warn:
            direction = held_dir = held_dir or direction
        else:
            held_dir = None
    # 4.11 방향 → 진동 명령 (경고일 때만, 레벨 2)
    motor, vib_count = vibration(2 if obs_warn else 0, direction)
    return dict(box_results=box_results, path_m=path_m, near_pt=near_pt, obs_m=obs_m, obs_src=obs_src,
                obs_warn=obs_warn, obs_target=obs_target, direction=direction,
                motor=motor, vib_count=vib_count), held_dir


def alarm_constants():
    """run_meta.json에 남길 판정 설정값 (run_model.py write_meta와 같은 항목)"""
    return {
        "FRAME_WIDTH": FRAME_WIDTH, "CONF_THRES": CONF_THRES, "USE_DEPTH": USE_DEPTH,
        "MODEL_PATH": MODEL_PATH, "IMG_SIZE": IMG_SIZE, "PATH_WIDTH_M": PATH_WIDTH_M,
        "HFOV_DEG": HFOV_DEG, "WARN_M": WARN_M, "PATH_Y": list(PATH_Y), "PATH_ROI": list(PATH_ROI),
        "BOX_SHRINK": BOX_SHRINK, "BOX_PERCENTILE": BOX_PERCENTILE, "NEAR_PERCENTILE": NEAR_PERCENTILE,
        "NEAR_BAND_M": NEAR_BAND_M, "HOLD_DIRECTION": HOLD_DIRECTION,
        "BOX_OVERLAP_MIN": BOX_OVERLAP_MIN, "ZONE_MODE": ZONE_MODE, "ZONE_DRAW": ZONE_DRAW,
        "ZONE_PATH_SPLIT": list(ZONE_PATH_SPLIT), "MOTOR_MAP": MOTOR_MAP, "VIB_COUNT": VIB_COUNT,
        "RUNS_DIR": RUNS_DIR, "SCHEMA_VERSION": SCHEMA_VERSION, "DEPTH_SCALE": DEPTH_SCALE, "choices": CHOICES,
        "TARGET_LABELS": sorted(TARGET_LABELS) if TARGET_LABELS else None,   # [3-1조] None = 깊이 기반(기본)
    }


# ===== 장치·모델 불러오기 (녹화·실시간 공통) =====
def resolve_device(pref=DEVICE):
    """설정값을 실제로 쓸 수 있는 장치 이름("cuda"/"mps"/"cpu")으로 바꿈 (원래 server.py에서 옮김)"""
    import torch
    has_cuda = torch.cuda.is_available()
    has_mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    if pref == "auto":
        if has_cuda:
            return "cuda"
        if has_mps:
            return "mps"
        return "cpu"
    if pref == "cuda" and not has_cuda:
        print("경고: CUDA를 쓸 수 없음 -> " + ("mps" if has_mps else "cpu") + "로 진행")
        return "mps" if has_mps else "cpu"
    if pref == "mps" and not has_mps:
        print("경고: MPS를 쓸 수 없음 -> cpu로 진행")
        return "cpu"
    return pref


def sync_device(device):
    """GPU/MPS는 비동기로 돌아가므로, 시간을 잴 때는 계산이 끝날 때까지 기다림 (시간만 바뀌고 판정은 안 바뀜)"""
    if device in ("cuda", "mps"):
        import torch
        (torch.cuda if device == "cuda" else torch.mps).synchronize()


def load_models(device, use_openvino=False, depth_optional=False):
    """YOLO와 depth 모델을 불러옴 → (yolo, depth_pipe, using_openvino)
    모델 파일은 judge.py가 있는 폴더 기준 (어디서 실행해도 같은 파일). fuse()는 하지 않음
    use_openvino: Intel CPU 전용 가속, device가 cpu일 때만 씀 (결과가 조금 다를 수 있어 run_meta.json에 기록됨)
    depth_optional: True면 depth 모델을 못 불러와도 depth 없이 진행 (실시간 서버), False면 오류로 멈춤 (녹화 평가)"""
    from ultralytics import YOLO
    model_path = os.path.join(HERE, MODEL_PATH)
    yolo, using_openvino = None, False
    if use_openvino and device == "cpu":
        ov_dir = model_path.replace(".pt", "_openvino_model")
        try:
            if not os.path.isdir(ov_dir):
                print("OpenVINO 변환 중...")
                YOLO(model_path).export(format="openvino", imgsz=IMG_SIZE)
            yolo = YOLO(ov_dir, task="detect")
            using_openvino = True
        except Exception as e:
            print(f"OpenVINO 실패 ({e}) -> 일반 모델로 진행")
    if yolo is None:
        print(f"YOLO 로딩 중... ({MODEL_PATH}, 처음엔 자동 다운로드)")
        yolo = YOLO(model_path)
    depth_pipe = None
    if USE_DEPTH:
        print(f"Depth 로딩 중... ({DEPTH_BACKEND}: {DEPTH_MODEL}, 처음엔 다운로드로 1~2분 걸릴 수 있어요)")
        try:
            depth_pipe = _load_depth(device)
        except Exception as e:
            if not depth_optional:
                raise
            print(f"Depth 로드 실패 ({e}) -> depth 없이 진행")
    return yolo, depth_pipe, using_openvino


_depth_runtime = {"coreml": False}   # pipeline_info에 남길 실제 실행 방식


def _load_depth(device):
    if DEPTH_BACKEND == "dav2":
        from transformers import pipeline
        # transformers 장치 표기: cuda → 0, 맥 → "mps", cpu → -1
        return DAV2Depth(pipeline("depth-estimation", model=DEPTH_MODEL,
                                  device={"cuda": 0, "mps": "mps", "cpu": -1}[device]))
    from ultralytics import YOLO
    path = DEPTH_MODEL if os.path.isabs(DEPTH_MODEL) else os.path.join(HERE, DEPTH_MODEL)
    if DEPTH_COREML:
        ml_path = path.replace(".pt", f"_{DEPTH_IMG_SIZE}.mlpackage")
        try:
            if not os.path.exists(ml_path):
                print(f"CoreML 변환 중... (imgsz {DEPTH_IMG_SIZE}, 처음 한 번만)")
                out = YOLO(path).export(format="coreml", imgsz=DEPTH_IMG_SIZE, half=True)
                os.replace(out, ml_path)
            _depth_runtime["coreml"] = True
            return Yolo26Depth(YOLO(ml_path, task="depth"), device, using_coreml=True)
        except Exception as e:
            print(f"CoreML 실패 ({e}) -> PyTorch({device})로 진행")
    return Yolo26Depth(YOLO(path), device)


# ===== 프레임 한 장 처리 (녹화·실시간 공통) =====
class FramePipeline:
    """프레임 한 장 → 판정. 녹화(run_model.py)와 실시간(server.py)이 이 순서 그대로 처리함
      폭 FRAME_WIDTH로 축소 → YOLO → depth (depth_every_n 프레임마다, 나머지는 직전 depth) → judge_frame
    프레임 번호와 경고 방향 유지(held_dir)도 여기서만 관리함
    reset(): 처음 상태로 (영상을 처음부터 트는 것과 같음). 실시간 시행을 시작할 때 부름"""

    def __init__(self, yolo, depth_pipe, device, depth_every_n=DEPTH_EVERY_N, using_openvino=False,
                 keep_going_on_depth_error=False):
        self.yolo, self.depth_pipe, self.device = yolo, depth_pipe, device
        self.depth_every_n = max(1, int(depth_every_n))
        self.keep_going_on_depth_error = keep_going_on_depth_error   # True(서버): depth 오류면 직전 depth로 계속
        self.predict_kwargs = dict(imgsz=IMG_SIZE, conf=CONF_THRES, verbose=False)
        if not using_openvino:
            # ultralytics 장치 표기: cuda → 0, 맥 → "mps", cpu → "cpu"
            self.predict_kwargs["device"] = {"cuda": 0, "mps": "mps", "cpu": "cpu"}[device]
        self._lock = threading.Lock()   # 서버에서 처리 중에 reset()이 끼어들지 않게
        self.epoch = 0                  # reset()할 때마다 1씩 늘어남 (어느 시행 상태로 처리한 프레임인지 구분)
        self._reset()

    def _reset(self):
        self.frame_idx = 0
        self.held_dir = None
        self.depth_m = None
        self.epoch += 1

    def reset(self):
        """프레임 번호 0, 방향 기억 없음, depth 없음으로 되돌림. 새 epoch 번호를 돌려줌"""
        with self._lock:
            self._reset()
            return self.epoch

    def step(self, frame):
        """원본 프레임(BGR) → 결과 dict
        idx: 이 프레임 번호 (reset 뒤 0부터), frame/W/H: 축소한 화면, sx: 원본 폭 / 축소 폭,
        r: YOLO 결과, depth_m: 이번에 쓴 depth 지도, yolo_ms, depth_ms(이번에 depth를 안 돌렸으면 None), j: judge_frame 결과"""
        with self._lock:
            h, w = frame.shape[:2]
            if w > FRAME_WIDTH:
                frame = cv2.resize(frame, (FRAME_WIDTH, int(h * FRAME_WIDTH / w)))
            H, W = frame.shape[:2]
            idx = self.frame_idx

            t0 = time.time()
            r = self.yolo.predict(frame, **self.predict_kwargs)[0]
            sync_device(self.device)
            yolo_ms = (time.time() - t0) * 1000

            # 4.1 Depth: N프레임마다 새로 계산, 나머지는 직전 depth
            depth_ms = None
            if self.depth_pipe is not None and idx % self.depth_every_n == 0:
                t1 = time.time()
                try:
                    self.depth_m = depth_map(frame, W, H, self.depth_pipe)
                    sync_device(self.device)
                    depth_ms = (time.time() - t1) * 1000
                except Exception as e:
                    if not self.keep_going_on_depth_error:
                        raise
                    print(f"\ndepth 에러: {e} (직전 depth로 판정)")
            self.frame_idx += 1

            # 4.2 ~ 4.11 판정
            j, self.held_dir = judge_frame(r, self.depth_m, W, H, self.held_dir)
            return dict(idx=idx, epoch=self.epoch, frame=frame, W=W, H=H, orig_wh=(w, h), sx=w / W,
                        r=r, depth_m=self.depth_m, yolo_ms=yolo_ms, depth_ms=depth_ms, j=j)


# ===== run 결과 저장 공통 (녹화 run_model.py·실시간 live_record.py가 같은 형식으로) =====
def make_run_dir(runner, trial, started=None):
    """runs/<runner>/<시행ID>_<MMDD>_<HHMMSS>/ 를 새로 만듦 (있으면 덮어쓰지 않고 오류)"""
    started = started or datetime.now()
    run_dir = Path(HERE) / RUNS_DIR / runner / f"{trial}_{started:%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json(path, data):
    with open(path, "x", encoding="utf-8") as f:   # "x": 이미 있으면 덮어쓰지 않음
        json.dump(data, f, ensure_ascii=False, indent=2)


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else "unknown"
    except Exception:
        return "unknown"


def pipeline_info(device, depth_every_n, using_openvino=False):
    """run_meta.json의 pipeline: 결과에 영향을 주는 실행 환경"""
    info = {"device": device, "yolo_weights": MODEL_PATH, "depth_backend": DEPTH_BACKEND,
            "depth_model": DEPTH_MODEL, "depth_imgsz": DEPTH_IMG_SIZE if DEPTH_BACKEND == "yolo26" else 518,
            "depth_coreml": _depth_runtime["coreml"],
            "conf": CONF_THRES, "imgsz": IMG_SIZE, "depth_every_n": depth_every_n, "openvino": using_openvino}
    for name in ("torch", "ultralytics", "transformers"):
        try:
            info[name] = __import__(name).__version__
        except Exception:
            pass
    info["opencv"] = cv2.__version__
    return info


def run_meta(video_source, args, pipeline, runner, video_file, run_at=None):
    """run_meta.json 내용. video_source: "file"(녹화 영상) / "live"(실시간)"""
    return {
        "run_at": (run_at or datetime.now()).isoformat(timespec="seconds"),
        "video_source": video_source,
        "args": args,
        "pipeline": pipeline,
        "alarm_constants": alarm_constants(),
        "runner": runner,
        "git_commit": git_commit(),
        "video_file": video_file,
        "schema_version": SCHEMA_VERSION,
    }


def summary(rows, trial, video_source, video, frame_size, video_fps, duration_sec, wall_sec,
            expected_dir=None, gt_2m_sec=None, note=None, server_ms=None):
    """summary.json 내용 (frames.csv 줄 rows에서 다시 셀 수 있는 값만)
    server_ms: 실시간이면 프레임별 서버 처리 시간 목록, 녹화면 None → server_ms_median null"""
    done = [r for r in rows if r["processed"] == 1]
    label_counts = {}
    for r in done:
        for label in filter(None, r["labels"].split(";")):
            label_counts[label] = label_counts.get(label, 0) + 1
    first2 = next((r for r in done if r["level"] == 2), None)
    first_vib = next((r for r in rows if r["motor"] != ""), None)   # 4.11
    t_sec = lambda r: r["t_ms"] / 1000 if r["t_ms"] != "" else None
    med = lambda v: round(float(np.median(v)), 1) if v else None
    return {
        "trial": trial,
        "video_source": video_source,
        "video": video,
        "frames": len(rows),
        "processed": len(done),
        "frame_size": frame_size,
        "video_fps": video_fps,
        "duration_sec": duration_sec,
        "frames_with_det": sum(1 for r in done if r["n_dets"] > 0),
        "frames_with_target": sum(1 for r in done if r["level"] == 2),
        "label_counts": label_counts,
        "first_level1": None,
        "first_level2": None if first2 is None else {
            "t_sec": t_sec(first2), "dist": first2["dist"], "direction": first2["direction"] or None},
        "first_level3": None,
        "max_level": max((r["level"] for r in done), default=0),
        "yolo_ms_median": med([r["yolo_ms"] for r in done]),
        "depth_ms_median": med([r["depth_ms"] for r in done if r["depth_ms"] != ""]),
        "server_ms_median": None if server_ms is None else med(server_ms),
        "wall_sec": round(wall_sec, 2),
        "expected_dir": expected_dir,
        "gt_2m_sec": gt_2m_sec,
        "note": note,
        "first_vibration": None if first_vib is None else {
            "t_sec": t_sec(first_vib), "dist": first_vib["dist"],
            "direction": first_vib["direction"] or None, "target_label": first_vib["target_label"]},
    }
