"""
judge.py — 장애물 경고 판정 (녹화 영상 run_model.py와 실시간 live_server.py가 함께 쓰는 코드)

run_model.py(연수 코드, 명세 4.1~4.11)의 설정값과 판정·표시·기록 함수를 그대로 옮긴 것.
판정 규칙을 바꿀 땐 이 파일만 고치면 녹화 평가와 실시간이 같이 바뀜.
옮기면서 바꾼 것은 두 가지뿐:
  1) depth_map()이 depth 모델을 인자로 받음 (원래는 전역 변수)
  2) 한 프레임 판정 순서(원래 main 루프의 4.2~4.11)를 judge_frame() 하나로 묶음
"""

import json
import os

import cv2
import numpy as np
from PIL import Image

# ===== 설정 (run_model.py에서 옮김) =====
FRAME_WIDTH = 640      # 프레임 축소 폭 (저장 좌표는 원본 픽셀로 되돌림)
MODEL_PATH = "yolo11n.pt"   # 2-4
CONF_THRES = 0.4
IMG_SIZE = 416         # YOLO imgsz (프레임 축소 폭 640과 별개)
USE_DEPTH = True
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"     # 실내
# DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"  # 실외
DEPTH_SCALE = 1.0      # 거리 보정 계수: 모델 출력에 곱함 (예: 실제 2m인데 4m로 나오면 0.5)

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
SCHEMA_VERSION = 1     # RUN_FORMAT과 맞춤
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
def depth_map(frame, W, H, depth_pipe):
    """모델 출력 → 프레임 크기로 resize → DEPTH_SCALE 곱하기. 결과 shape (H, W), 단위 m"""
    out = depth_pipe(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    d = out["predicted_depth"].squeeze().cpu().numpy()
    return cv2.resize(d, (W, H)) * DEPTH_SCALE


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
        "MODEL_PATH": MODEL_PATH, "IMG_SIZE": IMG_SIZE, "PATH_WIDTH_M": PATH_WIDTH_M,
        "HFOV_DEG": HFOV_DEG, "WARN_M": WARN_M, "PATH_Y": list(PATH_Y), "PATH_ROI": list(PATH_ROI),
        "BOX_SHRINK": BOX_SHRINK, "BOX_PERCENTILE": BOX_PERCENTILE, "NEAR_PERCENTILE": NEAR_PERCENTILE,
        "NEAR_BAND_M": NEAR_BAND_M, "HOLD_DIRECTION": HOLD_DIRECTION,
        "BOX_OVERLAP_MIN": BOX_OVERLAP_MIN, "ZONE_MODE": ZONE_MODE, "ZONE_DRAW": ZONE_DRAW,
        "ZONE_PATH_SPLIT": list(ZONE_PATH_SPLIT), "MOTOR_MAP": MOTOR_MAP, "VIB_COUNT": VIB_COUNT,
        "RUNS_DIR": RUNS_DIR, "SCHEMA_VERSION": SCHEMA_VERSION, "DEPTH_SCALE": DEPTH_SCALE, "choices": CHOICES,
        "TARGET_LABELS": sorted(TARGET_LABELS) if TARGET_LABELS else None,   # [3-1조] None = 깊이 기반(기본)
    }
