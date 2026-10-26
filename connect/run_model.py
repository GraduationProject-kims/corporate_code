"""
로컬 테스트: 맥북 카메라(또는 영상 파일)로 YOLO + Metric Depth 거리 계산·속도 측정
폰·서버 연결 없이 맥북 혼자 실행. 영상은 저장하지 않고 숫자만 터미널에 표시.
명세: yolo_depth_distance_spec_no_a4.md 4.1 ~ 4.9, 4.11  (2-5만 B: 판정은 경로 영역 3등분, 화면 칸은 전체 3등분)
      + DEPTH 경고 위치를 가까운 픽셀들의 중심으로 (NEAR_BAND_M) + 경고 동안 첫 방향 유지 (HOLD_DIRECTION)
      (명세 2-5 B, 2-11 B, 2-12 B)
  4.1 depth 거리 지도 (m) × DEPTH_SCALE
  4.2 YOLO 박스마다 거리(줄인 영역 하위 30%) + 경로 안/밖(겹침 30% 이상)
  4.3 경로 영역에서 depth만으로 가장 가까운 거리 (median blur 후 하위 5%)
  4.4 4.3 거리와 경로 안 YOLO 박스 거리 중 최솟값 → 장애물 거리·출처·경고·대상
  4.5 화면 표시 / 4.6 터미널 출력·요약 / 4.7 키 조작 / 4.8 방향 칸 / 4.9 run 결과 저장 / 4.11 진동 명령

실행:  python test_model3.py --runner 내이름 [--video 1009_front_01.mp4 --trial 1009_front_01
                                           --expected_dir 정면 --gt_2m_sec 4.2 --note "메모" --depth_every_n 1]
       --video 0 이면 맥북 카메라 (run 폴더 안 만듦, --runner 없어도 됨)
       영상 파일이면 runs/<runner>/<시행ID>_<MMDD>_<HHMMSS>/ 에 run_meta.json, frames.csv, summary.json 저장 (4.9)
조작:  영상 창을 클릭해 둔 상태에서
       q      종료 (창을 껐다면 터미널에서 Ctrl+C)
       space  일시정지 / 다시 재생
       마우스 클릭  그 지점의 depth 값(m)을 화면과 터미널에 표시 (일시정지 중에도 됨)
       영상 파일은 끝까지 재생되면 자동 종료
종료 시 터미널에 평균·최소·최대 요약이 나옴

[거리 보정 방법]
  줄자로 거리를 아는 물체(예: 2m 앞 벽)를 찍고, 일시정지 후 그 물체를 클릭
  → 터미널에 나온 값이 실제보다 크면 DEPTH_SCALE = 실제거리 / 화면값 으로 설정
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # MPS가 지원 안 하는 연산은 CPU로 자동 처리

import argparse
import csv
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import transformers
import ultralytics
from PIL import Image
from transformers import pipeline
from ultralytics import YOLO


import judge
from judge import *   # 판정 설정값·함수 (judge.py). 판정은 여기서 바꾸지 말고 judge.py에서

# ===== 설정 =====
HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "1009_left_01.mp4")   # 영상 파일 / 맥북 카메라로 하려면 SOURCE = 0
LOOP_VIDEO = False     # True면 영상 끝나면 처음부터 반복 (run 기록 중에는 꺼짐)
SHOW_WINDOW = True     # False면 창 없이 숫자만 (창을 안 띄우면 조금 더 빨라짐)
PRINT_EVERY_SEC = 1.0  # 몇 초마다 터미널에 한 줄씩 찍을지
WARMUP_FRAMES = 5      # 처음 몇 프레임은 모델 준비 때문에 느려서 요약에서 제외



# ===== 실행 인자 (4.9) =====
parser = argparse.ArgumentParser()
parser.add_argument("--video", default=str(SOURCE), help="영상 경로 (기본값: 코드의 SOURCE). 0이면 맥북 카메라")
parser.add_argument("--trial", default=None, help="시행ID (기본값: 영상 파일명에서 확장자를 뺀 것)")
parser.add_argument("--runner", default=None, help="실행한 사람 이름 (영상 입력이면 필수)")
parser.add_argument("--expected_dir", default=None, help="정답 방향")
parser.add_argument("--gt_2m_sec", type=float, default=None, help="실제로 2m에 도달한 영상 시각(초)")
parser.add_argument("--note", default=None)
parser.add_argument("--depth_every_n", type=int, default=1, help="N프레임마다 depth 1회 (1이면 매 프레임)")
args = parser.parse_args()
DEPTH_EVERY_N = max(1, args.depth_every_n)
VIDEO = 0 if args.video == "0" else args.video   # 실제로 여는 입력 (코드의 SOURCE를 고치지 않고 --video로 바꿈)
if isinstance(VIDEO, str) and not args.runner:
    raise SystemExit("영상 입력이면 --runner 이름 이 필요해요. 예: python test_model.py --runner 내이름 --video 1009_front_01.mp4")

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"계산 장치: {DEVICE}")

# ===== 모델 로딩 =====
print(f"YOLO 로딩 중... ({MODEL_PATH}, 처음엔 자동 다운로드)")
yolo = YOLO(os.path.join(HERE, MODEL_PATH))

depth_pipe = None
if USE_DEPTH:
    print("Depth 로딩 중... (처음엔 모델 다운로드로 1~2분 걸릴 수 있어요)")
    depth_pipe = pipeline("depth-estimation", model=DEPTH_MODEL, device=DEVICE)
print("모델 로드 완료.\n")

# ===== 입력 열기 =====
cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    raise SystemExit(
        f"입력을 열 수 없어요: {VIDEO}\n"
        "- 영상이면 파일 이름·확장자가 정확한지 확인하세요.\n"
        "- 카메라면 시스템 설정 → 개인정보 보호 및 보안 → 카메라 권한을 확인하세요."
    )
is_file = isinstance(VIDEO, str)
if is_file:
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(f"입력 영상: {os.path.basename(VIDEO)}  (원본 {src_fps:.0f}fps, {src_frames}프레임)\n")

# ===== 측정값 기록 =====
yolo_list, depth_list, frame_list, objs_list = [], [], [], []
path_list = []         # 경로 depth 거리(m) (4.3)
obs_list = []          # 장애물 거리(m) (4.4)
warn_count = 0         # 경고가 뜬 프레임 수
warn_by_src = {"YOLO": 0, "DEPTH": 0}   # 출처별 경고 횟수 (터미널 요약에만, summary.json엔 넣지 않음)
warn_by_motor = {"left": 0, "right": 0, "neck": 0}   # 모터별 경고 횟수 (4.6, 터미널 요약에만)

frame_idx = 0
held_dir = None        # HOLD_DIRECTION: 이번 경고 구간의 첫 방향
depth_m = None
depth_color = None
last_print = time.time()
window_count = 0       # 이번 1초 동안 처리한 프레임 수
window_start = time.time()
start_time = None

WIN = "Local Test (q: quit, space: pause, click: depth)"
click_pt = None        # 마지막으로 클릭한 지점 (왼쪽 화면 기준 좌표)
cur_W = FRAME_WIDTH    # 현재 프레임 폭 (클릭 좌표 계산용)


def stats(values):
    if not values:
        return "  -"
    a = np.array(values)
    return f"평균 {a.mean():7.1f} | 최소 {a.min():7.1f} | 최대 {a.max():7.1f}"


def point_depth(dm, x, y, r=2):
    """(x, y) 주변 (2r+1)x(2r+1) 픽셀의 중앙값"""
    patch = dm[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
    return float(np.median(patch)) if patch.size > 0 else None





# ----- 4.9 run 결과 저장 -----


def make_run_dir(runner, trial):
    """runs/<runner>/<시행ID>_<MMDD>_<HHMMSS>/ 를 새로 만듦 (있으면 덮어쓰지 않고 오류)"""
    run_dir = Path(HERE) / RUNS_DIR / runner / f"{trial}_{datetime.now():%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else "unknown"
    except Exception:
        return "unknown"


def write_json(path, data):
    with open(path, "x", encoding="utf-8") as f:   # "x": 이미 있으면 덮어쓰지 않음
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_meta(run_dir, args, runner):
    write_json(run_dir / "run_meta.json", {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "pipeline": {
            "device": DEVICE, "yolo_weights": MODEL_PATH, "depth_model": DEPTH_MODEL,
            "conf": CONF_THRES, "imgsz": IMG_SIZE,
            "torch": torch.__version__, "ultralytics": ultralytics.__version__,
            "transformers": transformers.__version__, "opencv": cv2.__version__,
        },
        "alarm_constants": judge.alarm_constants(),
        "runner": runner,
        "git_commit": git_commit(),
        "video_file": os.path.basename(VIDEO),
        "schema_version": SCHEMA_VERSION,
    })




pos_msec_ok = True     # False면 POS_MSEC를 못 믿어서 frame_idx × 1000 / fps로 되돌린 상태


def frame_time_ms(idx, pos_msec, fps):
    """영상 시각(ms): cap.read() 직후 읽은 CAP_PROP_POS_MSEC를 반올림
    폰으로 옮긴 영상은 프레임 간격이 일정하지 않을 수 있어서 프레임 번호로 계산하지 않음
    첫 프레임 뒤에도 0이나 음수면 그 뒤로는 frame_idx × 1000 / fps로 되돌리고 한 번만 경고"""
    global pos_msec_ok
    if pos_msec_ok and idx > 0 and pos_msec <= 0:
        pos_msec_ok = False
        print("[주의] 영상 시각(POS_MSEC)을 읽을 수 없어요. t_ms를 frame_idx × 1000 / fps로 계산합니다.")
    if pos_msec_ok:
        return round(max(pos_msec, 0))
    return round(idx * 1000 / fps) if fps else ""


def write_summary(run_dir, rows, trial, wall_sec, args):
    done = [r for r in rows if r["processed"] == 1]
    label_counts = {}
    for r in done:
        for label in filter(None, r["labels"].split(";")):
            label_counts[label] = label_counts.get(label, 0) + 1
    first2 = next((r for r in done if r["level"] == 2), None)
    first_vib = next((r for r in rows if r["motor"] != ""), None)   # 4.11: frames.csv에서 다시 셀 수 있는 값만
    yolo_ms = [r["yolo_ms"] for r in done]
    depth_ms = [r["depth_ms"] for r in done if r["depth_ms"] != ""]
    write_json(run_dir / "summary.json", {
        "trial": trial,
        "video": os.path.basename(VIDEO),
        "frames": len(rows),
        "processed": len(done),
        "frame_size": list(src_size),
        "video_fps": src_fps,
        "duration_sec": round(src_frames / src_fps, 3) if src_fps else None,
        "frames_with_det": sum(1 for r in done if r["n_dets"] > 0),
        "frames_with_target": sum(1 for r in done if r["level"] == 2),
        "label_counts": label_counts,
        "first_level1": None,
        "first_level2": None if first2 is None else {
            "t_sec": first2["t_ms"] / 1000 if first2["t_ms"] != "" else None,
            "dist": first2["dist"], "direction": first2["direction"] or None},
        "first_level3": None,
        "max_level": max((r["level"] for r in done), default=0),
        "yolo_ms_median": round(float(np.median(yolo_ms)), 1) if yolo_ms else None,
        "depth_ms_median": round(float(np.median(depth_ms)), 1) if depth_ms else None,
        "wall_sec": round(wall_sec, 2),
        "expected_dir": args.expected_dir,
        "gt_2m_sec": args.gt_2m_sec,
        "note": args.note,
        "first_vibration": None if first_vib is None else {
            "t_sec": first_vib["t_ms"] / 1000 if first_vib["t_ms"] != "" else None,
            "dist": first_vib["dist"], "direction": first_vib["direction"] or None,
            "target_label": first_vib["target_label"]},
    })


# ----- 4.7 키·마우스 -----
def on_mouse(event, x, y, flags, param):
    global click_pt
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    x = x % cur_W                      # 오른쪽(depth) 화면을 눌러도 같은 위치로 계산
    click_pt = (x, y)
    if depth_m is not None and 0 <= y < depth_m.shape[0]:
        v = point_depth(depth_m, x, y)
        print(f"[클릭] ({x}, {y}) → {v:.2f}m")


def draw_marker(img, pt, text, color):
    cv2.drawMarker(img, pt, color, cv2.MARKER_CROSS, 18, 2)
    put_text(img, text, (pt[0] + 8, max(15, pt[1] - 8)), color, 0.55)


def compose(left_base, right_base):
    """클릭 표시를 얹어서 좌우 화면을 합침 (일시정지 중에도 다시 그리려고 분리)"""
    left, right = left_base.copy(), right_base.copy()
    if click_pt is not None and depth_m is not None and click_pt[1] < depth_m.shape[0]:
        v = point_depth(depth_m, *click_pt)
        for img in (left, right):
            draw_marker(img, click_pt, f"{v:.2f}m", (255, 255, 0))
    return np.hstack([left, right])


if SHOW_WINDOW:
    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_mouse)

# ===== run 폴더 (4.9, 영상 파일일 때만) =====
run_dir, csv_file, csv_writer, rows, trial = None, None, None, [], None
if is_file:
    trial = args.trial or Path(VIDEO).stem
    if args.trial and args.trial != Path(VIDEO).stem:
        print(f"[주의] --trial '{args.trial}'가 영상 파일명 '{Path(VIDEO).stem}'과 달라요 "
              "(RUN_FORMAT §6: 폴더명과 영상 파일명의 시행ID가 같아야 함). 그대로 진행합니다.")
    args.trial = trial   # run_meta.json의 args에 실제로 쓴 시행ID가 들어가게
    if not check_trial(trial):
        print(f"[주의] 시행ID '{trial}'가 <MMDD>_<front|left|right|none>_<2자리> 형식이 아니에요. 그대로 진행합니다.")
    if LOOP_VIDEO:
        print("[주의] run 기록 중에는 반복 재생하지 않아요 (같은 프레임이 두 번 기록되지 않게)")
        LOOP_VIDEO = False
    run_dir = make_run_dir(args.runner, trial)
    write_meta(run_dir, args, args.runner)
    csv_file = open(run_dir / "frames.csv", "x", newline="", encoding="utf-8-sig")
    csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
    csv_writer.writeheader()
    print(f"run 폴더: {run_dir}\n")


def save_row(row):
    rows.append(row)
    csv_writer.writerow(row)
    csv_file.flush()


run_t0 = time.time()
quit_now = False
try:
    while True:
        f0 = time.time()
        ok, frame = cap.read()
        pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)   # 4.9 t_ms: 읽은 직후 그 프레임의 영상 시각
        if not ok:
            if is_file and LOOP_VIDEO:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            print("\n입력이 끝났어요." if is_file else "\n카메라 프레임을 읽지 못했어요.")
            break

        # 크기 줄이기
        h, w = frame.shape[:2]
        if w > FRAME_WIDTH:
            frame = cv2.resize(frame, (FRAME_WIDTH, int(h * FRAME_WIDTH / w)))
        H, W = frame.shape[:2]
        cur_W = W
        cur_idx = frame_idx

        # YOLO
        t0 = time.time()
        r = yolo.predict(frame, imgsz=IMG_SIZE, conf=CONF_THRES, device=DEVICE, verbose=False)[0]
        yolo_ms = (time.time() - t0) * 1000

        # 4.1 Depth (N프레임마다, 중간 프레임은 직전 depth 재사용)
        depth_ms = None
        if depth_pipe is not None and frame_idx % DEPTH_EVERY_N == 0:
            t1 = time.time()
            depth_m = depth_map(frame, W, H, depth_pipe)
            depth_ms = (time.time() - t1) * 1000
            if SHOW_WINDOW:
                dnorm = cv2.normalize(depth_m, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                depth_color = cv2.applyColorMap(dnorm, cv2.COLORMAP_INFERNO)
        frame_idx += 1

        # 4.2~4.11 판정 (judge.py — 실시간 서버와 같은 코드)
        j, held_dir = judge.judge_frame(r, depth_m, W, H, held_dir)
        box_results, path_m, near_pt = j["box_results"], j["path_m"], j["near_pt"]
        obs_m, obs_src, obs_warn, obs_target = j["obs_m"], j["obs_src"], j["obs_warn"], j["obs_target"]
        direction, motor, vib_count = j["direction"], j["motor"], j["vib_count"]

        # 4.9 frames.csv 한 줄 (일시정지 전에 한 번만 씀)
        if csv_writer:
            save_row(frame_row(trial, cur_idx, frame_time_ms(cur_idx, pos_msec, src_fps), w / W, box_results,
                               (obs_m, obs_src, obs_warn, obs_target), direction, (motor, vib_count),
                               yolo_ms, depth_ms))

        # 4.5 화면 표시 (선택)
        if SHOW_WINDOW:
            # 왼쪽: 원본 + 박스, 오른쪽: depth 컬러맵, 양쪽: 경로 테두리 + 장애물 글자 + 방향 칸
            annotated = frame.copy()
            draw_boxes(annotated, box_results)
            right = depth_color.copy() if depth_color is not None else np.zeros_like(annotated)
            for img in (annotated, right):
                draw_path(img, H, W, obs_m, obs_src, obs_warn, near_pt, motor)
                if ZONE_MODE != "off":
                    draw_zones(img, W, H)

            # depth 화면에 전체 범위 표시 (밝을수록 멂)
            if depth_m is not None:
                rng = f"depth {depth_m.min():.1f}~{depth_m.max():.1f}m (bright=far)"
                cv2.putText(right, rng, (8, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
                cv2.putText(right, rng, (8, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow(WIN, compose(annotated, right))
            key = cv2.waitKey(1) & 0xFF

            # 4.7 일시정지: 멈춘 동안은 측정 시간에서 빼고, 클릭 표시만 계속 갱신 (frames.csv엔 안 씀)
            if key == ord(" "):
                p0 = time.time()
                print("\n일시정지 — 클릭하면 거리 확인, space로 재생")
                while True:
                    cv2.imshow(WIN, compose(annotated, right))
                    k = cv2.waitKey(30) & 0xFF
                    if k == ord(" "):
                        break
                    if k == ord("q"):
                        quit_now = True
                        break
                paused = time.time() - p0
                f0 += paused
                last_print += paused
                window_start += paused
                if start_time is not None:
                    start_time += paused
            if key == ord("q") or quit_now:
                print("\nq를 눌러 종료했어요.")
                break

        frame_ms = (time.time() - f0) * 1000   # 프레임 하나 전체 처리 시간

        # 측정값 기록 (워밍업 이후만)
        if frame_idx > WARMUP_FRAMES:
            if start_time is None:
                start_time = time.time()
                window_start = start_time
            yolo_list.append(yolo_ms)
            frame_list.append(frame_ms)
            objs_list.append(len(r.boxes))
            if depth_ms is not None:
                depth_list.append(depth_ms)
            if path_m is not None:
                path_list.append(path_m)
            if obs_m is not None:
                obs_list.append(obs_m)
            if obs_warn:
                warn_count += 1
                warn_by_src[obs_src] += 1
                if motor:
                    warn_by_motor[motor] += 1
            window_count += 1

            # 4.6 1초마다 한 줄 출력
            now = time.time()
            if now - last_print >= PRINT_EVERY_SEC:
                fps_now = window_count / (now - window_start)
                elapsed = now - start_time
                depth_txt = f"{np.mean(depth_list[-5:]):6.0f}ms" if depth_list else "     -"
                path_txt = f"{path_m:5.2f}m" if path_m is not None else "    -"
                range_txt = f" (전체 {depth_m.min():.1f}~{depth_m.max():.1f}m)" if depth_m is not None else ""
                obs_txt = f"{obs_m:5.2f}m {obs_src:5s}" if obs_m is not None else "    -      "
                warn_txt = ("  ⚠ 경고" + (f" {direction}" if direction else "")
                            + (f" → 모터 {motor}" if motor else "")) if obs_warn else ""
                print(f"[{elapsed:6.1f}s] fps {fps_now:5.1f} | yolo {yolo_ms:5.0f}ms | "
                      f"depth {depth_txt} | 프레임 {frame_ms:6.0f}ms | objs {len(r.boxes)} | "
                      f"경로 {path_txt}{range_txt} | 장애물 {obs_txt}{warn_txt}")
                last_print = now
                window_start = now
                window_count = 0
        else:
            print(f"워밍업 중... ({frame_idx}/{WARMUP_FRAMES})")

except KeyboardInterrupt:
    print("\nCtrl+C로 종료했어요.")

cap.release()
cv2.destroyAllWindows()

# 4.9 summary.json (q·Ctrl+C로 중간 종료해도 씀)
if run_dir is not None:
    csv_file.close()
    write_summary(run_dir, rows, trial, time.time() - run_t0, args)

# ===== 4.6 요약 =====
total_sec = (time.time() - start_time) if start_time else 0.0
n = len(frame_list)
print("\n" + "=" * 62)
print(" 측정 요약 (워밍업 제외)")
print("=" * 62)
print(f" 장치          : {DEVICE}")
print(f" 입력          : {os.path.basename(VIDEO) if is_file else '맥북 카메라'}")
print(f" 설정          : 입력 {FRAME_WIDTH}px, YOLO {IMG_SIZE}px, depth {DEPTH_EVERY_N}프레임마다, 창 {'켜짐' if SHOW_WINDOW else '꺼짐'}")
print(f" 측정 프레임   : {n}장 / {total_sec:.1f}초")
print(f" 평균 fps      : {n / total_sec:.1f}" if total_sec > 0 else " 평균 fps      : -")
print(f" YOLO (ms)     : {stats(yolo_list)}")
print(f" Depth (ms)    : {stats(depth_list)}  ({len(depth_list)}회)")
print(f" 프레임 전체(ms): {stats(frame_list)}")
print(f" 탐지 물체 수  : {stats(objs_list)}")
print(f" 경로 거리(m)  : {stats(path_list)}")
print(f" 장애물 거리(m): {stats(obs_list)}")
warn_ratio = (warn_count / n * 100) if n else 0.0
print(f" 경고 프레임   : {warn_count}장 ({warn_ratio:.1f}%, 기준 {WARN_M}m 미만, 보정 x{DEPTH_SCALE})")
print(f" 출처별 경고   : YOLO {warn_by_src['YOLO']}회 / DEPTH {warn_by_src['DEPTH']}회")
print(f" 모터별 경고   : left {warn_by_motor['left']} / right {warn_by_motor['right']} / neck {warn_by_motor['neck']}")
if run_dir is not None:
    print(f" run 폴더      : {run_dir}")
print("=" * 62)
