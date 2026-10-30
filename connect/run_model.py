"""
로컬 테스트: 맥북 카메라(또는 영상 파일)로 YOLO + Metric Depth 거리 계산·속도 측정
폰·서버 연결 없이 맥북 혼자 실행. 영상은 저장하지 않고 숫자만 터미널에 표시.
명세: yolo_depth_distance_spec_no_a4.md 4.1 ~ 4.9, 4.11  (2-5만 B: 판정은 경로 영역 3등분, 화면 칸은 전체 3등분)
      + DEPTH 경고 위치를 가까운 픽셀들의 중심으로 (NEAR_BAND_M) + 경고 동안 첫 방향 유지 (HOLD_DIRECTION)
      (명세 2-5 B, 2-11 B, 2-12 B)

[10/26 통합] 처리·판정은 전부 judge.py (실시간 server.py와 같은 코드)
  축소 → YOLO → depth → 판정(4.1~4.11)은 judge.FramePipeline.step() 한 줄
  이 파일에는 영상 읽기, 화면 창(4.5·4.7), 터미널 출력(4.6), 결과 저장(4.9)만 남음
  오차를 고칠 땐 이 파일이 아니라 judge.py를 고칠 것 (여기서 설정값을 바꿔도 판정은 안 바뀜)

실행:  python run_model.py --runner 내이름 [--video 1009_front_01.mp4 --trial 1009_front_01
                                          --expected_dir 정면 --gt_2m_sec 4.2 --note "메모" --depth_every_n 1]
       --video 0 이면 맥북 카메라 (run 폴더 안 만듦, --runner 없어도 됨)
       영상 파일이면 runs/<runner>/<시행ID>_<MMDD>_<HHMMSS>/ 에 run_meta.json, frames.csv, summary.json 저장 (4.9)
       --depth_every_n을 안 주면 judge.py의 DEPTH_EVERY_N
조작:  영상 창을 클릭해 둔 상태에서
       q      종료 (창을 껐다면 터미널에서 Ctrl+C)
       space  일시정지 / 다시 재생
       마우스 클릭  그 지점의 depth 값(m)을 화면과 터미널에 표시 (일시정지 중에도 됨)
       영상 파일은 끝까지 재생되면 자동 종료
종료 시 터미널에 평균·최소·최대 요약이 나옴

[거리 보정 방법]
  줄자로 거리를 아는 물체(예: 2m 앞 벽)를 찍고, 일시정지 후 그 물체를 클릭
  → 터미널에 나온 값이 실제보다 크면 judge.py의 DEPTH_SCALE = 실제거리 / 화면값 으로 설정
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # MPS가 지원 안 하는 연산은 CPU로 자동 처리

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

import judge   # 판정·처리 순서·설정값은 전부 judge.py

# ===== 설정 (보기·출력용. 판정에는 영향 없음) =====
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
parser.add_argument("--depth_every_n", type=int, default=None,
                    help=f"N프레임마다 depth 1회 (기본값: judge.py의 DEPTH_EVERY_N = {judge.DEPTH_EVERY_N})")
args = parser.parse_args()
DEPTH_EVERY_N = max(1, args.depth_every_n if args.depth_every_n is not None else judge.DEPTH_EVERY_N)
args.depth_every_n = DEPTH_EVERY_N   # run_meta.json의 args에 실제로 쓴 값이 들어가게
VIDEO = 0 if args.video == "0" else args.video   # 실제로 여는 입력 (코드의 SOURCE를 고치지 않고 --video로 바꿈)
if isinstance(VIDEO, str) and not args.runner:
    raise SystemExit("영상 입력이면 --runner 이름 이 필요해요. 예: python run_model.py --runner 내이름 --video 1009_front_01.mp4")

DEVICE = judge.resolve_device(judge.DEVICE)
print(f"계산 장치: {DEVICE}")

# ===== 모델 로딩 (judge.py, 실시간 서버와 같은 방법) =====
yolo, depth_pipe, using_openvino = judge.load_models(DEVICE)
pipe = judge.FramePipeline(yolo, depth_pipe, DEVICE, DEPTH_EVERY_N, using_openvino=using_openvino)
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

frame_idx = 0          # 지금까지 처리한 프레임 수 (워밍업 판단용)
depth_m = None
depth_color = None
last_print = time.time()
window_count = 0       # 이번 1초 동안 처리한 프레임 수
window_start = time.time()
start_time = None

WIN = "Local Test (q: quit, space: pause, click: depth)"
click_pt = None        # 마지막으로 클릭한 지점 (왼쪽 화면 기준 좌표)
cur_W = judge.FRAME_WIDTH    # 현재 프레임 폭 (클릭 좌표 계산용)


def stats(values):
    if not values:
        return "  -"
    a = np.array(values)
    return f"평균 {a.mean():7.1f} | 최소 {a.min():7.1f} | 최대 {a.max():7.1f}"


def point_depth(dm, x, y, r=2):
    """(x, y) 주변 (2r+1)x(2r+1) 픽셀의 중앙값"""
    patch = dm[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
    return float(np.median(patch)) if patch.size > 0 else None


# ----- 4.9 영상 시각 -----
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
    judge.put_text(img, text, (pt[0] + 8, max(15, pt[1] - 8)), color, 0.55)


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
    if not judge.check_trial(trial):
        print(f"[주의] 시행ID '{trial}'가 <MMDD>_<front|left|right|none>_<2자리> 형식이 아니에요. 그대로 진행합니다.")
    if LOOP_VIDEO:
        print("[주의] run 기록 중에는 반복 재생하지 않아요 (같은 프레임이 두 번 기록되지 않게)")
        LOOP_VIDEO = False
    run_dir = judge.make_run_dir(args.runner, trial)
    judge.write_json(run_dir / "run_meta.json", judge.run_meta(
        "file", vars(args), judge.pipeline_info(DEVICE, DEPTH_EVERY_N, using_openvino),
        args.runner, os.path.basename(VIDEO)))
    csv_file = open(run_dir / "frames.csv", "x", newline="", encoding="utf-8-sig")
    csv_writer = csv.DictWriter(csv_file, fieldnames=judge.CSV_COLUMNS)
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

        # 4.1 ~ 4.11 축소 → YOLO → depth → 판정 (judge.py — 실시간 서버와 같은 코드)
        out = pipe.step(frame)
        frame, W, H, r, j = out["frame"], out["W"], out["H"], out["r"], out["j"]
        yolo_ms, depth_ms, depth_m = out["yolo_ms"], out["depth_ms"], out["depth_m"]
        cur_W = W
        frame_idx = out["idx"] + 1
        if SHOW_WINDOW and depth_ms is not None:   # depth를 새로 계산한 프레임만 컬러 화면 갱신
            depth_color = judge.depth_colormap(depth_m)
        box_results, path_m, near_pt = j["box_results"], j["path_m"], j["near_pt"]
        obs_m, obs_src, obs_warn, obs_target = j["obs_m"], j["obs_src"], j["obs_warn"], j["obs_target"]
        direction, motor, vib_count = j["direction"], j["motor"], j["vib_count"]

        # 4.9 frames.csv 한 줄 (일시정지 전에 한 번만 씀)
        if csv_writer:
            save_row(judge.frame_row(trial, out["idx"], frame_time_ms(out["idx"], pos_msec, src_fps), out["sx"],
                                     box_results, (obs_m, obs_src, obs_warn, obs_target), direction,
                                     (motor, vib_count), yolo_ms, depth_ms))

        # 4.5 화면 표시 (선택)
        if SHOW_WINDOW:
            # 왼쪽: 원본 + 박스, 오른쪽: depth 컬러맵, 양쪽: 경로 테두리 + 장애물 글자 + 방향 칸
            annotated = frame.copy()
            judge.draw_boxes(annotated, box_results)
            right = depth_color.copy() if depth_color is not None else np.zeros_like(annotated)
            for img in (annotated, right):
                judge.draw_path(img, H, W, obs_m, obs_src, obs_warn, near_pt, motor)
                if judge.ZONE_MODE != "off":
                    judge.draw_zones(img, W, H)

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
    judge.write_json(run_dir / "summary.json", judge.summary(
        rows, trial, "file", os.path.basename(VIDEO), list(src_size), src_fps,
        round(src_frames / src_fps, 3) if src_fps else None, time.time() - run_t0,
        args.expected_dir, args.gt_2m_sec, args.note))

# ===== 4.6 요약 =====
total_sec = (time.time() - start_time) if start_time else 0.0
n = len(frame_list)
print("\n" + "=" * 62)
print(" 측정 요약 (워밍업 제외)")
print("=" * 62)
print(f" 장치          : {DEVICE}")
print(f" 입력          : {os.path.basename(VIDEO) if is_file else '맥북 카메라'}")
print(f" 설정          : 입력 {judge.FRAME_WIDTH}px, YOLO {judge.IMG_SIZE}px, depth {DEPTH_EVERY_N}프레임마다, 창 {'켜짐' if SHOW_WINDOW else '꺼짐'}")
print(f" 측정 프레임   : {n}장 / {total_sec:.1f}초")
print(f" 평균 fps      : {n / total_sec:.1f}" if total_sec > 0 else " 평균 fps      : -")
print(f" YOLO (ms)     : {stats(yolo_list)}")
print(f" Depth (ms)    : {stats(depth_list)}  ({len(depth_list)}회)")
print(f" 프레임 전체(ms): {stats(frame_list)}")
print(f" 탐지 물체 수  : {stats(objs_list)}")
print(f" 경로 거리(m)  : {stats(path_list)}")
print(f" 장애물 거리(m): {stats(obs_list)}")
warn_ratio = (warn_count / n * 100) if n else 0.0
print(f" 경고 프레임   : {warn_count}장 ({warn_ratio:.1f}%, 기준 {judge.WARN_M}m 미만, 보정 x{judge.DEPTH_SCALE})")
print(f" 출처별 경고   : YOLO {warn_by_src['YOLO']}회 / DEPTH {warn_by_src['DEPTH']}회")
print(f" 모터별 경고   : left {warn_by_motor['left']} / right {warn_by_motor['right']} / neck {warn_by_motor['neck']}")
if run_dir is not None:
    print(f" run 폴더      : {run_dir}")
print("=" * 62)
