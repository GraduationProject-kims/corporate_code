"""
compare_depth.py — 같은 영상에서 depth 모델 여러 개의 속도·경로 거리 비교 [10/30 YOLO26]

판정 코드(judge.py)의 depth_map·path_obstacle을 그대로 써서, 모델만 바꿔 가며 같은 프레임을 처리함.
  - 속도: 모델별 depth 한 번 걸린 시간(ms) 중앙값 (처음 몇 프레임은 준비 시간이라 뺌)
  - 거리: 프레임마다 경로 장애물 거리(m, judge.path_obstacle). 정지 촬영 영상이면 --gt_m 으로 실제 거리를 주면 오차를 냄

예)
  # 정면 2 m 정지 촬영 영상으로 세 모델 비교
  python tools/compare_depth.py --video 1017_d2m_01.mp4 --gt_m 2.0
  # 모델을 직접 고르기 (백엔드:모델:입력크기)
  python tools/compare_depth.py --video 1019_front_01.mp4 \
      --configs dav2 yolo26:yolo26n-depth.pt:768 yolo26:yolo26n-depth.pt:640 yolo26:yolo26s-depth.pt:768

결과: 화면 표 + <영상이름>_depth_compare.csv (프레임별 거리)
"""

import argparse
import csv
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2  # noqa: E402
import numpy as np  # noqa: E402

import judge  # noqa: E402

DEFAULT_CONFIGS = ["dav2", "yolo26:yolo26n-depth.pt:768", "yolo26:yolo26s-depth.pt:768"]

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS,
                help="백엔드[:모델[:입력크기]]. 예: dav2 / yolo26:yolo26n-depth.pt:640")
ap.add_argument("--every", type=int, default=5, help="몇 프레임마다 하나씩 쓸지 (기본 5)")
ap.add_argument("--max_frames", type=int, default=60, help="최대 몇 프레임 (기본 60)")
ap.add_argument("--warmup", type=int, default=3, help="속도에서 뺄 처음 프레임 수")
ap.add_argument("--gt_m", type=float, default=None, help="정지 촬영이면 실제 장애물 거리(m)")
ap.add_argument("--coreml", action="store_true", help="yolo26 설정을 CoreML로 (맥 전용)")
args = ap.parse_args()

device = judge.resolve_device(judge.DEVICE)
print(f"장치: {device}")

# 프레임 읽기 (judge.FramePipeline과 같은 폭으로 축소)
cap = cv2.VideoCapture(args.video)
frames, i = [], 0
while len(frames) < args.max_frames:
    ok, f = cap.read()
    if not ok:
        break
    if i % args.every == 0:
        h, w = f.shape[:2]
        if w > judge.FRAME_WIDTH:
            f = cv2.resize(f, (judge.FRAME_WIDTH, int(h * judge.FRAME_WIDTH / w)))
        frames.append(f)
    i += 1
cap.release()
if not frames:
    raise SystemExit(f"영상을 못 읽었어요: {args.video}")
H, W = frames[0].shape[:2]
print(f"프레임 {len(frames)}장 ({W}x{H})\n")


def load(cfg):
    parts = cfg.split(":")
    backend = parts[0]
    model = parts[1] if len(parts) > 1 else judge.DEPTH_MODELS[backend]
    imgsz = int(parts[2]) if len(parts) > 2 else 768
    judge.DEPTH_BACKEND, judge.DEPTH_MODEL, judge.DEPTH_IMG_SIZE = backend, model, imgsz
    judge.DEPTH_COREML = args.coreml and backend == "yolo26"
    return judge._load_depth(device)


results = {}
for cfg in args.configs:
    print(f"== {cfg} 불러오는 중")
    try:
        dp = load(cfg)
    except Exception as e:
        print(f"   실패: {e}")
        continue
    ms, dists = [], []
    for k, f in enumerate(frames):
        t0 = time.time()
        dm = judge.depth_map(f, W, H, dp)
        judge.sync_device(device)
        if k >= args.warmup:
            ms.append((time.time() - t0) * 1000)
        d, _ = judge.path_obstacle(dm)
        dists.append(d)
    results[cfg] = (ms, dists)
    print(f"   depth {statistics.median(ms):.1f} ms (중앙값)")

if not results:
    raise SystemExit("돌아간 모델이 없어요")

# 표
print("\n설정                              | depth ms | 경로거리 중앙값 | " +
      ("|오차| 중앙값 | 오차 평균(부호) " if args.gt_m else "프레임 간 흔들림(표준편차)"))
for cfg, (ms, dists) in results.items():
    v = np.array([d for d in dists if d is not None])
    row = f"{cfg:<34}| {statistics.median(ms):8.1f} | {np.median(v):12.2f} m | "
    if args.gt_m:
        err = v - args.gt_m
        row += f"{np.median(np.abs(err)):10.2f} m | {err.mean():+.2f} m"
    else:
        row += f"{v.std():.2f} m"
    print(row)
if args.gt_m:
    print("\n오차 평균(부호)가 일정하게 + 또는 - 로 치우쳐 있으면 judge.DEPTH_SCALE로 보정 가능 (실제 ÷ 측정)")

out = os.path.splitext(os.path.basename(args.video))[0] + "_depth_compare.csv"
with open(out, "w", newline="", encoding="utf-8") as fp:
    wr = csv.writer(fp)
    wr.writerow(["sample"] + list(results))
    for k in range(len(frames)):
        wr.writerow([k] + [("" if results[c][1][k] is None else round(results[c][1][k], 3)) for c in results])
print(f"\n프레임별 거리: {out}")
