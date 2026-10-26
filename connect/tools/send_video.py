"""
send_video.py — 폰 대신 녹화 영상을 server.py로 보내는 시험용 (글래스·폰 없이 맥 하나로 확인)

  (터미널 1) RUNNER=내이름 uvicorn server:app --host 0.0.0.0 --port 8001
  (터미널 2) python tools/send_video.py 1009_left_01.mp4 --port 8001

영상 속도에 맞춰 프레임을 JPEG로 보내고, 서버 회신(경고·방향·모터)을 1초마다 찍음.
기록까지 하려면 터미널 1에 시행ID를 입력해 시작한 뒤 이걸 돌리고, 끝나면 터미널 1에서 Enter.
"""

import argparse
import json
import time

import cv2
from websockets.sync.client import connect

p = argparse.ArgumentParser()
p.add_argument("video")
p.add_argument("--host", default="127.0.0.1")
p.add_argument("--port", type=int, default=8001)
p.add_argument("--quality", type=int, default=80, help="JPEG 품질")
a = p.parse_args()

cap = cv2.VideoCapture(a.video)
if not cap.isOpened():
    raise SystemExit(f"영상을 열 수 없어요: {a.video}")
fps = cap.get(cv2.CAP_PROP_FPS) or 30
url = f"ws://{a.host}:{a.port}/stream"
print(f"{url} 로 전송 ({fps:.0f}fps)")

n_sent = n_recv = n_warn = 0
last = None
t0 = time.time()
last_print = 0
with connect(url, max_size=None) as ws:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        target = t0 + n_sent / fps
        wait = target - time.time()
        if wait > 0:
            time.sleep(wait)
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, a.quality])
        ws.send(jpg.tobytes())
        n_sent += 1
        try:
            while True:
                msg = json.loads(ws.recv(timeout=0))
                n_recv += 1
                last = msg
                n_warn += bool(msg.get("judge", {}).get("warn"))
        except TimeoutError:
            pass
        if last and time.time() - last_print >= 1:
            jd = last["judge"]
            print(f"[{time.time() - t0:5.1f}s] 보냄 {n_sent} 회신 {n_recv} | server {last['server_ms']}ms | "
                  f"장애물 {jd['obstacle_m']} m ({jd['source']}) | "
                  + (f"경고 {jd['direction']} → 모터 {jd['motor']}" if jd["warn"] else "경고 없음"))
            last_print = time.time()
print(f"끝. 보낸 프레임 {n_sent}, 회신 {n_recv}, 경고 회신 {n_warn}")
