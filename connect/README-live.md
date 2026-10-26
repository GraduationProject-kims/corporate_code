# 실시간 연결 코드 · 3-1조

**맥북 서버 링크를 사이트에 붙여넣는 간단한 영상 연결은 README-server-link.md를 먼저 읽으세요.** 아래는 별도 브리지로 영상과 판정을 결합하는 연결 방식입니다.

이미 작동하는 `메타글래스 → 휴대폰 → 노트북(YOLO + Depth) → 휴대폰` 흐름에서, **노트북이 휴대폰에 보내는 결과를 사이트에도 복사 전송**하는 연결입니다. 모델 추론을 한 번 더 실행하지 않고, MP4 저장이나 GitHub 업로드도 필요 없습니다.

## 1. 연결 방식

- `live_bridge.py`: 사이트와 노트북 판정 코드를 이어 주는 WebSocket 서버.
- `live_publisher.py`: 기존 Python 판정 코드에 붙이는 전송 클래스. 네트워크 전송은 별도 스레드에서 처리하며 최신 프레임 한 개만 대기시킵니다.
- 사이트의 **실시간** 탭: JPEG와 그 JPEG에 대응하는 판정 값을 한 메시지로 받습니다.
- 시각화용 연결입니다. 기존 휴대폰/BLE 경고 경로를 변경하지 않습니다.

## 2. 노트북에서 준비

압축을 풀고 이 폴더에서 Python 3.10 이상으로 실행하세요.

```sh
python -m pip install -r requirements-live.txt
```

서버와 판정 코드가 같은 연결 키를 사용하게 설정합니다. 임의로 긴 문자열을 정하고 사이트의 연결 키 칸에도 입력하세요. 코드/GitHub에는 키를 넣지 않습니다.

macOS/Linux 터미널 예시:

```sh
export LIVE_TOKEN='직접-정한-긴-연결키'
python live_bridge.py
```

Windows PowerShell 예시:

```powershell
$env:LIVE_TOKEN='직접-정한-긴-연결키'
python live_bridge.py
```

별도 터미널에서 기존 모델 서버를 실행할 때도 같은 LIVE_TOKEN을 설정하세요.

## 3. 기존 판정 코드에 붙이기

두 `.py` 파일을 기존 서버의 Python import 경로에 둡니다. 서버 시작할 때:

```python
from live_publisher import LivePublisher
publisher = LivePublisher()  # 같은 노트북의 ws://127.0.0.1:8765/live에 연결
```

**한 프레임의 추론과 경고 판정이 끝난 뒤**, 휴대폰에 보낼 JPEG와 같은 프레임의 기록을 함께 전달하세요.

```python
# jpeg_bytes: 기존 서버가 휴대폰에 보내는 결과 JPEG 바이트
# log_row: frames.csv에 쓰는 것과 같은 판정 딕셔너리
publisher.submit(jpeg_bytes, log_row)
```

기존 코드에 JPEG 인코딩이 없다면 OpenCV 결과 프레임을 이렇게 인코딩할 수 있습니다. 이 코드의 `annotated_frame`은 기존 코드의 실제 결과 프레임 변수로 바꿔야 합니다.

```python
ok, encoded = cv2.imencode('.jpg', annotated_frame,
                         [cv2.IMWRITE_JPEG_QUALITY, 75])
if ok:
    publisher.submit(encoded.tobytes(), log_row)
```

`log_row`의 필수 필드:

```python
log_row = {
    'frame_idx': frame_idx,       # 스트림 내 증가하는 프레임 번호
    't_ms': t_ms,                 # 스트림 시작 기준 ms, 뒤로 가지 않도록
    'processed': 1,
    'dist': dist,                 # 없으면 None 또는 빈 문자열
    'ttc': ttc,                   # 없으면 None 또는 빈 문자열
    'level': level,               # 0~3, 기존 판정 그대로
    'level_name': level_name,     # 없음/주의/경고/위험
    'direction': direction,       # 화면의 장애물 위치
    'motor': motor,               # left/right/neck/빈 문자열, 회피할 쪽
    'vib_count': vib_count,
    'vibrate_ms': vibrate_ms,
    'target_label': target_label,
    'yolo_ms': yolo_ms,
    'depth_ms': depth_ms,
}
```

위 예시는 판정 딕셔너리가 이미 있는 서버용입니다. 확인한 현재 팀 프로토타입은 노트북 `server/server.py`가 탐지·거리만 돌려주고, 휴대폰 `ObstacleAlarm.kt`에서 경고를 계산합니다. 따라서 현재 서버에 단순히 log_row가 있다고 가정하면 안 됩니다. 아래 7절의 연결 방법을 사용하세요. 레벨·거리·모터 값은 여기서 새로 계산하지 않습니다. processed=0은 전송/표시에서 건너뜁니다. 브리지 중단 시 추론과 휴대폰 경로는 계속 작동하고 `publisher.connected`, `publisher.last_error`, `publisher.dropped`로 전송 상태를 확인할 수 있습니다. 서버 종료 시 `publisher.close()`를 호출하세요.

## 4. 공개 사이트에서 접속할 주소

사이트가 HTTPS이므로 **사이트 입력칸에는 `wss://.../live`가 필요**합니다. `ws://127.0.0.1:8765/live`는 노트북 내부 전송용 주소이며 다른 컴퓨터나 공개 사이트가 접속할 주소가 아닙니다.

이미 쓰는 HTTPS 서버/역프록시에서 `/live`의 WebSocket 연결을 브리지의 127.0.0.1:8765로 전달하세요. 프록시는 WebSocket Upgrade를 지원해야 합니다. 브리지 자체에 신뢰되는 TLS 인증서를 적용할 수도 있습니다:

```sh
python live_bridge.py --host 0.0.0.0 --certfile 인증서.pem --keyfile 개인키.pem
```

현재 팀 서버 주소/HTTPS 구성은 제공되지 않았습니다. 실제 접속 주소는 팀 서버 담당자가 확인해야 합니다. 이 ZIP을 실행하는 것만으로 공개 wss 주소가 자동 생성되지는 않습니다.

## 5. 사이트에서 사용

1. [보행 경고 시연](https://walk-warning-lab.awareangel1.chatgpt.site/) → **실시간** 탭.
2. 사이트에서 접속 가능한 wss 주소와 LIVE_TOKEN 입력 → 연결.
3. `서버 연결됨 · 영상 대기`가 표시되면 인증 완료. 이후 모델 프레임이 들어오면 `실시간 수신 중`으로 바뀝니다.
4. 같은 JPEG와 판정이 함께 표시되고 최근 30초 거리/진동 명령 그래프가 갱신됩니다.
5. 3초 이상 새 프레임이 없거나 연결이 끊기면 현재 모터 명령을 표시하지 않고 신호 중단을 표시합니다.

연결 키는 브라우저 메모리에서만 사용하고 연결 해제 시 비웁니다. 서버는 기본적으로 해당 사이트 Origin과 Python publisher만 받아들입니다. 다른 개발 Origin이 필요하면 LIVE_ALLOWED_ORIGIN을 설정하세요.

## 6. 확인 범위

브리지의 인증, JPEG/판정 동시 전송, 증가하는 프레임 순서, processed=0 제외, 송신 종료, 서버/클라이언트의 최신 프레임 처리를 테스트합니다. 실제 메타글래스/휴대폰 장비와 기존 판정 서버에 붙이는 단계는 해당 소스와 주소가 있어야 검증할 수 있습니다. 브라우저에서의 FPS는 수신된 프레임 수이며 모델 추론 FPS나 종단 지연과 동일한 측정이 아닙니다.

## 7. 확인한 현재 팀 서버에 붙이는 방법

현재 `server/pipeline.py`는 추론(탐지·거리) 공용 코드이고, 실시간 입출력은 `server/server.py`의 `inference_worker()`와 `/stream`입니다. 휴대폰의 `ObstacleAlarm.kt.process()`가 TTC·레벨과 진동을 계산하며, 현재 `FrameSender.kt`는 탐지 응답만 받습니다. 이 확인은 저장소 기본 브랜치 기준입니다. 아까 시연한 노트북의 로컬 수정본은 다를 수 있습니다.

### A. 모델 영상부터 사이트로 받기

`live_bridge.py`, `live_publisher.py`, `live_mirror.py`를 `server/server.py`와 같은 폴더에 둡니다. 서버 파일에 추가:

```python
from live_mirror import LiveMirror
live_mirror = LiveMirror(wait_for_phone=False)
```

`inference_worker()`에서 기존 `latest_view["annotated"] = combined` 다음에 추가:

```python
try:
    live_mirror.remember_bgr(combined, json.loads(result_json), frame_idx)
except Exception as error:
    print("사이트 영상 전송 실패:", type(error).__name__)
```

이 경로는 기존 휴대폰으로 보내는 JSON을 바꾸지 않습니다. 모델 박스·거리가 그려진 JPEG를 사이트에 전송합니다. **휴대폰 판정을 받지 않았으므로 사이트 위험도에는 '판정 미수신', 모터에는 미수신 상태를 표시합니다. 경고 없음(레벨 0)으로 꾸며 표시하지 않습니다.** 그래프의 경고 대상 dist도 휴대폰의 선택 대상 기록이 도착하기 전에는 비어 있습니다. 박스별 거리는 결과 영상에 보입니다.

### B. 실제 휴대폰 판정까지 같은 프레임으로 받기

1. 위 LiveMirror를 `wait_for_phone=True`로 바꿉니다. 이때 JPEG는 잠깐 보관하고 휴대폰의 같은 프레임 판정이 도착할 때 함께 전송합니다.
2. 서버 응답 JSON에 `live_frame_idx: frame_idx`를 추가해 휴대폰이 판정한 탐지 프레임을 식별합니다.
3. 휴대폰은 ObstacleAlarm에서 실제 계산한 레벨·대상 거리·TTC·진동 명령을 로그 딕셔너리로 남기고, 같은 WebSocket으로 `{type:"live_alarm", frame_idx:live_frame_idx, record:판정기록}`을 회신해야 합니다. 현재 함수는 결과를 반환하지 않아 이 기록 추가가 필요합니다. 사이트에서 임의로 TTC/레벨을 재계산하면 실제 폰 명령과 일치하지 않을 수 있습니다.
4. 서버의 `/stream`은 현재 바이너리 JPEG만 받으므로 `ws.receive()`로 바이너리와 텍스트를 구분합니다. JPEG 경로는 기존대로 유지하고, live_alarm 텍스트 경로에서는 `live_mirror.report_alarm(payload["frame_idx"], payload["record"])`를 호출합니다.

3초보다 오래된 판정이나 대응하는 JPEG가 없는 프레임은 붙이지 않습니다. 이 방식은 실제 장치 ACK를 증명하지 않으며, 휴대폰이 낸 명령을 보여줍니다. 현재 연결 코드에는 프레임 결합 함수가 구현돼 있지만 기존 Android 앱/실행 중 서버에 원격 적용하지는 않았습니다. 실제 실행 서버의 주소와 적용할 앱 소스를 확인하고 그 환경에서 빌드·시연을 검증해야 합니다.
