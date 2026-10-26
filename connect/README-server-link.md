# 맥북 서버 링크로 시연 사이트 연결

## 저장소에서 확인한 구조

`CameraViewModel.kt`의 SERVER_URL은 휴대폰이 맥북으로 JPEG를 보내는 WebSocket 주소입니다. 폰이 새 공유 링크를 생성하는 구조가 아닙니다. `server/server.py`의 `/stream`은 JPEG를 받아 탐지 JSON을 폰에 돌려주며, 웹 페이지용 영상 응답은 없습니다. 실제 경고 판정은 폰의 ObstacleAlarm에서 수행합니다.

아래는 맥북에서 이미 성공한 모델 서버를 유지하며, 사이트가 결과 영상을 읽는 HTTP 경로만 추가하는 방법입니다. 기존 `/stream`과 GPU 설정은 그대로 둡니다.

## 1. 파일과 두 줄 추가

ZIP의 `visualization_http.py`를 실제 실행 중인 `server.py` 옆에 복사하세요. 이 방식에는 별도 live_bridge 실행이나 websockets 버전 변경이 필요 없습니다. 기존 FastAPI·OpenCV 환경을 사용합니다.

server.py에서 `latest_view = {"annotated": None}`를 찾고 그 아래, 함수 밖에 추가:

```python
from visualization_http import install_visualization
install_visualization(app, latest_view)
```

확인한 원본에서는 이 위치 전에 `app = FastAPI()`가 선언돼 있습니다. 실행본 구조가 다르면 app과 latest_view 두 변수가 선언된 뒤에 추가하세요. `latest_view["annotated"]`에 매 추론마다 새 결과 배열을 저장하는 기존 동작을 사용합니다. LiveMirror 코드를 이미 붙였다면 별도 브리지는 이번 연결에 필요 없습니다.

## 2. 맥북 모델 서버 재실행

기존에 사용한 Python 환경과 server.py 폴더에서:

```bash
export LIVE_TOKEN='REPLACE-WITH-YOUR-OWN-LONG-KEY'
uvicorn server:app --host 0.0.0.0 --port 8001
```

8001은 실제 서버 포트로 맞춥니다. 연결 키는 공백 없이 영문·숫자·기호로 직접 정하고, 사이트 연결 키 칸에도 똑같이 넣습니다. 키를 URL이나 GitHub 코드에 넣지 마세요. 기존 실행 프로세스를 먼저 종료하고 재실행하며, 맥북 MPS 등 기존 모델 설정은 바꾸지 않습니다.

## 3. HTTPS 링크 생성

현재 공개 사이트는 HTTPS 서버 링크가 필요합니다. 이미 HTTPS로 맥북의 모델 서버에 접속하고 있다면 그 기본 주소를 그대로 쓰세요. 새 링크가 필요하면 시연용 Cloudflare Quick Tunnel을 사용할 수 있습니다. 영상은 이 터널 서비스를 거쳐 브라우저로 전달됩니다.

Homebrew 설치 환경에서 새 터미널:

```bash
brew install cloudflared
cloudflared tunnel --url http://127.0.0.1:8001
```

이번에는 **모델 서버 포트 8001**입니다. 기존 브리지 방식의 8765가 아닙니다. 터미널에 나온 실제 `https://…trycloudflare.com` 주소를 사용하세요. 재실행 시 주소가 바뀔 수 있으며 터미널은 켜두어야 합니다.

## 4. 사이트에 입력

https://walk-warning-lab.awareangel1.chatgpt.site → 실시간 → 연결 방식 **맥북 서버 링크**.

- 맥북 서버 링크: 방금 얻은 HTTPS 기본 주소.
- 연결 키: 모델 서버 실행 때 설정한 LIVE_TOKEN.
- 연결을 누르고 휴대폰 앱에서 기존처럼 서버 모드로 영상 송신.

사이트는 기본 주소에 `/visualization/frame`을 붙여 읽습니다. HTTPS `/stream` 또는 WSS `/stream` 주소를 붙여도 같은 서버의 `/visualization/frame`으로 변환해 입력칸에 보여줍니다. 이 변환이 서버 기능을 설치해 주는 것은 아닙니다. 1번의 코드 추가가 필요합니다. 일반 공유 페이지·임의 영상 링크는 지원하지 않습니다. `/live`는 기존 브리지 방식에서 사용합니다.

## 표시 범위와 오류

박스와 거리 문구가 그려진 서버 영상을 받습니다. 경고 대상 dist·TTC·레벨·모터를 추측하지 않으므로 폰 기록이 없으면 판정 미수신과 빈 그래프가 정상입니다. 실제 판정까지 필요하면 README-live.md 7B의 휴대폰 기록 결합을 적용하고 브리지 방식을 사용하세요. 실제 모터 작동 ACK와도 구분합니다.

서버가 새 결과 배열을 제공하지 않으면 3초 후 영상과 현재 명령을 숨깁니다. 프레임 시각은 이 HTTP 연결이 관측한 서버 결과 시각이며 촬영 시각이나 종단 지연 측정값이 아닙니다. 최대 약 5회/초로 화면을 요청하므로 모델 FPS와도 다릅니다.

- 404: 연결 파일 미설치, 서버 재시작 누락 또는 터널이 다른 포트를 가리킴.
- 연결 키 오류: LIVE_TOKEN 불일치.
- 서버 연결됨·영상 대기: 휴대폰 송신 상태와 모델 실행 콘솔 확인.
- 링크 접근 실패: HTTPS 터널, 서버 실행, CORS·네트워크 확인.

연결 키는 브라우저 메모리에서만 사용합니다. 서버 CORS는 위 시연 사이트만 허용합니다. URL을 입력하는 행위 자체로 휴대폰 앱이나 모델 서버의 설정을 변경하지 않습니다.
