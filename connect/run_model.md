# 장애물 경고 코드 명세 (짧은 버전)

> 코드: `judge.py`(처리·판정·설정값) + `run_model.py`(녹화 영상) / `server.py`·`live_record.py`(실시간) · 결과 형식: `RUN_FORMAT.md`(이 문서 5장과 같음)
>
> **11/8 기준.** 기본 판정은 **YOLO가 잡는 클래스 전부 + 3D 통로**다. 3D 통로 규칙도 이 문서에 다 들어 있다 (`3-1조_3D통로_변경사항.md`는 11/2 변경 기록으로만 남긴다. 거기 적힌 "기본 타겟 suitcase"는 이제 맞지 않는다).
>
> 결과를 맞추는 공식 방법은 **모두 같은 커밋의 `run_model.py`를 같은 인자·같은 환경변수로 돌리는 것**이다. 이 문서만 보고 코드를 다시 짜도 같은 숫자가 나오도록, 계산 규칙(4장)은 반올림까지 빠짐없이 적었다.

### 바뀐 순서

| 날짜 | 바뀐 것 |
|---|---|
| 10/26 | 녹화·실시간이 같은 처리 함수(`FramePipeline.step()`)를 씀. RUN_FORMAT schema_version 2 |
| 10/30 | 모델 교체: 탐지 yolo11n → **yolo26s**, 거리 Depth Anything V2 → **yolo26n-depth** |
| 11/2 | 클래스 모드에서 경로 판정을 화면 사각형 → **3D 통로**로. 방향도 미터 기준. 타겟 전용 결과 폴더 추가 |
| 11/8 | 기본 타겟 suitcase → **all**(클래스 전부). 서버는 `LIVE_TOKEN` 없이도 켜짐(사이트 전송만 꺼짐). `environment.yml` websockets 16.0 |
| 11/8 | 카메라 좌우 각도 보정 `CAMERA_YAW_DEG` 추가, 기본 **9°**(메타 안경 카메라가 걷는 방향보다 오른쪽을 봄). 화면에 1·2 m 단면 추가 |
| 11/13 | 녹화 영상을 `connect/videos/`에 모아 두고 `--video 1109_front_01`처럼 이름만으로 찾기 |
| 11/9 | `run_model.py --save_video`: 화면 그림을 run 폴더에 `annotated.mp4`로 저장 (보기용) |
| 11/8 | `CAMERA_HEIGHT_M` 기본 1.55 → **1.65** (통로 천장 1.75 → 1.85 m). 화면 통로를 0.3 m부터 그리고 눈높이선·머리선 표시 |

---

## 1. 무엇을 하나

영상 프레임마다 앞 경로에서 가장 가까운 장애물까지의 거리를 잰다. 3 m보다 가까우면 **피할 방향**으로 진동 명령(`left` / `right` / `neck`)을 기록한다.

```
프레임 → 폭 640px로 축소 → YOLO 박스 + depth 지도(m)
      → 박스마다 거리(depth) → 그 거리에서 박스를 미터 좌표로 바꿔 3D 통로와 겹치는지
      → 통로 안 박스 중 가장 가까운 거리 < 3 m 이면 경고
      → 물체 중심이 통로 중심선에서 왼쪽/정면/오른쪽 → 반대쪽 모터
      → frames.csv에 한 줄 (+ target_all/frames.csv에 박스마다 한 줄)
```

### 판정 모드 세 가지

환경변수 `TARGET_LABELS`로 고른다. 실행할 때 터미널 첫 줄에 `판정 모드: ...`가 나온다.

| 모드 | 켜는 법 | 장애물 후보 | 경로 안 판정 | 방향 판정 | 타겟 폴더 |
|---|---|---|---|---|---|
| **클래스 전부 (기본)** | 아무것도 안 붙임 (`TARGET_LABELS=all`과 같음) | 모든 YOLO 박스 | 3D 통로 (4-3 A) | 미터 기준 (4-6 A) | `target_all/` |
| 지정 클래스 | `TARGET_LABELS=suitcase` / `TARGET_LABELS=bottle,chair` | 그 클래스 박스만 | 3D 통로 | 미터 기준 | `target_suitcase/` 등 |
| 깊이 기반 | `TARGET_LABELS=` (빈 값) | 경로 depth + 모든 YOLO 박스 | 화면 사각형 (4-3 B) | 화면 칸 (4-6 B) | 없음 |

- 클래스 모드(위 두 줄)에 `PATH_MODE=2d`를 붙이면 3D 통로 대신 예전 화면 사각형으로 경로 안을 판정한다 (비교용).
- **클래스 모드에서는 경로 depth(DEPTH)가 후보가 아니다.** 그래서 YOLO가 박스로 못 잡는 물체(폼보드 간판, A4 등)는 경고가 안 뜬다. 그런 물체까지 잡는 건 깊이 기반 모드뿐이다.
- 클래스 전부 모드에서 클래스(사람·의자 등)는 화면 표시와 기록에만 쓰고 판정에는 안 쓴다. 지정 클래스 모드에서는 "후보에 넣을지"를 거르는 데만 쓰고, 클래스마다 규칙을 다르게 두지 않는다.
- 통로 밖 박스는 어느 모드든 경고 후보가 아니다. 더 가까워도 화면에 얇은 박스로만 그려진다.

### 녹화와 실시간이 같은 코드를 쓴다
위 흐름(축소 → YOLO → depth → 판정 → 방향 유지)은 전부 `judge.py`의 `FramePipeline.step()` 안에 있다. 녹화 영상(`run_model.py`)과 실시간 서버(`server.py`)는 모두 이 함수 하나를 부르고, 각자 하는 일은 입력과 출력뿐이다.

| 파일 | 하는 일 |
|---|---|
| `judge.py` | 설정값, 처리 순서, 판정, 기록 형식 (**오차를 고치는 곳은 여기 하나**) |
| `run_model.py` | 영상 파일 읽기, 화면 창, frames.csv 저장 |
| `server.py` | 폰 사진 받기·풀기, 폰 회신, 시연 사이트 전송(`LIVE_TOKEN`이 있을 때만) |
| `live_record.py` | 실시간 시행 기록 (판정 안 함) |
| `live_mirror.py`·`live_publisher.py`·`live_bridge.py` | 시연 사이트 전송 (판정 안 함) |

녹화 영상으로 오차를 보고 `judge.py`를 고친 뒤: `run_model.py`로 다시 확인 → 커밋 → (서버 컴퓨터에서 받기) → **서버 재시작**. 켜져 있는 서버는 시작할 때 읽은 `judge.py`와 환경변수를 계속 쓴다.

실시간 시행 기록을 시작하면(시행ID + Enter) 판정 상태가 처음으로 돌아간다(`FramePipeline.reset()`): `frame_idx` 0, 방향 기억 없음, depth 새로 계산. 그래서 같은 화면이 들어오면 실시간 frames.csv와 녹화 frames.csv의 판정 칸이 같다. 시작 전 상태로 판정한 프레임과 Enter 전에 받은 프레임은 기록하지 않는다. 단, 실시간은 서버가 못 따라가면 밀린 프레임을 버리고 최신 프레임만 처리하므로 처리 프레임 수가 녹화보다 적을 수 있다.

---

## 2. 실행

```bash
# 녹화 영상 (기본: 클래스 전부 + 3D 통로)
python run_model.py --video 1108_front_01.mp4 --runner 연수

# 실시간 (폰 ↔ 맥북만)
RUNNER=연수 uvicorn server:app --host 0.0.0.0 --port 8001

# 실시간 + 시연 사이트
LIVE_TOKEN=연결키 RUNNER=연수 uvicorn server:app --host 0.0.0.0 --port 8001
```

### 인자 (`run_model.py`)

| 인자 | 뜻 |
|---|---|
| `--video` | 영상 경로 또는 파일 이름. 이름만 주면 `connect/videos/`(환경변수 `VIDEO_DIR`로 변경 가능)에서 찾고, 확장자(`.mp4`/`.mov`)는 생략해도 됨. `0`이면 맥북 카메라(결과 저장 안 함). 없으면 코드의 `SOURCE` |
| `--runner` | 돌린 사람 이름. 영상이면 필수 |
| `--trial` | 시행ID. 없으면 영상 파일명에서 확장자를 뺀 것 |
| `--expected_dir`, `--gt_2m_sec`, `--note` | 기록만 함. 계산에는 안 씀 |
| `--save_video` | 붙이면 화면 그림(왼쪽 원본+박스+통로, 오른쪽 depth)을 run 폴더에 `annotated.mp4`로 저장. 판정과 무관. 영상 파일 입력일 때만 |
| `--depth_every_n` | N프레임마다 depth 1번 (기본: `judge.py`의 `DEPTH_EVERY_N` = 1). **결과가 달라지니 팀 전체가 같은 값을 쓴다**. 실시간 서버는 환경변수 `DEPTH_EVERY_N` |

- 시행ID 형식은 `<MMDD>_<front|left|right|none>_<2자리>`이고, 영상 파일명과 같게 한다. 예: `1108_left_01.mp4`. 형식이 다르면 경고만 하고 그대로 진행한다.
- 키 조작 (영상 창을 클릭해 둔 상태): `space` 일시정지, `q` 종료, 마우스 클릭 → 그 점의 거리
- 실시간 서버: 서버 터미널에 시행ID + Enter로 기록 시작, 그냥 Enter로 끝

### 환경변수

코드를 고치지 않고 앞에 붙여서 바꾸는 값이다. **판정이 바뀌는 값은 팀 전체가 같은 값을 쓰고**, 모두 `run_meta.json`에 남는다.

| 환경변수 | 기본 | 판정 영향 | 뜻 |
|---|---|---|---|
| `TARGET_LABELS` | `all` | 있음 | 판정 모드 (1장 표) |
| `PATH_MODE` | `3d` | 있음 | 클래스 모드의 경로 안 판정. `2d`면 화면 사각형 |
| `HFOV_DEG` | 70 | 있음 | 카메라 가로 화각. 3D 통로 계산에 직접 쓰임 (**실측 필요**, 8-1) |
| `CAMERA_HEIGHT_M` | 1.65 | 있음 | 바닥 ~ 카메라 높이 (m). 착용자마다 잼. 다른 사람이면 `CAMERA_HEIGHT_M=1.58`처럼 앞에 붙임 |
| `CAMERA_PITCH_DEG` | 0 | 있음 | 카메라가 수평보다 아래로 숙인 각도 (+ = 아래) |
| `CAMERA_YAW_DEG` | 9 | 있음 | 카메라가 걷는 방향보다 오른쪽으로 돌아간 각도 (+ = 오른쪽을 봄). 메타 안경 잠정값. 폰 등 다른 카메라는 `0` |
| `CORRIDOR_BOTTOM_M` | 0 | 있음 | 통로 바닥 높이 (m) |
| `CORRIDOR_TOP_M` | `CAMERA_HEIGHT_M` + 0.2 | 있음 | 통로 천장 높이 (m) |
| `YOLO_MODEL` | `yolo26s.pt` | 있음 | 탐지 모델 (예전 것: `yolo11n.pt`) |
| `DEPTH_BACKEND` | `yolo26` | 있음 | `dav2`면 예전 Depth Anything V2 |
| `DEPTH_MODEL` | `yolo26n-depth.pt` (dav2면 DA V2 Indoor Small) | 있음 | depth 모델 |
| `DEPTH_IMG_SIZE` | 768 | 있음 | YOLO26-depth 입력 크기 (32의 배수). dav2는 518 고정 |
| `DEPTH_COREML` | 꺼짐 | 소수점 | `1`이면 맥 Neural Engine(CoreML)으로 depth. 변환 실패하면 PyTorch로 |
| `DEPTH_EVERY_N` | 1 | 있음 | 실시간 서버 전용 (녹화는 `--depth_every_n`) |
| `RUNNER` | `live` | 없음 | 실시간 기록의 돌린 사람 이름 |
| `LIVE_TOKEN` | 없음 | 없음 | 있으면 시연 사이트 전송 켜짐, 없으면 꺼짐 |
| `SITE_JUDGE` | 꺼짐 | 없음 | `1`이면 사이트에 판정(경고·방향·모터)도 보냄. `LIVE_TOKEN`이 있을 때만 의미 있음 |

비교용 조합:

| 돌리고 싶은 것 | 앞에 붙일 것 |
|---|---|
| 11/2 기본값 (suitcase만, 3D 통로, 좌우 보정 없음) | `TARGET_LABELS=suitcase CAMERA_YAW_DEG=0` |
| 좌우 보정 전후 비교 | `CAMERA_YAW_DEG=0` |
| 클래스 모드 + 예전 화면 사각형 | `PATH_MODE=2d` |
| 깊이 기반 판정 (클래스 안 씀) | `TARGET_LABELS=` |
| 7주 베이스라인 (예전 모델 + 깊이 기반) | `DEPTH_BACKEND=dav2 YOLO_MODEL=yolo11n.pt TARGET_LABELS=` |

---

## 3. 설정값 (`judge.py` 맨 위)

`run_model.py`·`server.py`에서 이 값을 바꿔도 판정은 바뀌지 않는다. 반드시 `judge.py`에서 바꾼다 (또는 2장의 환경변수).

```python
# 모델
FRAME_WIDTH = 640                     # 프레임 축소 폭
MODEL_PATH = "yolo26s.pt"             # 환경변수 YOLO_MODEL
CONF_THRES = 0.4                      # YOLO 확률 하한
IMG_SIZE = 416                        # YOLO imgsz
USE_DEPTH = True
DEPTH_BACKEND = "yolo26"              # 환경변수 DEPTH_BACKEND ("yolo26" / "dav2")
DEPTH_MODEL = "yolo26n-depth.pt"      # 환경변수 DEPTH_MODEL
DEPTH_IMG_SIZE = 768                  # 환경변수 DEPTH_IMG_SIZE
DEPTH_COREML = False                  # 환경변수 DEPTH_COREML=1
DEPTH_SCALE = 1.0                     # depth 보정 배율 (모델마다 다시 잼)
DEVICE = "auto"                       # 연산 장치: cuda → mps → cpu 순서로 자동
DEPTH_EVERY_N = 1                     # N프레임마다 depth 1번

# 경로
PATH_WIDTH_M = 0.8                    # 경로 폭 (3D 통로 폭으로 직접 쓰임)
HFOV_DEG = 70.0                       # 카메라 가로 화각 (환경변수 HFOV_DEG, 3D 통로에 직접 쓰임)
WARN_M = 3                            # 이보다 가까우면 경고
PATH_Y = (0.25, 0.75)                 # (기록용)
PATH_ROI = (0.40, 0.60, 0.25, 0.75)   # 화면 사각형 (x 시작, x 끝, y 시작, y 끝) 화면 비율. 깊이 기반·PATH_MODE=2d에서만
NEAR_PERCENTILE = 5                   # 경로 depth 하위 % (깊이 기반만)
NEAR_BAND_M = 0.3                     # 경로 거리 + 이 값 안의 픽셀들 중심 = 장애물 위치 (깊이 기반만)

# 판정 모드·3D 통로
TARGET_LABELS = "all"                 # 환경변수 TARGET_LABELS (1장)
PATH_MODE = "3d"                      # 환경변수 PATH_MODE
CAMERA_HEIGHT_M = 1.65                # 환경변수 (11/8 1.55 → 1.65)
CAMERA_PITCH_DEG = 0.0                # 환경변수
CAMERA_YAW_DEG = 9.0                  # 환경변수. 메타 안경 잠정값 (1009 영상 3개에서 8~10°, HFOV 70 기준)
CORRIDOR_BOTTOM_M = 0.0               # 환경변수
CORRIDOR_TOP_M = CAMERA_HEIGHT_M + 0.2   # 환경변수 (기본 1.85)
CORRIDOR_MIN_OVERLAP_M = 0.05         # 통로와 가로·세로로 이만큼 이상 겹쳐야 안
CORRIDOR_FRONT_M = 0.15               # 물체 중심이 중심선에서 이 안이면 정면
CORRIDOR_MARKS_M = (1.0, 2.0)         # 화면에 얇게 그리는 거리 단면 (보기용)
CORRIDOR_NEAR_M = 0.3                 # 화면에 통로를 몸 앞 이 거리부터 그림 (보기용)
HEAD_ABOVE_EYE_M = 0.10               # 눈에서 머리 꼭대기까지, 머리선 표시용 (보기용)

# YOLO 박스
BOX_SHRINK = 0.2                      # 박스를 각 변에서 줄이는 비율
BOX_PERCENTILE = 30                   # 박스 안 depth 하위 %
BOX_OVERLAP_MIN = 0.3                 # 박스 폭 중 화면 사각형 안 비율 하한 (화면 사각형 판정만)

# 방향·진동
ZONE_MODE = "path"                    # 화면 칸 방향 판정: 경로 사각형 3등분 (3D 통로 모드에서는 안 씀)
ZONE_DRAW = "screen"                  # 화면에 그리는 칸: 화면 전체 3등분 (보기용, 3D 통로 모드에서는 안 그림)
ZONE_PATH_SPLIT = (1 / 3, 2 / 3)
HOLD_DIRECTION = True                 # 경고 동안 첫 방향 유지
MOTOR_MAP = {"왼쪽": "right", "오른쪽": "left", "정면": "neck"}
VIB_COUNT = {1: 1, 2: 2}
MOTOR_TEXT_SCALE = 1.6                # 화면 모터 글자 크기

# 기록
RUNS_DIR = "runs"
SCHEMA_VERSION = 2
TARGET_SCHEMA_VERSION = 1             # 타겟 폴더 형식 버전
```

`CHOICES`는 판정 모드에 따라 값이 달라지고, `run_meta.json`의 `alarm_constants.choices`에 그대로 기록된다. 번호의 뜻은 부록에 있다. 이 값이 다르면 `score.py`가 "설정이 다른 run"으로 경고한다.

| 번호 | 3D 통로 (클래스 모드 기본) | 화면 사각형 (깊이 기반, `PATH_MODE=2d`) |
|---|---|---|
| 2-1 | A | A |
| 2-2 | B | B |
| 2-3 | 경고하지 않음 | 경고하지 않음 |
| 2-5 | `corridor (미터 기준)` | `path` |
| 2-7 | `3D 통로 겹침` | B |
| 2-9 | B (PATH_ROI 다시 계산) | B (PATH_ROI 다시 계산) |
| 2-10 | A (정면 → neck) | A (정면 → neck) |
| 2-11 | B | B |
| 2-12 | B | B |

---

## 4. 계산 규칙 (바꾸면 결과가 달라지는 부분)

`W`, `H`는 축소한 프레임의 폭·높이다. 계산 좌표는 모두 축소 프레임 픽셀 기준이다. 백분위는 `np.percentile` 기본값(선형 보간)을 쓴다. 4-1 ~ 4-7은 모두 `judge.FramePipeline.step()` 안에서 이 순서로 한다 (녹화·실시간 같음).

### 4-1. 프레임 준비
- 원본 폭 `w`가 640보다 크면 `cv2.resize(frame, (640, int(h * 640 / w)))`(기본 보간). 아니면 그대로 쓴다.
- `sx = w / W`: 저장할 때 원본 픽셀로 되돌리는 배율
- YOLO: `yolo.predict(frame, imgsz=416, conf=0.4, verbose=False, device=DEVICE)`. 박스 좌표는 `int()`로 소수점을 버린다. 모델 파일은 `judge.py`가 있는 폴더 기준이고 `fuse()`는 하지 않는다.
- Depth (`DEPTH_BACKEND=yolo26`, 기본): 축소 프레임(BGR)을 그대로 `depth_model.predict(frame, imgsz=768, verbose=False, device=DEVICE)`에 넣고 `r.depth.data`를 float32 배열로 꺼낸다. CoreML이면 `device`를 넘기지 않는다.
- Depth (`DEPTH_BACKEND=dav2`): 축소 프레임을 RGB로 바꿔 transformers pipeline에 넣고 `predicted_depth`를 꺼낸다.
- 두 경우 모두: 크기가 `(H, W)`와 다르면 `cv2.resize(d, (W, H))`(기본 보간)한 뒤 `DEPTH_SCALE`을 곱한다. 단위는 m.
  - `frame_idx % depth_every_n == 0`인 프레임에서만 새로 계산하고, 나머지 프레임은 직전 depth를 쓴다.
  - 첫 depth가 나오기 전에는 거리가 없다. 박스 거리는 `None`이고 경고도 없다.
  - 실시간 서버는 depth 모델을 못 불러오거나 한 프레임 depth가 실패해도 멈추지 않고 depth 없이(또는 직전 depth로) 계속한다. 녹화는 오류로 멈춘다.
- `DEVICE`는 `"auto"`: cuda가 있으면 cuda, 없으면 mps, 그것도 없으면 cpu다. 장치가 다르면 소수점 끝자리가 다를 수 있다.

### 4-2. YOLO 박스 거리
- 박스를 각 변에서 폭·높이의 20%씩 줄인 영역의 depth 하위 30% 백분위
  - `dx = (x2-x1)·0.2`, `dy = (y2-y1)·0.2`
  - 줄인 영역은 `x: round(x1+dx) ~ round(x2-dx)`, `y: round(y1+dy) ~ round(y2-dy)`. 끝은 포함하지 않고(파이썬 슬라이스), 화면 밖으로 나간 부분은 잘라낸다.
  - 줄인 영역의 폭이나 높이가 1픽셀보다 작으면 원래 박스 영역을 쓴다.
- 모든 박스(통로 밖, 타겟이 아닌 클래스 포함)의 거리를 계산하고 기록한다.

### 4-3. 박스가 "경로 안"인지

**A. 3D 통로** (클래스 모드 + `PATH_MODE=3d`, 기본)

사용자 앞 공간을 미터로 정한다: 가로 폭 `PATH_WIDTH_M`(0.8, 중심선 좌우 0.4), 높이 `CORRIDOR_BOTTOM_M` ~ `CORRIDOR_TOP_M`(0 ~ 1.85). 물체는 거리 `dist`(4-2)에 서 있는 납작한 판으로 본다.

- 거리가 `None`이거나 0 이하면 경로 밖이다.
- 초점거리 `f = (W/2) / tan(HFOV_DEG/2)` (라디안 변환 후). 화면 중심 `cx = W/2`, `cy = H/2`. 주점은 화면 중앙, 픽셀은 정사각형, 왜곡 보정 없음으로 가정한다.
- 가로·앞 거리의 기준은 **걷는 방향**이다. 카메라 방향과는 `CAMERA_YAW_DEG`만큼 다르다 (0이면 카메라 방향과 같음).
- 숙임각 `p = CAMERA_PITCH_DEG`, 좌우 각도 `a = CAMERA_YAW_DEG` (둘 다 라디안). 숙임을 먼저 되돌리고 좌우를 되돌린다.
- 높이와 수평 앞 거리: 화면 y 픽셀 `v`마다
  - `y_cam = (v - cy)·dist/f`
  - 높이 = `CAMERA_HEIGHT_M - (y_cam·cos p + dist·sin p)`
  - 수평 앞 거리 `z(v)` = `dist·cos p - y_cam·sin p`
  - `top` = `y1`의 높이, `bottom` = `y2`의 높이, `z_mid = (z(y1) + z(y2)) / 2`
- 가로 (오른쪽 +): 화면 x 픽셀 `u`의 카메라 가로 `x_cam(u) = (u - cx)·dist/f`일 때
  - 걷는 방향 기준 가로 = `x_cam·cos a + z·sin a`
  - `x_left` = `x_cam(x1)`, `z_mid`로 계산 / `x_right` = `x_cam(x2)`, `z_mid`로 계산 / `x_center = (x_left + x_right)/2`
- 앞 거리 (기록용, 판정에는 안 씀): `-x_cam((x1+x2)/2)·sin a + z·cos a`를 `z(y1)`, `z(y2)`로 각각 구해 평균한 것이 `forward`
- `a = 0`이면 11/2 식(`x_left = (x1 - cx)·dist/f` 등)과 정확히 같다.
- 겹친 길이:
  - `overlap_x = min(x_right, 0.4) - max(x_left, -0.4)`
  - `overlap_y = min(top, CORRIDOR_TOP_M) - max(bottom, CORRIDOR_BOTTOM_M)`
- `overlap_x ≥ 0.05` 이고 `overlap_y ≥ 0.05` 이면 경로 안
- 통로의 길이 방향(3 m)은 여기서 따지지 않는다. 3 m 조건은 4-5의 경고(`dist < 3`)가 맡는다.
- 같은 화면 위치라도 가까우면 통로 안, 멀면 통로 밖이 될 수 있다 (화면 고정 사각형과 가장 다른 점). 화면 중심에서 같은 픽셀만큼 떨어진 박스도 거리가 멀수록 실제 옆 거리(m)가 커지기 때문이다.

**B. 화면 사각형** (깊이 기반 모드, 또는 `PATH_MODE=2d`)
- 아래 두 조건을 모두 만족해야 한다. 박스 폭이 0이면 경로 밖이다.
  - 박스 폭 중 `0.40·W ~ 0.60·W` 안에 들어온 비율이 0.3 이상
  - 박스 세로 범위가 `0.25·H ~ 0.75·H`와 겹침 (`min(y2, 0.75H) > max(y1, 0.25H)`)
  - 이 경계값은 소수 그대로 쓴다 (`int` 하지 않음).

### 4-4. 경로 depth: 거리와 위치 (깊이 기반 모드만 후보로 씀)
- 경로 사각형 픽셀: `x int(0.40W) ~ int(0.60W)`, `y int(0.25H) ~ int(0.75H)` (끝은 포함하지 않음)
- depth 전체에 `cv2.medianBlur(depth.astype(np.float32), 5)`를 적용한 뒤 사각형을 잘라낸다.
- 거리 `near` = 사각형 안 하위 5% 백분위
- 위치 = 사각형 안에서 값이 `near + 0.3` 이하인 픽셀들의 x 평균과 y 평균을 각각 `round`한 값 + 사각형 시작점
- 클래스 모드에서도 계산은 하지만(`path_m`) 후보에 넣지 않고 기록에도 안 남는다.

### 4-5. 합치기와 경고
- 후보 (YOLO 출력 순서):
  - 클래스 모드: [경로 안이고 거리가 있는 타겟 박스들]. 타겟 = `all`이면 모든 박스, 지정 클래스면 그 클래스 박스
  - 깊이 기반: [경로 depth 거리] + [경로 안이고 거리가 있는 YOLO 박스들]
  - 경로 밖 박스는 후보가 아니다.
- 장애물 거리 = 후보 중 최솟값. 같은 값이 여럿이면 앞의 것을 고른다 (깊이 기반이면 DEPTH가 맨 앞).
- 후보가 없으면 거리 없음, 경고 없음.
- 경고 = 장애물 거리 **< 3**. 정확히 3이면 경고가 아니다.
- 경고 대상: YOLO면 그 박스, DEPTH면 4-4의 위치

### 4-6. 방향

**A. 3D 통로 모드** (경고 대상이 YOLO 박스일 때)
- 경고 대상 박스의 `x_center`(4-3 A, 걷는 방향 기준)로 정한다. 좌우 보정이 맞으면 걷는 길 정면의 물체가 0 근처로 나온다.
  - `x_center < -0.15` → 왼쪽, `x_center > 0.15` → 오른쪽, 그 외(경계값 포함) → 정면

**B. 화면 사각형 모드** (깊이 기반, `PATH_MODE=2d`)
- 판정에 쓰는 x: DEPTH면 4-4 위치의 x, YOLO면 `(x1 + x2) / 2`
- 경계: `x0 = 0.40·W`, `x1 = 0.60·W`일 때
  - `L = int(x0 + (x1 - x0) * (1/3))`
  - `R = int(x0 + (x1 - x0) * (2/3))`
  - W = 640이면 L = 298, R = 341
- `x < L` → 왼쪽, `x > R` → 오른쪽, 그 외(경계값 포함) → 정면
- 화면에 그려지는 칸 선(W/3, 2W/3)은 보기용이다. 판정 경계와 다르다.

**공통: 방향 유지**
- 경고가 이어지는 동안에는 그 경고 구간 첫 프레임의 방향을 계속 쓴다. 경고가 아닌 프레임이 한 번이라도 나오면 초기화하고, 다음 경고에서 새로 정한다.
- 경고 대상이 바뀌어도(다른 박스가 더 가까워져도) 경고가 끊기지 않으면 방향은 그대로다.

### 4-7. 진동 명령
진동은 **피할 방향**을 알려준다.

| 장애물 위치 `direction` | `motor` | 뜻 |
|---|---|---|
| 왼쪽 | `right` | 오른쪽으로 피하라 |
| 오른쪽 | `left` | 왼쪽으로 피하라 |
| 정면 | `neck` | 정면 장애물 |

- `vib_count` = 2 (경고 = 레벨 2). 경고가 아니면 `motor`와 `vib_count` 모두 빈칸이다.
- 레벨 1(주의)·3(위험)은 쓰지 않는다.

### 화면 표시 (판정과 무관)
- 3D 통로 모드: 통로를 화면에 투영해 그린다 (4-3 A의 반대 방향 계산, 좌우 각도·숙임 반영). 3 m 지점 단면 사각형(굵게) + 0.3 m 지점까지 이어지는 네 모서리 선 + 1·2 m 단면(얇게, 거리 글자). 정상 초록, 경고 빨강.
- 통로 양 벽과 3 m 단면에 점선 두 개: 하늘색 = 눈높이(`CAMERA_HEIGHT_M`, `eye`), 주황 = 머리 꼭대기(+`HEAD_ABOVE_EYE_M`, `head`). 통로 천장(굵은 사각형 윗변)은 머리선보다 0.1 m 위다. 숙임 0°면 눈높이선은 화면 가운데를 가로지르는 수평선이 되고, 그보다 위에 찍힌 것은 눈보다 높은 곳에 있다. 화면 3분할 선과 경로 depth 점은 안 그린다.
- 단면은 카메라 높이·숙임·좌우·화각으로 **계산한 눈금**이고 depth와 무관하다. 경고는 depth 거리 `< 3`으로 정하므로, 물체가 그림의 3 m 사각형보다 멀어 보여도 경고가 뜰 수 있다. 그림 눈금이 바닥 테이프와 안 맞으면 카메라 설정값을, 테이프 위 물체의 depth 거리가 안 맞으면 `DEPTH_SCALE`을 의심한다 (8-1).
- 화면 사각형 모드: 경로 사각형, 3분할 선, 경로 안 가장 가까운 점(원)
- 공통: 경로 안 박스는 굵게, 밖은 얇게. 위쪽에 `OBSTACLE 2.85m (YOLO)`, 경고면 그 아래 크게 `-> NECK`

---

## 5. 저장 (영상 입력, 또는 실시간 시행 기록 중일 때)

> 이 장이 RUN_FORMAT이다 (schema_version 2). 팀에 따로 `RUN_FORMAT.md`가 있으면 아래 바뀐 것들을 거기에도 적는다.
>
> **버전 2에서 바뀐 것 (10/26)**: `video_source` 칸 추가 (run_meta.json, summary.json), `server_ms_median` 칸 추가 (summary.json). 실시간 기록에 있던 `source` 칸은 `video_source`로 바뀌었다. `alarm_constants`에 `FRAME_WIDTH`, `CONF_THRES`, `USE_DEPTH`, `pipeline`에 `depth_every_n`, `openvino`가 추가됐다. frames.csv는 그대로다.
>
> **버전 2 안에서 칸이 늘어난 것 (10/30 ~ 11/8)**: frames.csv·summary.json 칸은 그대로다. `pipeline`에 `yolo_weights`, `depth_backend`, `depth_model`, `depth_imgsz`, `depth_coreml`, `alarm_constants`에 `TARGET_LABELS`, `PATH_MODE`, `USE_CORRIDOR`, `CAMERA_HEIGHT_M`, `CAMERA_PITCH_DEG`, `CAMERA_YAW_DEG`, `CORRIDOR_BOTTOM_M`, `CORRIDOR_TOP_M`, `CORRIDOR_MIN_OVERLAP_M`, `CORRIDOR_FRONT_M`가 추가됐고, 클래스 모드면 타겟 폴더가 생긴다. 10장 규칙상 `schema_version`을 3으로 올릴지는 팀에서 정한다 (지금은 2 그대로, 11장).
>
> 녹화(`run_model.py`)와 실시간(`live_record.py`)은 `judge.py`의 같은 함수(`frame_row`, `run_meta`, `summary`, `TargetRecorder`)로 저장하므로 칸 이름과 순서가 같다. 실시간에는 `latency.csv`(구간별 지연)가 하나 더 있다.

폴더는 `runs/<runner>/<시행ID>_<MMDD>_<HHMMSS>/`이다 (뒤의 날짜·시각은 돌린 시각). 폴더나 파일이 이미 있으면 오류가 나고 덮어쓰지 않는다. 표시 영상은 기본으로 저장하지 않고, `--save_video`를 주면 `annotated.mp4`(처리한 프레임마다 한 장, 원본 fps, 클릭 표시 없음)로 저장한다. 채점에는 쓰지 않는다.

```
runs/<runner>/<시행ID>_<MMDD>_<HHMMSS>/
├─ run_meta.json, frames.csv, summary.json, (실시간) latency.csv   ← 전체
├─ annotated.mp4         ← --save_video일 때만 (녹화 영상)
└─ target_all/            ← 클래스 모드일 때만 (지정 클래스면 target_suitcase/, target_bottle_chair/ 등)
   ├─ run_meta.json
   ├─ frames.csv
   ├─ latency.csv
   └─ summary.json
```

### frames.csv
프레임마다 한 줄씩 쓴다. 인코딩은 utf-8-sig이고, 일시정지 중에는 줄을 쓰지 않는다. 칸 순서:

`trial, frame_idx, t_ms, processed, n_dets, labels, dets_json, target_label, target_conf, target_cx, dist, ttc, level, level_name, direction, beep, vibrate_ms, tts, yolo_ms, depth_ms, target_box, motor, vib_count`

| 칸 | 값 |
|---|---|
| `trial` | 시행ID |
| `frame_idx` | 0부터 (실시간: 시행 시작 때 0으로 돌아감) |
| `t_ms` | 녹화: `cap.read()` 직후 읽은 `CAP_PROP_POS_MSEC`를 `round(max(값, 0))`. 두 번째 프레임부터 값이 0 이하로 나오면 그 뒤로는 `round(frame_idx × 1000 / fps)`를 쓰고 터미널에 한 번 경고. 실시간: 시행 시작(Enter)부터 서버가 그 프레임을 받은 시각 |
| `processed` | 1 |
| `n_dets`, `labels` | 박스 수, 클래스 이름을 `;`로 이은 것 (모든 박스) |
| `dets_json` | `json.dumps([{"label", "conf", "box", "dist"}, ...], ensure_ascii=False)`. conf·dist는 소수 3자리(dist 없으면 null), box는 원본 픽셀 정수 `[round(x1·sx), …]` |
| `dist` | 장애물 거리(4-5), 소수 3자리 (경고가 아니어도 기록. 후보가 없으면 빈칸) |
| `level`, `level_name` | 경고면 `2` / `경고`, 아니면 `0` / `없음` |
| `target_label` | 경고일 때: YOLO 클래스 이름 또는 `DEPTH`(깊이 기반만) |
| `target_conf` | 경고 + YOLO일 때: 소수 3자리 |
| `target_cx` | 경고일 때: YOLO면 `round((x1 + x2) / 2 × sx)`, DEPTH면 `round(위치 x × sx)` |
| `target_box` | 경고 + YOLO일 때: `x1 y1 x2 y2` (각각 `round(v × sx)`, 공백으로 구분) |
| `direction` | 경고일 때: `왼쪽` / `정면` / `오른쪽` |
| `motor`, `vib_count` | 경고일 때: 4-7 |
| `yolo_ms`, `depth_ms` | 처리 시간, 소수 1자리 (depth를 안 돌린 프레임은 빈칸) |
| `ttc`, `beep`, `vibrate_ms`, `tts` | 항상 빈칸 |

### summary.json
종료할 때 한 번 쓴다. `q`나 Ctrl+C로 중간에 끝내도 쓴다.

| 칸 | 값 |
|---|---|
| `trial` | 시행ID |
| `video_source` | `file`(녹화 영상) / `live`(실시간) |
| `video` | 영상 파일명 (실시간은 null) |
| `frames`, `processed` | frames.csv 줄 수, 처리한 줄 수 |
| `frame_size`, `video_fps`, `duration_sec` | 원본 크기, fps, 길이. 녹화: 영상 정보 (길이 = 프레임 수 / fps, 소수 3자리). 실시간: 서버가 처리한 프레임 수 / 마지막 `t_ms` |
| `frames_with_det`, `frames_with_target` | 박스가 있던 프레임 수, 경고 프레임 수 |
| `label_counts` | 클래스별 등장 횟수 |
| `first_level2` | 첫 경고 줄의 `{t_sec, dist, direction}` 또는 null |
| `first_vibration` | `motor`가 처음 채워진 줄의 `{t_sec, dist, direction, target_label}` 또는 null |
| `first_level1`, `first_level3` | null |
| `max_level` | 가장 높은 레벨 |
| `yolo_ms_median`, `depth_ms_median` | 처리 시간 중앙값, 소수 1자리 |
| `server_ms_median` | 실시간: 서버가 프레임 하나를 처리한 시간(받은 사진 풀기 ~ 회신 준비) 중앙값, 소수 1자리. 녹화: null |
| `wall_sec` | 실제 걸린 시간 |
| `expected_dir`, `gt_2m_sec`, `note` | 인자 그대로 (실시간은 null) |

### run_meta.json
시작할 때 한 번 쓴다.

| 칸 | 값 |
|---|---|
| `run_at` | 돌린 시각 |
| `video_source` | `file` / `live` |
| `args` | 인자 전부 (`trial`, `depth_every_n`은 실제로 쓴 값). 실시간: `trial`, `device`, `depth_every_n` |
| `pipeline` | `device`, `yolo_weights`, `depth_backend`, `depth_model`, `depth_imgsz`, `depth_coreml`, `conf`, `imgsz`, `depth_every_n`, `openvino`, torch·ultralytics·transformers·opencv 버전 |
| `alarm_constants` | 판정 설정값: `FRAME_WIDTH`, `CONF_THRES`, `USE_DEPTH`, `MODEL_PATH`, `IMG_SIZE`, `PATH_WIDTH_M`, `HFOV_DEG`, `WARN_M`, `PATH_Y`, `PATH_ROI`, `BOX_SHRINK`, `BOX_PERCENTILE`, `NEAR_PERCENTILE`, `NEAR_BAND_M`, `HOLD_DIRECTION`, `BOX_OVERLAP_MIN`, `ZONE_MODE`, `ZONE_DRAW`, `ZONE_PATH_SPLIT`, `MOTOR_MAP`, `VIB_COUNT`, `RUNS_DIR`, `SCHEMA_VERSION`, `DEPTH_SCALE`, `choices`, `TARGET_LABELS`(`["all"]` / `["suitcase"]` / null = 깊이 기반), `PATH_MODE`, `USE_CORRIDOR`, `CAMERA_HEIGHT_M`, `CAMERA_PITCH_DEG`, `CAMERA_YAW_DEG`, `CORRIDOR_BOTTOM_M`, `CORRIDOR_TOP_M`, `CORRIDOR_MIN_OVERLAP_M`, `CORRIDOR_FRONT_M` |
| `runner` | 이름 |
| `git_commit` | `git rev-parse HEAD`. 실패하면 `"unknown"` |
| `video_file` | 파일명만 (경로 제외). 실시간은 null |
| `schema_version` | 2 |

### 타겟 폴더 (`target_<...>/`, 클래스 모드만)

전체 결과와 같은 구성인데, **타겟 클래스 박스를 잡은 프레임만** 남긴다. 전체 `frames.csv`와는 `frame_idx`로 이을 수 있다. `all` 모드면 모든 박스가 타겟이라 프레임에 박스가 하나라도 있으면 줄이 생긴다.

- `run_meta.json`: 전체와 같음 + `target_labels`(`["all"]` 등), `target_schema_version`(1)
- `latency.csv`: 타겟을 잡은 프레임의 지연. 칸은 `frame_idx, t_ms, recv_to_start_ms, decode_ms, yolo_ms, depth_ms, server_ms`. 녹화는 `yolo_ms`·`depth_ms`만 있고 나머지는 빈칸

**frames.csv**: 타겟 박스 1개당 1줄 (통로 밖 박스도 줄을 쓴다)

| 칸 | 값 |
|---|---|
| `trial`, `frame_idx`, `t_ms` | 전체 frames.csv와 같음 |
| `det_idx`, `n_targets` | 그 프레임 안 타겟 번호(0부터), 타겟 수 |
| `label`, `conf` | 클래스, 확률(소수 3자리) |
| `x1 y1 x2 y2`, `box_w_px`, `box_h_px`, `box_cx`, `box_cy` | 박스 (원본 픽셀, 각각 `round(v × sx)`) |
| `box_clipped` | 1이면 화면 끝에 잘림 (축소 화면에서 `x1 ≤ 0`, `y1 ≤ 0`, `x2 ≥ W-1`, `y2 ≥ H-1` 중 하나) → 크기·중심 부정확 |
| `dist` | 박스 거리(m, 4-2). 없으면 빈칸 |
| `depth_fresh` | 1이면 이 프레임에서 새로 계산한 depth, 0이면 직전 depth (`depth_every_n` > 1일 때) |
| `forward_m` | 사용자 기준 앞 거리(m). 숙임각 0이면 `dist`와 같음 |
| `x_left_m`, `x_right_m`, `x_center_m` | 통로 중심선(걷는 방향) 기준 가로 위치(m, 오른쪽 +) |
| `top_m`, `bottom_m` | 바닥 기준 높이(m) |
| `width_m`, `height_m` | 추정 크기(m). 실제 크기와 비교하면 화각·거리 점검이 됨 |
| `overlap_x_m`, `overlap_y_m` | 통로와 겹친 길이(m). 음수면 그만큼 떨어져 있음 |
| `lateral_ok`, `vertical_ok` | 가로·세로 겹침 통과 (0/1) |
| `in_path` | 이 프레임 판정에 쓴 경로 안 여부 (`PATH_MODE=2d`면 화면 사각형 기준) |
| `in_warn_range` | `dist < WARN_M` (0/1, dist 없으면 빈칸) |
| `is_warn_target` | 이 박스가 경고 대상이면 1 |
| `level`, `level_name` | 그 프레임의 경고 |
| `direction`, `motor`, `vib_count` | 경고 대상 줄에만 |
| `yolo_ms`, `depth_ms` | 처리 시간 (depth 안 돌린 프레임은 빈칸) |

미터 칸(`forward_m` ~ `vertical_ok`)은 거리가 있을 때만 채운다. `PATH_MODE=2d`에서도 3D 통로 기준 값으로 계산해서 남긴다 (비교용).

**summary.json**

| 칸 | 값 |
|---|---|
| `trial`, `video_source`, `video`, `target_labels` | |
| `frames_total` | 처리한 전체 프레임 수 |
| `frames_detected`, `detection_rate` | 타겟을 잡은 프레임 수, 그 비율 (소수 4자리) |
| `longest_miss_frames` | 타겟을 연달아 놓친 가장 긴 구간 (프레임 수) |
| `detections` | 타겟 frames.csv 줄 수 |
| `frames_in_path`, `frames_warned` | 경로 안 타겟이 있던 프레임 수, 경고 프레임 수 |
| `first_detection`, `first_in_path`, `first_warning` | `{frame_idx, t_sec, dist, x_center_m, direction, motor}` 또는 null |
| `conf_median`, `conf_min_max`, `dist_median`, `dist_min_max` | |
| `width_m_median`, `height_m_median` | 추정 크기 중앙값 (잘린 박스 제외) |
| `yolo_ms_median`, `depth_ms_median`, `server_ms_median` | 지연 중앙값 (녹화면 `server_ms_median` null) |
| `focal_px` | 쓴 초점거리 (축소 화면 기준 픽셀) |
| `corridor` | 쓴 통로 설정 `{width_m, bottom_m, top_m, length_m, camera_height_m, camera_pitch_deg, camera_yaw_deg, hfov_deg}`. 화면 사각형 모드면 null |
| `wall_sec`, `expected_dir`, `gt_2m_sec`, `note`, `target_schema_version` | |

---

## 6. 채점 (`score.py`, 만들 예정)

```bash
python score.py --runs runs/<이름> --trials trials.csv
```

run 폴더는 읽기만 하고, 결과는 `scores/<MMDD_HHMMSS>/`에 쓴다 (`score.json`, `trial_scores.csv`). 채점은 전체 `frames.csv`로 한다 (타겟 폴더는 분석용).

### trials.csv (사람이 채우는 정답표, 영상 1개당 1줄)

| 칸 | 값 |
|---|---|
| `trial` | 시행ID |
| `condition` | `front` / `left` / `right` / `none` |
| `place` | `복도` / `스룸` |
| `expected_dir` | 장애물 위치. 비우면 front→정면, left→오른쪽, right→왼쪽 |
| `gt_far_sec` | 바깥 경계 통과 시각(초). 복도는 4 m, 스룸은 3 m 지점 |
| `gt_2m_sec` | 2 m 지점 통과 시각(초) |
| `end_sec` | none 시행만: 장애물이 있던 위치를 지난 시각(초). 비우면 영상 끝까지 |
| `stopped` | 실착용 시행만: 장애물 전에 멈췄으면 1, 닿았으면 0 |
| `note` | 메모 |

조건 이름이 곧 피할 방향이다. `left`는 장애물이 오른쪽에 걸쳐 있어 왼쪽으로 피하는 경우이고, 정답 모터는 `left`다. `right`의 정답 모터는 `right`, `front`의 정답 모터는 `neck`이다. 바닥에 2 m와 4 m(스룸은 3 m) 지점을 테이프로 표시해 두면 영상에서 통과 시각을 잴 수 있다.

### 시행별 판정
"진동 줄" = `level ≥ 2`이고 `motor`가 빈칸이 아닌 줄

- **장애물 시행**: 영상 시각이 `gt_far_sec` 이상, `gt_2m_sec` 미만인 구간에서
  - 구간 안 첫 진동의 모터가 정답과 같음 → **성공**
  - 첫 진동의 모터가 다름 → **방향틀림**
  - 구간 안엔 없고 2 m 이후에만 있음 → **늦음**
  - 아예 없음 → **없음**
- **none 시행**: `end_sec` 전에 진동 줄이 하나라도 있으면 **오경고**, 없으면 **정상**

### 지표

| 지표 | 식 | 목표 |
|---|---|---|
| 회피 안내 성공률 | 성공 ÷ 장애물 시행 (전체와 방향별 둘 다) | ≥ 85% |
| 오경고율 | 오경고 ÷ none 시행 | ≤ 10% |
| 정지 성공률 | `stopped = 1` ÷ (진동이 있었고 `stopped`를 적은 시행) | 보고만 |

- 두 목표를 점추정치로 모두 만족하면 "달성"이다. Wilson 95% 신뢰구간을 같이 적는다.
  - 확인용 값: 26/30 → 86.7% [70.3, 94.7], 3/30 → 10.0% [3.5, 25.6]
- 시행이 30회보다 적으면 "시행 수 부족"이라고 표시한다.
- 판정 모드(`TARGET_LABELS`, `PATH_MODE`)가 다른 run은 한 표에 섞지 않는다. `TARGET_LABELS`만 다르면(`all` vs `suitcase`) `choices`가 같아서 경고가 안 뜨니 `alarm_constants.TARGET_LABELS`를 직접 확인한다.

---

## 7. 진동 인지 시험 (11주, `score.py --haptic`, 보조 지표)

모터 위치(`left` / `right` / `neck`)와 횟수(1·2회)를 무작위 순서로 울리고 착용자가 맞힌다. 결과는 `haptic_test.csv`에 제시 1번당 1줄로 적는다.

칸: `participant, trial_no, shown_motor, shown_count, answer_motor, answer_count, rt_ms`

결과로 나오는 것:
- 위치 구분율, 횟수 구분율 (Wilson 구간 포함)
- 혼동표: 위치 3×3, 횟수 2×2
- 반응 시간 중앙값

---

## 8. 파일럿 4회

### 8-1. 실험 전에 잴 값

3D 통로는 아래 값으로 미터 좌표를 계산하므로, 틀리면 경로 안/밖과 방향이 같이 틀어진다.

| 값 | 기본 | 틀리면 |
|---|---|---|
| `HFOV_DEG` | 70 (미확인) | 좌우 거리가 전부 비례해서 틀어짐. **가장 중요** |
| `DEPTH_SCALE` | 1.0 | 거리와 좌우 거리가 같이 틀어짐. YOLO26-depth 기준으로 다시 잴 것 |
| `CAMERA_HEIGHT_M` | 1.65 | 높이 판정과 통로 천장이 틀어짐. 착용자마다 줄자로 (바닥 ~ 안경 카메라) |
| `CAMERA_YAW_DEG` | 9 (잠정) | 3 m 앞에서 1°당 약 5 cm씩 통로가 옆으로 비켜서, 방향(±0.15 m)이 틀어짐. 9°를 안 넣으면 약 0.48 m |
| `CAMERA_PITCH_DEG` | 0 | 앞 3 m에서 10°면 높이가 약 0.5 m 틀어짐. 15°쯤까지는 바닥 물체가 통로 안에 남지만, 30° 넘게 숙이면 바닥 물체가 통로 위로 계산돼 경고가 안 뜸. 걸으면서 계속 바뀌어 고정값으로는 보정 안 됨 (11장) |

**화각 재는 법**: 폭을 아는 물체(수트케이스 등)를 정확히 2 m 앞에 정면으로 두고 정지 촬영한 뒤, 타겟 `frames.csv`의 `box_w_px`(원본 픽셀)로 계산한다.

```
HFOV = 2 × atan( 영상 가로 픽셀 × 실제 폭(m) / (2 × box_w_px × 실제 거리(m)) )
```

- 여러 장의 중앙값을 쓴다. 비스듬히 놓으면 박스가 넓어진다.
- `HFOV_DEG`는 영상 **가로(폭) 방향** 화각이다. 세로 영상이면 짧은 변 쪽이다.
- **영상 종류별로 따로 잰다.** 지금까지 run 기록에서 실시간은 360×640(가로/세로 0.56), 녹화는 1080×1432·1442(0.75)로 비율이 다르다. 잘린 범위가 다르면 화각도 다르다.
- 잰 뒤 확인: 같은 영상을 다시 돌려 타겟 summary의 `width_m_median`이 실제 폭과 맞는지 본다.

**좌우 각도 재는 법**: 벽이나 바닥 줄눈처럼 걷는 방향과 나란한 직선이 보이는 곳에서, 그 선과 나란히 걸으며(또는 서서) 5초 찍는다. 그 선들이 모이는 점(소실점)의 화면 x를 `u`라고 하면

```
CAMERA_YAW_DEG = atan( (W/2 - u) / f )      (f = (W/2) / tan(HFOV_DEG/2), 숙임이 작을 때)
```

- 11/8 잠정값 9°는 1009 front/left/right(같은 착용자·같은 날) 앞부분 프레임의 중앙값(7.9 ~ 9.7°)이다. 다른 날 교실 영상은 약 6°였다.
- 화각을 다시 재면 같은 화각으로 이 값도 다시 잰다.
- 걷는 방향과 나란한 선이 없으면, 몸 중심선 정면 3 m에 물체를 두고 가만히 서서 찍은 뒤 그 물체의 화면 x를 `u`로 쓴다.

### 8-2. 촬영과 실행

조건마다 1회씩 촬영해서 촬영 → 실행 → 채점이 끝까지 이어지는지 확인한다. 4회로는 목표 달성 여부를 말할 수 없다.

1. `<MMDD>_front_01`, `_left_01`, `_right_01`, `_none_01`을 촬영한다. 같은 5 m 경로, 같은 촬영 모드로 찍는다.
2. 영상마다 통과 시각을 재서 `trials.csv` 4줄을 쓴다.
3. 코드를 커밋한 뒤 4번 돌린다: `CAMERA_HEIGHT_M=<잰 값> HFOV_DEG=<잰 값> python run_model.py --video <영상> --runner <이름>`
4. `python score.py --runs runs/<이름> --trials trials.csv`
5. 결과와 해석을 `experiments.md`에 적는다.

볼 것:
- 장애물을 YOLO가 잡는지 (타겟 summary의 `detection_rate`, `longest_miss_frames`). 못 잡으면 클래스 모드에서는 경고가 안 뜬다
- left/right 시행에서 `x_center_m`이 실제 옆 거리(예: 30 cm)와 맞는지, ±0.15 m 경계를 제대로 넘는지
- 경로에 사람이 들어와 생긴 경고가 있는지 (`all` 모드에서는 사람도 장애물)
- none 시행 오경고의 원인 (어떤 클래스, 어느 위치)
- 추정 거리와 실제 거리 차이 (`DEPTH_SCALE` 보정이 필요한지)
- 깊이 기반 모드로 같은 영상을 돌렸다면: 첫 경고가 DEPTH인지 YOLO인지, 옆 벽이 경로 사각형에 들어와 처음부터 경고가 뜨지 않는지

---

## 9. 돌린 뒤 확인

- [ ] 터미널 첫 줄 `판정 모드: ...`가 의도한 모드다
- [ ] run 폴더에 `run_meta.json`, `frames.csv`, `summary.json` 세 파일이 있다 (실시간은 `latency.csv`도)
- [ ] 클래스 모드면 `target_<...>/` 안에 네 파일이 있다
- [ ] `frames.csv` 줄 수 = `summary.json`의 `frames`
- [ ] `git_commit`이 `unknown`이 아니다 (커밋하고 돌렸다)
- [ ] `alarm_constants`의 `HFOV_DEG`, `CAMERA_HEIGHT_M`이 잰 값이다
- [ ] 폴더명의 시행ID = 영상 파일명
- [ ] 경고 줄에만 `motor`가 있고, 한 경고 구간 안에서는 `motor`가 바뀌지 않는다
- [ ] 정면 장애물 시행은 경고 내내 `neck`
- [ ] 엑셀로 열었을 때 한글이 안 깨진다

---

## 10. 하지 말 것

- 각자 코드를 고쳐서 돌리기. 바꿀 게 있으면 팀에서 정하고 커밋한 뒤 모두 다시 돌린다.
- 각자 다른 환경변수로 돌리기. 판정이 바뀌는 환경변수(2장 표)는 팀 전체가 같은 값을 쓴다.
- 판정이나 처리 순서를 `run_model.py`·`server.py`에서 고치기. 녹화와 실시간이 갈라진다. `judge.py`에서 고친다.
- 시행마다 코드의 `SOURCE` 고치기. `--video`를 쓴다.
- 클래스별로 규칙을 다르게 두기 (사람이면 ~, 의자면 ~). `TARGET_LABELS`는 후보에 넣을지 거르는 데만 쓴다.
- RUN_FORMAT 칸 이름 바꾸기·칸 더하기. 바꾸려면 팀 합의 후 `schema_version`을 올린다.
- 성공률 같은 채점 결과를 run 폴더에 넣거나 run 폴더 파일을 손으로 고치기
- 새 패키지 추가하기. `score.py`는 파이썬 기본 모듈만 쓴다.

---

## 11. 아직 정할 것

- `WARN_M` 3 m, 정면 장애물이면 `neck`: 팀 확정
- 타겟 범위: `all`로 계속 갈지, 장애물을 정하면 그 클래스로 좁힐지 (`all`이면 경로의 사람도 경고)
- YOLO가 못 잡는 물체(폼보드 간판 등): 파인튜닝할지, 깊이 기반 모드를 같이 쓸지
- `schema_version`을 3으로 올릴지 (`alarm_constants`·`pipeline` 칸과 타겟 폴더가 늘었음)
- 목표의 "30회"가 방향별 30회인지 전체 30회인지
- `--sim_fps`의 뜻과 기본값
- `trials.csv`의 `place`, `gt_far_sec`, `end_sec`를 RUN_FORMAT §4에 추가할지
- 촬영 모드 확정, 영상 종류별 `HFOV_DEG` 실측, YOLO26-depth 기준 `DEPTH_SCALE`
- `CAMERA_YAW_DEG` 9° 확정 (다른 착용자·다른 날 측정, 화각 실측 뒤 다시 재기)
- 고개 숙임: 지금은 숙이지 않는다고 가정(`CAMERA_PITCH_DEG` 고정). 걸으며 바뀌는 숙임을 프레임마다 잴지 (예: depth 바닥 평면)

---

## 부록: 왜 이렇게 정했나 (코드 주석의 번호)

| 번호 | 선택 | 이유 |
|---|---|---|
| 2-1 A | 화면 경로 = 고정 사각형 (x 0.40~0.60, y 0.25~0.75) | 3 m 앞에서 폭 0.8 m가 되도록 화각 70°로 계산함. y는 천장·바닥 제외. 지금은 깊이 기반·`PATH_MODE=2d`에서만 씀 |
| 2-2 B | 박스 거리 = 줄인 영역 하위 30% | 박스 중앙 한 점은 배경에 걸리면 틀림 |
| 2-3 | 경로 밖 박스는 경고 안 함 | 거리만 표시 |
| 2-5 B | 방향 = 경로 사각형 3등분 (화면 사각형 모드) | 화면 전체 3등분이면 경로가 통째로 정면 칸이라 DEPTH 경고가 항상 `neck`이 됨 |
| 2-5 corridor | 방향 = 통로 중심선에서 ±0.15 m (3D 통로 모드) | 화면 위치가 아니라 실제 옆 거리로 판단. 0.15는 옆 접근 시행(30 cm)과 정면(0 cm)의 중간 |
| 2-7 B | 박스 폭 30% 이상이 사각형 안이면 경로 안 (화면 사각형 모드) | 가장자리에 살짝 닿은 박스는 무시 |
| 2-7 3D 통로 | 미터 좌표로 통로와 가로·세로 5 cm 이상 겹치면 경로 안 | 화면 고정 사각형은 3 m 지점에서만 폭 0.8 m가 맞음. 그보다 가까우면 실제 폭이 0.8 m보다 좁게, 멀면 넓게 잡힘. 5 cm는 가장자리 흔들림으로 깜빡이지 않게 |
| 2-9 B | 경고 거리 3 m | 2 m면 경고가 2 m 안에서만 시작돼 성공률이 0%가 됨 |
| 2-10 A | 정면 장애물이면 `neck` | 피할 방향 대신 멈춤 신호 |
| 2-11 B | DEPTH 위치 = 가까운 픽셀들의 중심 | 한 점은 프레임마다 흔들려 좌우가 섞임 |
| 2-12 B | 경고 동안 첫 방향 유지 | 착용자에게 좌우가 번갈아 울리지 않게. 단, 첫 방향이 틀리면 그 경고 내내 틀림 |
| 클래스 모드 (10/16) | 지정 클래스 박스만 장애물, DEPTH 후보 제외 | 정한 물체(물병, 11/2부터 수트케이스)만 보고 시험하려고 |
| 11/8 | 좌우 보정 `CAMERA_YAW_DEG` 9° | 메타 안경 영상 3개 모두 카메라가 걷는 방향보다 오른쪽 8 ~ 10°를 봄 (벽 소실점과 화면 흐름 방향으로 확인). 보정 없이는 3 m 앞 정면 물체가 약 0.48 m 왼쪽으로 계산돼 `right`가 울림 |
| 11/8 | 기본 타겟 `all` | 실험 장애물을 아직 정하지 못했음. 물체 종류는 판정에 쓰지 않는다는 원칙과도 맞음 |
