"""
baseline_report.py — 시스템 기본 성능(베이스라인) 표를 run 폴더들에서 뽑기  [3-1조]

  python tools/baseline_report.py runs/이름
  (여러 폴더면 이어서: python tools/baseline_report.py runs/지은 runs/연수)

1) 지연·속도: 실시간 서버(live) 기록의 latency.csv 전체 → 구간별 중앙값, 처리 fps
2) 거리 오차·탐지율: 시행ID에 실제 거리가 들어간 정지 촬영 기록 → 거리별 표
     시행ID 예) 1017_d1m_01, 1017_d2m_01, 1017_d2.5m_01  (← "_d<거리>m_" 형식)
   - 추정 거리 = frames.csv의 dist(판정에 쓰는 장애물 거리) 중앙값
   - 탐지율 = YOLO 박스가 경로 영역 안에 잡힌 프레임 비율
결과는 화면에 출력 + 첫 폴더에 baseline_report.md 저장
"""

import csv
import json
import re
import statistics as st
import sys
from pathlib import Path

DIST_RE = re.compile(r"_d(\d+(?:\.\d+)?)m(?:_|$)")


def med(v):
    v = [x for x in v if x is not None]
    return round(st.median(v), 1) if v else None


def pct(v, q):
    v = sorted(x for x in v if x is not None)
    if not v:
        return None
    k = (len(v) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def runs_in(folders):
    for f in folders:
        for meta in sorted(Path(f).rglob("run_meta.json")):
            d = meta.parent
            if (d / "frames.csv").exists():
                yield d


def read_csv(p):
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def in_path(box, fw, fh, roi):
    """judge.box_in_path와 같은 규칙 (박스 폭의 BOX_OVERLAP_MIN 이상이 경로 x 범위 안 + 세로 겹침)"""
    x1, y1, x2, y2 = box
    if x2 - x1 <= 0:
        return False
    rx0, rx1, ry0, ry1 = roi[0] * fw, roi[1] * fw, roi[2] * fh, roi[3] * fh
    return (max(0, min(x2, rx1) - max(x1, rx0)) / (x2 - x1) >= roi[4]) and (min(y2, ry1) > max(y1, ry0))


def main(folders):
    lat = {k: [] for k in ("recv_to_start_ms", "decode_ms", "yolo_ms", "depth_ms", "server_ms")}
    fps_list, n_live = [], 0
    static = {}   # 실제 거리 → {dist:[], det:[], frames:int, trials:[]}
    for d in runs_in(folders):
        meta = json.load(open(d / "run_meta.json", encoding="utf-8"))
        summ = json.load(open(d / "summary.json", encoding="utf-8")) if (d / "summary.json").exists() else {}
        trial = summ.get("trial") or meta.get("args", {}).get("trial") or d.name
        rows = read_csv(d / "frames.csv")
        if (d / "latency.csv").exists():
            n_live += 1
            for r in read_csv(d / "latency.csv"):
                for k in lat:
                    lat[k].append(num(r.get(k)))
            if summ.get("video_fps"):
                fps_list.append(summ["video_fps"])
        m = DIST_RE.search(trial)
        if not m:
            continue
        actual = float(m.group(1))
        c = meta.get("alarm_constants", {})
        roi = list(c.get("PATH_ROI", [0.4, 0.6, 0.25, 0.75])) + [c.get("BOX_OVERLAP_MIN", 0.3)]
        fw, fh = (summ.get("frame_size") or [None, None])
        s = static.setdefault(actual, {"dist": [], "det": [], "trials": []})
        s["trials"].append(trial)
        for r in rows:
            s["dist"].append(num(r["dist"]))
            hit = False
            if fw and r.get("dets_json"):
                hit = any(in_path(o["box"], fw, fh, roi) for o in json.loads(r["dets_json"]))
            s["det"].append(hit)

    out = ["# 시스템 기본 성능 (베이스라인)", ""]
    out += ["## 지연·속도 (실시간 서버 기록 %d개, 프레임 %d개)" % (n_live, len([x for x in lat["server_ms"] if x is not None])), "",
            "| 구간 | 중앙값 |", "| --- | --- |"]
    names = [("recv_to_start_ms", "서버 대기 (받은 뒤 처리 시작까지)"), ("decode_ms", "JPEG 디코딩"),
             ("yolo_ms", "YOLO"), ("depth_ms", "깊이 추정"), ("server_ms", "서버 전체 (판정 포함)")]
    for k, n in names:
        out.append(f"| {n} | {med(lat[k])} ms |")
    out.append(f"| 처리 속도 | {med(fps_list)} fps |")
    out += ["", "폰 쪽(전송~회신 RTT, 네트워크)은 Android Studio Logcat `NET_METRICS`에서 따로 읽기", ""]

    out += ["## 거리 추정·탐지 (정지 촬영)", "", "| 실제 | 추정 (중앙값) | 오차 | 흔들림 (25~75%) | 탐지율 | 프레임 | 시행 |",
            "| --- | --- | --- | --- | --- | --- | --- |"]
    for actual in sorted(static):
        s = static[actual]
        dv = [x for x in s["dist"] if x is not None]
        if dv:
            m_ = st.median(dv)
            spread = f"{pct(dv, .25):.2f}~{pct(dv, .75):.2f} m"
            est, err = f"{m_:.2f} m", f"{m_ - actual:+.2f} m ({(m_ - actual) / actual * 100:+.0f}%)"
        else:
            est = err = spread = "-"
        rate = sum(s["det"]) / len(s["det"]) * 100 if s["det"] else 0
        out.append(f"| {actual:g} m | {est} | {err} | {spread} | {rate:.0f}% | {len(s['dist'])} | {len(s['trials'])} |")
    if not static:
        out.append("| (시행ID에 _d<거리>m_ 가 들어간 기록이 없음) | | | | | | |")
    out += ["", "추정 거리 = 판정에 쓰는 장애물 거리(경로 깊이와 경로 안 박스 거리 중 가까운 값)",
            "탐지율 = YOLO 박스가 경로 영역 안에 잡힌 프레임 비율"]
    text = "\n".join(out)
    print(text)
    p = Path(folders[0]) / "baseline_report.md"
    p.write_text(text + "\n", encoding="utf-8")
    print(f"\n저장: {p}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("사용법: python tools/baseline_report.py runs/이름")
    main(sys.argv[1:])
