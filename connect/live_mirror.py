"""Mirror existing server annotations; attach phone alarm telemetry only by exact frame ID."""
import json, threading, time
from collections import OrderedDict
from live_publisher import LivePublisher

class LiveMirror:
    def __init__(self,publisher=None,wait_for_phone=False):
        self.wait_for_phone=wait_for_phone;self.publisher=publisher or LivePublisher();self.started=time.monotonic();self.pending=OrderedDict();self.lock=threading.Lock()
    # [3-1조] 서버 판정(judge.frame_row 한 줄)을 사이트 record에 넣을 때 쓰는 칸 (live_bridge FIELDS 안에 있는 것만)
    JUDGE_KEYS=('dist','ttc','level','level_name','direction','motor','vib_count','vibrate_ms','target_label','target_conf','target_box')
    def remember(self,jpeg_bytes,response,frame_idx,judge_row=None):
        record={'frame_idx':frame_idx,'t_ms':round((time.monotonic()-self.started)*1000,3),'processed':1,'decision_available':False,'dist':None,'ttc':None,'level':None,'level_name':'','direction':'','motor':'','vib_count':0,'vibrate_ms':0,'dets_json':json.dumps(response.get('dets',[]),ensure_ascii=False),'yolo_ms':response.get('yolo_ms'),'depth_ms':response.get('depth_ms')}
        if judge_row is not None:   # [3-1조] 서버 판정을 같이 보냄 (frames.csv와 같은 값)
            record.update({k:judge_row[k] for k in self.JUDGE_KEYS if k in judge_row})
            for k in ('vib_count','vibrate_ms'):   # frames.csv에선 빈칸이지만 중계 서버는 0 이상 숫자를 요구함
                if record.get(k) in ('',None):record[k]=0
            for k in ('dist','ttc'):
                if record.get(k)=='':record[k]=None
            record['decision_available']=True
        with self.lock:
            self.pending[frame_idx]=(time.monotonic(),bytes(jpeg_bytes),record)
            while len(self.pending)>60:self.pending.popitem(last=False)
        if not self.wait_for_phone:self.publisher.submit(jpeg_bytes,record)
        return frame_idx
    def remember_bgr(self,annotated_bgr,response,frame_idx,judge_row=None):
        import cv2
        ok,jpeg=cv2.imencode('.jpg',annotated_bgr,[cv2.IMWRITE_JPEG_QUALITY,75])
        if not ok:return None
        return self.remember(jpeg.tobytes(),response,frame_idx,judge_row)
    def report_alarm(self,frame_idx,alarm_record):
        """Call with the phone's actual decision/command for this server frame, not a recomputed rule."""
        if not self.wait_for_phone:return False
        with self.lock:cached=self.pending.pop(frame_idx,None)
        if not cached:return False
        received,jpeg,record=cached
        if time.monotonic()-received>3:return False
        if alarm_record.get('level') is None:return False
        allowed=('dist','ttc','level','level_name','direction','motor','vib_count','vibrate_ms','target_label','target_box')
        record.update({k:alarm_record[k] for k in allowed if k in alarm_record});record['decision_available']=True
        return self.publisher.submit(jpeg,record)
    def close(self):self.publisher.close()
