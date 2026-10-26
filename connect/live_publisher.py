"""A non-blocking side output for the existing phone -> model -> phone pipeline."""
import base64, json, os, queue, threading, time, uuid
from websockets.sync.client import connect
from live_bridge import validate_packet, MAX_MESSAGE

class LivePublisher:
    def __init__(self,url=None,token=None,stream_id=None):
        self.url=url or os.environ.get('LIVE_PUBLISH_URL','ws://127.0.0.1:8765/live')
        self.token=token or os.environ.get('LIVE_TOKEN','')
        if not self.token: raise ValueError('LIVE_TOKEN is required')
        self.stream_id=stream_id or uuid.uuid4().hex
        self.pending=queue.Queue(maxsize=1);self.stopped=threading.Event();self.socket=None
        self.connected=False;self.dropped=0;self.last_error=''
        self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start()
    def submit(self,jpeg_bytes,record):
        """Use the SAME JPEG and log record from one inference frame; return immediately."""
        if self.stopped.is_set(): return False
        packet={'type':'frame','protocol':1,'stream_id':self.stream_id,'record':dict(record),'image':{'mime':'image/jpeg','base64':base64.b64encode(bytes(jpeg_bytes)).decode('ascii')}}
        packet=validate_packet(packet)
        if packet is None: return False
        message=json.dumps(packet,ensure_ascii=False,allow_nan=False,default=lambda v: v.item() if hasattr(v,'item') else v.tolist() if hasattr(v,'tolist') else str(v))
        try: self.pending.put_nowait((time.monotonic(),message))
        except queue.Full:
            try: self.pending.get_nowait(); self.dropped+=1
            except queue.Empty: pass
            self.pending.put_nowait((time.monotonic(),message))
        return True
    def _run(self):
        while not self.stopped.is_set():
            try:
                with connect(self.url,proxy=None,compression=None,open_timeout=3,close_timeout=1,max_size=MAX_MESSAGE,ping_interval=10,ping_timeout=10) as ws:
                    self.socket=ws;ws.send(json.dumps({'type':'auth','role':'publisher','token':self.token}))
                    ready=json.loads(ws.recv(timeout=5))
                    if ready.get('type')!='ready': raise ValueError('bridge did not authenticate publisher')
                    self.connected=True;self.last_error=''
                    while not self.stopped.is_set():
                        try: created,message=self.pending.get(timeout=.2)
                        except queue.Empty: continue
                        if time.monotonic()-created>3:
                            self.dropped+=1;continue
                        ws.send(message)
            except Exception as error:
                # Never log the URL, token, frame image or payload.
                self.last_error=type(error).__name__
            finally: self.connected=False;self.socket=None
            self.stopped.wait(1)
    def close(self):
        self.stopped.set()
        if self.socket:
            try: self.socket.close()
            except Exception: pass
        self.thread.join(timeout=4)
