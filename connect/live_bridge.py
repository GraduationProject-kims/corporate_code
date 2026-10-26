"""Forward paired annotated JPEG + inference record; never runs YOLO or changes levels."""
import argparse, asyncio, base64, hmac, json, math, os, ssl, time
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

SITE_ORIGIN = 'https://walk-warning-lab.awareangel1.chatgpt.site'
FIELDS = ('decision_available','trial','frame_idx','t_ms','processed','dist','ttc','level','level_name','direction','motor','vib_count','vibrate_ms','target_label','target_conf','target_box','dets_json','yolo_ms','depth_ms','obstacle_passed','pulse_on','capture_to_phone_ms','decode_ms','upload_ms','response_ms','ble_ms')
MAX_MESSAGE = 4 * 1024 * 1024

def number(value):
    if value is None or value == '': return None
    n = float(value)
    if not math.isfinite(n): raise ValueError('non-finite numeric field')
    return n

def validate_packet(packet):
    if not isinstance(packet, dict) or packet.get('type') != 'frame' or packet.get('protocol') != 1: raise ValueError('frame protocol 1 required')
    record = packet.get('record')
    if not isinstance(record, dict): raise ValueError('record required')
    if str(record.get('processed', 1)).lower() in ('0','false'): return None
    stream_id = packet.get('stream_id')
    if not isinstance(stream_id, str) or not 1 <= len(stream_id) <= 80: raise ValueError('stream_id required')
    for key in ('frame_idx','t_ms','vib_count','vibrate_ms'):
        n = number(record.get(key))
        if n is None or n < 0: raise ValueError(f'{key} must be non-negative')
    if number(record['frame_idx']) % 1: raise ValueError('invalid frame index')
    if record.get('decision_available') is False:
        if record.get('level') not in (None,'') or record.get('motor','') or number(record['vibrate_ms']) or number(record['vib_count']): raise ValueError('perception-only frame cannot contain an alarm command')
    elif number(record.get('level')) not in (0,1,2,3): raise ValueError('invalid level')
    if number(record['vib_count']) % 1 or number(record['vib_count']) > 20: raise ValueError('invalid vib_count')
    if record.get('motor', '') not in ('','left','right','neck'): raise ValueError('invalid motor')
    if not isinstance(record.get('level_name'), str) or len(record['level_name']) > 40: raise ValueError('level_name required')
    for key in ('dist','ttc'):
        n = number(record.get(key))
        if n is not None and n < 0: raise ValueError(f'invalid {key}')
    image = packet.get('image', {})
    if image.get('mime') != 'image/jpeg' or not isinstance(image.get('base64'), str): raise ValueError('paired JPEG required')
    if len(image['base64']) > 3 * 1024 * 1024: raise ValueError('JPEG too large; lower JPEG quality or resolution')
    binary = base64.b64decode(image['base64'], validate=True)
    if not binary.startswith(b'\xff\xd8') or not binary.endswith(b'\xff\xd9'): raise ValueError('invalid JPEG')
    return {'type':'frame','protocol':1,'stream_id':stream_id,'record':{k:record[k] for k in FIELDS if k in record},'image':{'mime':'image/jpeg','base64':image['base64']}}

class Hub:
    def __init__(self, token):
        if not token: raise ValueError('LIVE_TOKEN is required')
        self.token=token; self.viewers={}; self.publisher=None; self.latest=None; self.latest_at=0
    def broadcast(self, packet):
        message=json.dumps(packet,ensure_ascii=False,allow_nan=False)
        for queue in list(self.viewers.values()):
            if queue.full(): queue.get_nowait()
            queue.put_nowait(message)
    async def handler(self, ws):
        role=None
        try:
            if ws.request.path != '/live': await ws.close(1008,'use /live'); return
            auth=json.loads(await asyncio.wait_for(ws.recv(),timeout=10))
            if not isinstance(auth,dict) or auth.get('type')!='auth' or not isinstance(auth.get('token'),str) or not hmac.compare_digest(auth['token'],self.token):
                await ws.close(1008,'authentication failed'); return
            role=auth.get('role')
            if role not in ('viewer','publisher'): await ws.close(1008,'invalid role'); return
            if role=='publisher':
                if self.publisher is not None: await ws.close(1013,'publisher already connected'); return
                self.publisher=ws; self.latest=None
                await ws.send(json.dumps({'type':'ready','protocol':1,'role':role}))
                self.broadcast({'type':'status','state':'waiting','message':'송신 연결됨 · 첫 프레임 대기'})
                stream=None; frame=-1; timestamp=-1
                async for message in ws:
                    packet=validate_packet(json.loads(message))
                    if packet is None: continue
                    r=packet['record']
                    if packet['stream_id']!=stream: stream=packet['stream_id']; frame=-1; timestamp=-1
                    if number(r['frame_idx'])<=frame or number(r['t_ms'])<timestamp: continue
                    frame=number(r['frame_idx']); timestamp=number(r['t_ms'])
                    self.latest=packet; self.latest_at=time.monotonic(); self.broadcast(packet)
            else:
                queue=asyncio.Queue(maxsize=1); self.viewers[ws]=queue
                await ws.send(json.dumps({'type':'ready','protocol':1,'role':role}))
                if self.latest and time.monotonic()-self.latest_at<3: queue.put_nowait(json.dumps(self.latest,ensure_ascii=False))
                else: queue.put_nowait(json.dumps({'type':'status','state':'waiting','message':'영상 송신 대기'}))
                async def writer():
                    while True: await ws.send(await queue.get())
                task=asyncio.create_task(writer())
                try:
                    async for _ in ws: pass
                finally: task.cancel(); await asyncio.gather(task,return_exceptions=True)
        except (ConnectionClosed, asyncio.TimeoutError): pass
        except (ValueError, TypeError, KeyError):
            try: await ws.close(1008,'invalid frame/auth payload')
            except ConnectionClosed: pass
        finally:
            self.viewers.pop(ws,None)
            if self.publisher is ws:
                self.publisher=None; self.latest=None
                self.broadcast({'type':'status','state':'waiting','message':'송신 연결 끊김 · 영상 대기'})

async def main():
    parser=argparse.ArgumentParser();parser.add_argument('--host',default='127.0.0.1');parser.add_argument('--port',type=int,default=8765);parser.add_argument('--certfile');parser.add_argument('--keyfile');args=parser.parse_args()
    token=os.environ.get('LIVE_TOKEN','')
    if not token: raise SystemExit('Set LIVE_TOKEN before starting. See README-live.md.')
    tls=None
    if args.certfile or args.keyfile:
        if not (args.certfile and args.keyfile): raise SystemExit('Both --certfile and --keyfile are required.')
        tls=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);tls.load_cert_chain(args.certfile,args.keyfile)
    origins=[SITE_ORIGIN,None]
    extra=os.environ.get('LIVE_ALLOWED_ORIGIN')
    if extra: origins.append(extra)
    hub=Hub(token)
    async with serve(hub.handler,args.host,args.port,ssl=tls,origins=origins,max_size=MAX_MESSAGE,max_queue=2,compression=None,ping_interval=10,ping_timeout=10):
        print(f'Live bridge listening on {args.host}:{args.port}/live; token is not logged.',flush=True)
        await asyncio.Future()

if __name__=='__main__': asyncio.run(main())
