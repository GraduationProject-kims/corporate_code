"""Optional read-only visualization routes for the existing FastAPI server.

Does not receive phone frames, run inference, or invent alarm telemetry.
"""
import base64
import hmac
import os
import threading
import time
import uuid

from fastapi import Request
from fastapi.responses import JSONResponse, Response

SITE_ORIGIN = 'https://walk-warning-lab.awareangel1.chatgpt.site'


def install_visualization(app, latest_view):
    token = os.environ.get('LIVE_TOKEN', '')
    if not token:
        raise ValueError('Set LIVE_TOKEN before starting the model server.')
    if not token.isascii() or any(ord(c) < 33 or ord(c) > 126 for c in token):
        raise ValueError('Use printable ASCII without spaces for LIVE_TOKEN (letters, digits, symbols).')
    stream_id = str(uuid.uuid4())
    started = time.monotonic()
    lock = threading.Lock()
    state = {'image': None, 'seen': 0.0, 'packet': None, 'index': 0}

    def headers(request):
        return {'Access-Control-Allow-Origin': SITE_ORIGIN,
                'Vary': 'Origin', 'Cache-Control': 'no-store'}

    @app.options('/visualization/frame')
    def preflight(request: Request):
        if request.headers.get('origin') != SITE_ORIGIN:
            return Response(status_code=403)
        return Response(status_code=204, headers={**headers(request),
            'Access-Control-Allow-Methods': 'GET, OPTIONS',
            'Access-Control-Allow-Headers': 'Authorization',
            'Access-Control-Max-Age': '600'})

    @app.get('/visualization/frame')
    def visualization_frame(request: Request):
        origin = request.headers.get('origin')
        if origin and origin != SITE_ORIGIN:
            return Response(status_code=403)
        auth = request.headers.get('authorization', '')
        if not hmac.compare_digest(auth.encode('utf-8'), ('Bearer ' + token).encode('utf-8')):
            return JSONResponse({'error': 'authentication required'}, status_code=401, headers=headers(request))
        with lock:
            image = latest_view.get('annotated')
            now = time.monotonic()
            if image is None:
                return JSONResponse({'state': 'waiting'}, status_code=202, headers=headers(request))
            if image is not state['image']:
                import cv2
                ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if not ok:
                    return Response(status_code=503, headers=headers(request))
                encoded_image = base64.b64encode(encoded.tobytes()).decode('ascii')
                if len(encoded_image) > 3 * 1024 * 1024:
                    return Response(status_code=413, headers=headers(request))
                state['index'] += 1
                state['image'], state['seen'] = image, now
                state['packet'] = {'type': 'frame', 'protocol': 1, 'stream_id': stream_id,
                    'record': {'frame_idx': state['index'], 't_ms': round((now-started)*1000, 3),
                        'processed': 1, 'decision_available': False, 'dist': None, 'ttc': None,
                        'level': None, 'level_name': '', 'direction': '', 'motor': '',
                        'vib_count': 0, 'vibrate_ms': 0},
                    'image': {'mime': 'image/jpeg', 'base64': encoded_image}}
            if now - state['seen'] > 3:
                return JSONResponse({'state': 'waiting'}, status_code=202, headers=headers(request))
            return JSONResponse(state['packet'], headers=headers(request))
