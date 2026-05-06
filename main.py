import cv2
import numpy as np
import os
import time
import json
import paho.mqtt.client as mqtt
from dotenv import load_dotenv
from flask import Flask, Response
import logging

load_dotenv()
RTSP_URL = os.getenv("CAM2_RTSP_URL")
CAM_NAME = os.getenv("CAM2_NAME", "Brama AI")
CONFIG_FILE = "config.json"

MQTT_BROKER = os.getenv("MQTT_BROKER")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_USER = os.getenv("MQTT_USER")
MQTT_PASS = os.getenv("MQTT_PASSWORD")
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "dom/brama/status")

app = Flask(__name__)

last_status = None
status_buffer = []
BUFFER_SIZE = 15

try:
    mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
except AttributeError:
    mqtt_client = mqtt.Client()

if MQTT_USER and MQTT_PASS:
    mqtt_client.username_pw_set(MQTT_USER, MQTT_PASS)

try:
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client.loop_start()
    print("✅ Połączono z MQTT!", flush=True)
except Exception as e:
    print(f"❌ Błąd MQTT: {e}", flush=True)

def send_mqtt_update(status):
    payload = "ON" if status == "OTWARTA" else "OFF"
    try:
        mqtt_client.publish(MQTT_TOPIC, payload, retain=True)
        print(f"📡 Wysłano do HA: {payload}", flush=True)
    except Exception as e:
        pass

def load_config():
    try:
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    except:
        return {"ROI_X": 100, "ROI_Y": 100, "ROI_W": 200, "ROI_H": 200, "THRESHOLD": 1000}

def get_error_frame(text="BRAK SYGNALU"):
    frame = np.zeros((480, 640, 3), np.uint8)
    cv2.putText(frame, text, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    ret, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()


def build_roi_mask(conf, w_frame, h_frame):
    if "ROI_POINTS" in conf and isinstance(conf["ROI_POINTS"], list) and len(conf["ROI_POINTS"]) >= 3:
        pts = np.array(conf["ROI_POINTS"], dtype=np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, w_frame - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h_frame - 1)
        mask = np.zeros((h_frame, w_frame), np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        return mask, pts, True

    rx = int(conf.get("ROI_X", 0))
    ry = int(conf.get("ROI_Y", 0))
    rw = int(conf.get("ROI_W", w_frame))
    rh = int(conf.get("ROI_H", h_frame))
    rx = max(0, min(rx, w_frame - 1))
    ry = max(0, min(ry, h_frame - 1))
    rw = min(rw, w_frame - rx)
    rh = min(rh, h_frame - ry)
    mask = np.zeros((h_frame, w_frame), np.uint8)
    if rw > 0 and rh > 0:
        mask[ry:ry+rh, rx:rx+rw] = 255
    pts = np.array([[rx, ry], [rx+rw, ry], [rx+rw, ry+rh], [rx, ry+rh]], dtype=np.int32)
    return mask, pts, False

camera = cv2.VideoCapture(RTSP_URL)

def generate_frames():
    global last_status, camera, status_buffer
    frame_count = 0
    
    while True:
        if not camera.isOpened():
            try:
                camera.release()
            except:
                pass
            camera = cv2.VideoCapture(RTSP_URL)
            time.sleep(1)

        conf = load_config()
        thresh = conf.get("THRESHOLD", 1000)

        success, frame = camera.read()
        
        if not success or frame is None:
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + get_error_frame() + b'\r\n')
            time.sleep(2)
            camera.release()
            camera = cv2.VideoCapture(RTSP_URL)
            continue

        frame = cv2.resize(frame, (640, 480))

        try:
            h_frame, w_frame = frame.shape[:2]
            mask, roi_pts, is_polygon = build_roi_mask(conf, w_frame, h_frame)
            if cv2.countNonZero(mask) == 0:
                raise ValueError("ROI maska jest pusta")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            enhanced = cv2.equalizeHist(gray)
            blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
            masked = cv2.bitwise_and(blurred, blurred, mask=mask)
            edges = cv2.Canny(masked, 15, 50)
            edges = cv2.bitwise_and(edges, mask)
            edge_count = int(np.sum(edges > 0))

            current_raw = "OTWARTA" if edge_count < thresh else "ZAMKNIETA"
            
            status_buffer.append(current_raw)
            if len(status_buffer) > BUFFER_SIZE:
                status_buffer.pop(0)

            if all(s == "OTWARTA" for s in status_buffer):
                confirmed_status = "OTWARTA"
            elif all(s == "ZAMKNIETA" for s in status_buffer):
                confirmed_status = "ZAMKNIETA"
            else:
                confirmed_status = last_status if last_status is not None else current_raw

            if confirmed_status != last_status and confirmed_status is not None:
                send_mqtt_update(confirmed_status)
                last_status = confirmed_status

            # --- LOGOWANIE DO KONSOLI CO OK. 1 SEKUNDE ---
            frame_count += 1
            if frame_count % 25 == 0:
                timestamp = time.strftime("%H:%M:%S")
                print(f"[{timestamp}] AI RAPORT: {confirmed_status} | Edges: {edge_count} | Prog: {thresh}", flush=True)

            color = (0, 0, 255) if confirmed_status == "OTWARTA" else (0, 255, 0)
            roi_x, roi_y, roi_w, roi_h = cv2.boundingRect(roi_pts)
            preview = edges[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            preview_w = min(240, roi_w)
            preview_h = min(160, roi_h)
            if preview_w > 0 and preview_h > 0 and preview.size > 0:
                preview = cv2.resize(preview, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
                preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
                frame[10:10+preview_h, 10:10+preview_w] = preview
                cv2.rectangle(frame, (8, 8), (12+preview_w, 12+preview_h), (255, 255, 255), 1)
                cv2.putText(frame, "ROI preview", (10, 10 + preview_h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            
            if is_polygon:
                cv2.polylines(frame, [roi_pts], True, color, 3)
            else:
                cv2.rectangle(frame, tuple(roi_pts[0]), tuple(roi_pts[2]), color, 3)
            cv2.rectangle(frame, (0, h_frame-80), (w_frame, h_frame), (0,0,0), -1)
            
            ts_display = time.strftime("%H:%M:%S")
            cv2.putText(frame, f"{ts_display} | {confirmed_status}", (10, h_frame-45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"Edges: {edge_count} (Prog: {thresh})", (10, h_frame-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        except Exception as e:
            print(f"❌ Blad przetwarzania: {e}", flush=True)
            pass

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret: continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return f"""
    <html>
        <head><title>{CAM_NAME}</title></head>
        <body style="background:#111; color:white; font-family:sans-serif; text-align:center;">
            <h1>{CAM_NAME} - AI Shield</h1>
            <img src="/video_feed" style="border: 4px solid #333; border-radius:10px; max-width:90%;">
            <p>Status MQTT: <b>{MQTT_TOPIC}</b></p>
        </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    print(f"🚀 Startuje Flask na porcie 5000...", flush=True)
    app.run(host='0.0.0.0', port=5000, debug=False)