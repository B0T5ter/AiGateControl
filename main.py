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
        rx, ry, rw, rh = conf["ROI_X"], conf["ROI_Y"], conf["ROI_W"], conf["ROI_H"]
        thresh = conf["THRESHOLD"]

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
            roi = frame[ry:ry+rh, rx:rx+rw]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            enhanced_roi = cv2.equalizeHist(gray_roi)
            blurred = cv2.GaussianBlur(enhanced_roi, (3, 3), 0)
            edges = cv2.Canny(blurred, 15, 50)
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
            debug_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            dh, dw = debug_edges.shape[:2]
            frame[0:dh, 0:dw] = debug_edges
            
            cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), color, 3)
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