import time
from datetime import datetime
import requests
import grabber_part2
import json

# === CONFIGURACIÓN ===
CAMERA_ID = "cam01"
STREAM_URL = "https://video2archives.earthcam.com/earthcamtv-vod/_definst_/mp4:archives/AbbeyRoadHD1/backup.mp4/playlist.m3u8"
VM_URL = "http://4.251.107.204:7001/upload"
JPEG_QUALITY = 85
SLEEP_INTERVAL = 5  # segundos entre envíos

# === Función para enviar un frame ===
def send_frame():
    cap = grabber_part2.open_capture(STREAM_URL)
    if not cap.isOpened():
        print(f"[{CAMERA_ID}] ❌ No se pudo abrir el stream.")
        return

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"[{CAMERA_ID}] ⚠️ Error leyendo frame, reintentando...")
            cap.release()
            time.sleep(3)
            cap = grabber_part2.open_capture(STREAM_URL)
            continue

        jpg = grabber_part2.encode_jpeg(frame, quality=JPEG_QUALITY, max_width=1280)
        meta = {
            "timestamp_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "camera_id": CAMERA_ID
        }
        files = {"frame": ("frame.jpg", jpg, "image/jpeg")}
        data = {"meta": json.dumps(meta, ensure_ascii=False)}

        try:
            t0 = time.time()
            resp = requests.post(VM_URL, files=files, data=data, timeout=(5, 5))
            latency = time.time() - t0

            if resp.ok:
                metrics = resp.json()
                print(f"[{CAMERA_ID}] ✅ OK: {metrics.get('person_count')}p | "
                      f"Proc: {metrics.get('proc_ms')} ms | "
                      f"Latencia: {latency:.2f}s")
            else:
                print(f"[{CAMERA_ID}] ❌ VM ERROR {resp.status_code}")
        except Exception as e:
            print(f"[{CAMERA_ID}] ⚠️ Error: {e}")

        # Espera 5 segundos antes de enviar el siguiente frame
        time.sleep(SLEEP_INTERVAL)


if __name__ == "__main__":
    send_frame()
