import time
import csv
import os
from datetime import datetime
import requests
import grabber_part2
import cv2
import json
from multiprocessing import Process

# === CONFIGURACIÓN ===
CAMERAS = [
]

VM_URL = "http://4.251.107.204:7001/upload"
JPEG_QUALITY = 85
CSV_FILE = "metrics_client.csv"


# === Inicializa el CSV ===
def init_csv():
    write_header = not os.path.exists(CSV_FILE)
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "timestamp_local", "camera_id", "latency_s",
                "status_code", "ok", "person_count", "proc_ms",
                "sys_cpu_pct", "proc_cpu_pct", "proc_rss_mb",
                "rx_bytes", "tx_bytes"
            ])


# === Función que envía frames y guarda métricas ===
def send_frames(camera_id, stream_url):
    print(f"[PROCESS] {camera_id} iniciado -> {stream_url}")
    cap = grabber_part2.open_capture(stream_url)
    if not cap.isOpened():
        print(f"[{camera_id}] ❌ No se pudo abrir el stream.")
        return

    while True:
        start_t = time.time()

        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"[{camera_id}] ⚠️ Error leyendo frame, reintentando...")
            cap.release()
            time.sleep(3)
            cap = grabber_part2.open_capture(stream_url)
            continue

        jpg = grabber_part2.encode_jpeg(frame, quality=JPEG_QUALITY, max_width=1280)
        meta = {"timestamp_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "camera_id": camera_id}
        files = {"frame": ("frame.jpg", jpg, "image/jpeg")}
        data = {"meta": json.dumps(meta, ensure_ascii=False)}

        latency = None
        metrics = {}
        try:
            t0 = time.time()
            resp = requests.post(VM_URL, files=files, data=data, timeout=(5, 10))
            latency = time.time() - t0
            if resp.ok:
                metrics = resp.json()
                print(f"[{camera_id}] ✅ OK: {metrics.get('person_count')}p ({metrics.get('proc_ms')} ms, {latency:.2f}s)")
            else:
                print(f"[{camera_id}] ❌ VM ERROR {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            print(f"[{camera_id}] ⚠️ Error: {e}")

        # === Guarda resultados en CSV ===
        with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                camera_id,
                round(latency or 0, 3),
                getattr(resp, "status_code", None),
                getattr(resp, "ok", False),
                metrics.get("person_count"),
                metrics.get("proc_ms"),
                metrics.get("sys_cpu_pct"),
                metrics.get("proc_cpu_pct"),
                metrics.get("proc_rss_mb"),
                metrics.get("rx_bytes"),
                metrics.get("tx_bytes"),
            ])

        time.sleep(5)


# === MAIN ===
def main():
    init_csv()
    processes = []
    for cam_id, url in CAMERAS:
        p = Process(target=send_frames, args=(cam_id, url), daemon=True)
        p.start()
        processes.append(p)

    print(f"[MAIN] {len(CAMERAS)} procesos ejecutándose cada 5s. CSV: {CSV_FILE}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[MAIN] Finalizando...")
        for p in processes:
            p.terminate()


if __name__ == "__main__":
    main()
