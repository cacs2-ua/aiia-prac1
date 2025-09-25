# practica_tarea13_cli.py
# Tarea 1.3 — Detección de personas por frame con YOLOv8, CSV y gráfica (.jpg)
# Uso:
#   python practica_tarea13_cli.py --samples 30
#   python practica_tarea13_cli.py --samples 30 --stream-url "http://.../mjpg/video.mjpg" --interval 5 --plot-out person_count.jpg

import os
import time
import csv
import argparse
from datetime import datetime
from typing import List

import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# ============================
# ======= UTILIDADES =========
# ============================

def ensure_csv_with_header(csv_path: str) -> None:
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["Timestamp", "Person_Count"])
            writer.writeheader()

def append_csv(csv_path: str, ts: str, count: int) -> None:
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Timestamp", "Person_Count"])
        writer.writerow({"Timestamp": ts, "Person_Count": count})

def open_capture(url: str) -> cv2.VideoCapture:
    # Intento con FFmpeg (para m3u8) y fallback sin flag
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el stream: {url}")
    return cap

def draw_boxes(frame: np.ndarray, boxes_xyxy: List[List[int]], confs: List[float]) -> np.ndarray:
    for (x1, y1, x2, y2), score in zip(boxes_xyxy, confs):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"person {score:.2f}"
        cv2.putText(frame, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return frame

def plot_csv(csv_path: str, out_path: str) -> None:
    """Genera una gráfica JPG de Person_Count vs. tiempo a partir del CSV."""
    timestamps, counts = [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(row["Timestamp"])
            counts.append(int(row["Person_Count"]))

    if not counts:
        print("CSV vacío: no hay datos para graficar.")
        return

    x = list(range(len(counts)))
    plt.figure(figsize=(10, 4.5))
    plt.plot(x, counts, marker="o")
    plt.title("Evolución del número de personas detectadas")
    plt.xlabel("Muestras (en el tiempo)")
    plt.ylabel("Personas por frame")
    plt.grid(True, linestyle="--", alpha=0.4)

    # Etiquetas de tiempo espaciadas
    N = max(1, len(timestamps) // 10)
    xticks_positions = list(range(0, len(x), N))
    xticks_labels = [timestamps[i] for i in xticks_positions]
    plt.xticks(xticks_positions, xticks_labels, rotation=45, ha="right")

    plt.tight_layout()
    # Aseguramos extensión JPG
    if not out_path.lower().endswith(".jpg") and not out_path.lower().endswith(".jpeg"):
        out_path += ".jpg"
    plt.savefig(out_path, dpi=150)  # Matplotlib guarda JPG si la extensión es .jpg/.jpeg
    plt.close()
    print(f"Gráfica guardada en: {out_path}")

# ============================
# =========== MAIN ===========
# ============================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Cuenta personas con YOLOv8, guarda CSV y genera gráfica .jpg tras N muestras."
    )
    parser.add_argument("--samples", type=int, required=True,
                        help="Número de muestras a tomar (frames muestreados).")
    parser.add_argument("--stream-url", type=str, default="https://video2archives.earthcam.com/earthcamtv-vod/_definst_/mp4:archives/AbbeyRoadHD1/backup.mp4/playlist.m3u8",
                        help="URL del stream (HLS .m3u8 o MJPEG).")
    parser.add_argument("--interval", type=float, default=5.0,
                        help="Intervalo de muestreo en segundos (p. ej., 5.0).")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="Ruta al modelo YOLOv8 (se descarga si no existe).")
    parser.add_argument("--csv-out", type=str, default="person_counts.csv",
                        help="Ruta del CSV de salida.")
    parser.add_argument("--plot-out", type=str, default="person_count.jpg",
                        help="Ruta del .jpg para la gráfica.")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Umbral de confianza para detección.")
    parser.add_argument("--show", action="store_true",
                        help="Muestra ventana con los frames anotados.")
    parser.add_argument("--save-frame", action="store_true",
                        help="Guarda el último frame anotado como last_annotated_frame.jpg")
    return parser.parse_args()

def main():
    args = parse_args()

    SAMPLES_TARGET = max(1, args.samples)
    STREAM_URL = args.stream_url
    SAMPLE_EVERY_SECONDS = max(0.0, args.interval)
    YOLO_MODEL_PATH = args.model
    CSV_PATH = args.csv_out
    PLOT_PATH = args.plot_out
    CONF_THRESHOLD = float(args.conf)
    SHOW_WINDOW = bool(args.show)
    SAVE_ANNOTATED_FRAME = bool(args.save_frame)
    PERSON_CLASS_ID = 0

    ensure_csv_with_header(CSV_PATH)

    print("Cargando modelo YOLOv8...")
    model = YOLO(YOLO_MODEL_PATH)  # descarga si no existe

    print(f"Abrriendo stream: {STREAM_URL}")
    cap = open_capture(STREAM_URL)
    print("Stream abierto correctamente.")

    last_sample_time = 0.0
    taken = 0

    try:
        while taken < SAMPLES_TARGET:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Frame no válido. Reintentando en 3s...")
                cap.release()
                time.sleep(3)
                cap = open_capture(STREAM_URL)
                continue

            now = time.time()
            if (now - last_sample_time) < SAMPLE_EVERY_SECONDS:
                if SHOW_WINDOW:
                    cv2.imshow("Preview (sin muestrear)", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue

            last_sample_time = now

            # Inferencia solo clase "person"
            results = model.predict(
                source=frame,
                conf=CONF_THRESHOLD,
                classes=[PERSON_CLASS_ID],
                verbose=False
            )
            res = results[0]

            boxes_xyxy, confs = [], []
            if res.boxes is not None and len(res.boxes) > 0:
                for b in res.boxes:
                    cls_id = int(b.cls[0].item()) if b.cls is not None else -1
                    score = float(b.conf[0].item()) if b.conf is not None else 0.0
                    if cls_id == PERSON_CLASS_ID and score >= CONF_THRESHOLD:
                        x1, y1, x2, y2 = b.xyxy[0].int().tolist()
                        boxes_xyxy.append([x1, y1, x2, y2])
                        confs.append(score)

            person_count = len(boxes_xyxy)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            append_csv(CSV_PATH, timestamp, person_count)
            taken += 1
            print(f"[{taken}/{SAMPLES_TARGET}] {timestamp} → Personas: {person_count}")

            annotated = draw_boxes(frame.copy(), boxes_xyxy, confs) if boxes_xyxy else frame
            if SAVE_ANNOTATED_FRAME:
                cv2.imwrite("last_annotated_frame.jpg", annotated)

            if SHOW_WINDOW:
                cv2.imshow("YOLOv8 Person Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("Interrumpido por el usuario.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    # Gráfica final en .jpg
    try:
        plot_csv(CSV_PATH, PLOT_PATH)
    except Exception as e:
        print(f"No se pudo generar la gráfica: {e}")

if __name__ == "__main__":
    main()
