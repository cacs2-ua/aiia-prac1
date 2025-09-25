# practica_tarea13.py
# Tarea 1.3 - Detección de personas por frame con YOLOv8, CSV y gráfica
# UA - Arquitecturas e Infraestructuras para IA — Práctica 1 (Parte 1)
# Autor: (tu nombre/grupo)

import os
import sys
import time
import csv
from datetime import datetime
from typing import Optional, Tuple, List

import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt


# ============================
# ====== CONFIGURACIÓN =======
# ============================

# URL de la webcam pública (ejemplo HLS de EarthCam)
# Puedes sustituirla por otra URL MJPEG/HLS que cumpla la Tarea 1.1.
STREAM_URL = "https://video2archives.earthcam.com/earthcamtv-vod/_definst_/mp4:archives/AbbeyRoadHD1/backup.mp4/playlist.m3u8"

# Frecuencia de muestreo (segundos) – coherente con Tarea 1.2 (cada 5 s)
SAMPLE_EVERY_SECONDS = 5.0

# Umbral de confianza para detección
CONF_THRESHOLD = 0.5

# Solo clase "person" (id 0 en COCO)
PERSON_CLASS_ID = 0

# Archivo del modelo (se descargará automáticamente si no está)
YOLO_MODEL_PATH = "yolov8n.pt"

# Salidas
CSV_PATH = "person_counts.csv"
PLOT_PATH = "person_count.png"

# Mostrar ventana con boxes (opcional)
SHOW_WINDOW = False  # pon True si quieres una ventana en vivo

# Guardar imagen anotada cada muestreo
SAVE_ANNOTATED_FRAME = True
ANNOTATED_FRAME_PATH = "last_annotated_frame.jpg"


# ============================
# ======= UTILIDADES =========
# ============================

def ensure_csv_with_header(csv_path: str) -> None:
    """Crea CSV con cabecera si no existe."""
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["Timestamp", "Person_Count"])
            writer.writeheader()

def append_csv(csv_path: str, ts: str, count: int) -> None:
    """Añade una fila (timestamp, conteo) al CSV."""
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Timestamp", "Person_Count"])
        writer.writerow({"Timestamp": ts, "Person_Count": count})

def open_capture(url: str) -> cv2.VideoCapture:
    """
    Abre el stream con OpenCV.
    Forzamos CAP_FFMPEG para soportar HLS (m3u8) si el build de OpenCV lo permite.
    """
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        # Reintento sin flag explícito (algunos builds no aceptan el flag)
        cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el stream: {url}")
    return cap

def draw_boxes(frame: np.ndarray, boxes_xyxy: List[List[int]], confs: List[float]) -> np.ndarray:
    """Dibuja bounding boxes y confianzas sobre el frame."""
    for (x1, y1, x2, y2), score in zip(boxes_xyxy, confs):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"person {score:.2f}"
        cv2.putText(frame, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                    1, cv2.LINE_AA)
    return frame

def plot_csv(csv_path: str, out_path: str) -> None:
    """Genera una gráfica (PNG) de Person_Count vs. tiempo a partir del CSV."""
    timestamps = []
    counts = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(row["Timestamp"])
            counts.append(int(row["Person_Count"]))

    if not counts:
        print("CSV vacío: no hay datos para graficar.")
        return

    # Convertimos timestamps a una escala homogénea en x (índice) para simplicidad
    x = list(range(len(counts)))

    plt.figure(figsize=(10, 4.5))
    plt.plot(x, counts, marker="o")
    plt.title("Evolución del número de personas detectadas")
    plt.xlabel("Muestras (en el tiempo)")
    plt.ylabel("Personas por frame")
    plt.grid(True, linestyle="--", alpha=0.4)
    # Etiquetas de tiempo (opcional): mostrar algunas
    # Para no saturar, mostramos cada N etiquetas
    N = max(1, len(timestamps) // 10)
    xticks_positions = list(range(0, len(x), N))
    xticks_labels = [timestamps[i] for i in xticks_positions]
    plt.xticks(xticks_positions, xticks_labels, rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Gráfica guardada en: {out_path}")

# ============================
# =========== MAIN ===========
# ============================

def main():
    # 1) CSV preparado
    ensure_csv_with_header(CSV_PATH)

    # 2) Modelo YOLOv8 (descarga automática si no está)
    print("Cargando modelo YOLOv8...")
    model = YOLO(YOLO_MODEL_PATH)  # "yolov8n.pt" (rápido y ligero)

    # 3) Abrir stream
    print(f"Abrriendo stream: {STREAM_URL}")
    cap = open_capture(STREAM_URL)
    print("Stream abierto correctamente.")

    last_sample_time = 0.0
    collected_counts = 0  # solo para feedback en consola

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Frame no válido. Reintentando en 3s...")
                cap.release()
                time.sleep(3)
                cap = open_capture(STREAM_URL)
                continue

            now = time.time()
            # Muestreo cada SAMPLE_EVERY_SECONDS (≈ Tarea 1.2)
            if (now - last_sample_time) < SAMPLE_EVERY_SECONDS:
                # Si quieres procesar TODOS los frames, comenta esta línea.
                # Esto baja carga de CPU/GPU y se ajusta a la práctica.
                if SHOW_WINDOW:
                    cv2.imshow("Preview (sin muestrear)", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue

            last_sample_time = now

            # 4) Inferencia: SOLO clase persona (id 0) y conf mínima
            results = model.predict(
                source=frame,
                conf=CONF_THRESHOLD,
                classes=[PERSON_CLASS_ID],
                verbose=False
            )
            res = results[0]  # un frame → un Results

            # 5) Extraer boxes y conteo
            person_count = 0
            boxes_xyxy = []
            confs = []
            if res.boxes is not None and len(res.boxes) > 0:
                # Filtramos por si acaso (aunque ya pedimos classes=[0])
                for b in res.boxes:
                    cls_id = int(b.cls[0].item()) if b.cls is not None else -1
                    score = float(b.conf[0].item()) if b.conf is not None else 0.0
                    if cls_id == PERSON_CLASS_ID and score >= CONF_THRESHOLD:
                        x1, y1, x2, y2 = b.xyxy[0].int().tolist()
                        boxes_xyxy.append([x1, y1, x2, y2])
                        confs.append(score)
                person_count = len(boxes_xyxy)

            # 6) Timestamp + CSV
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            append_csv(CSV_PATH, timestamp, person_count)
            collected_counts += 1
            print(f"[{timestamp}] Personas: {person_count}  (muestras={collected_counts})")

            # 7) Dibujo y guardado del frame anotado
            if boxes_xyxy:
                annotated = draw_boxes(frame.copy(), boxes_xyxy, confs)
            else:
                annotated = frame

            if SAVE_ANNOTATED_FRAME:
                cv2.imwrite(ANNOTATED_FRAME_PATH, annotated)

            if SHOW_WINDOW:
                cv2.imshow("YOLOv8 Person Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("Interrumpido por el usuario.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    # 8) Graficar a partir del CSV
    try:
        plot_csv(CSV_PATH, PLOT_PATH)
    except Exception as e:
        print(f"No se pudo generar la gráfica: {e}")

if __name__ == "__main__":
    main()
