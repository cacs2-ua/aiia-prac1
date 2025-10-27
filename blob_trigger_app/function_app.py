import azure.functions as func
import logging
import os
import tempfile
import csv
from datetime import datetime
from ultralytics import YOLO
import cv2

app = func.FunctionApp()

# Cargar el modelo YOLO una sola vez (mejor rendimiento)
MODEL_PATH = "yolov8n.pt"
model = YOLO(MODEL_PATH)
CSV_PATH = "person_counts.csv"
PERSON_CLASS_ID = 0  # COCO class for person
CONF_THRESHOLD = 0.5


def ensure_csv_with_header(csv_path: str):
    """Crea CSV con encabezado si no existe."""
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["Timestamp", "Person_Count"])
            writer.writeheader()


def append_csv(csv_path: str, ts: str, count: int):
    """Agrega una fila (timestamp, count) al CSV."""
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Timestamp", "Person_Count"])
        writer.writerow({"Timestamp": ts, "Person_Count": count})


@app.blob_trigger(
    arg_name="myblob",
    path="process/{name}",
    connection="AzureWebJobsStorage"
)
def ProcessUploadedImage(myblob: func.InputStream):
    """Se activa cuando se sube un nuevo blob (imagen)."""
    logging.info(f"📦 Nuevo blob detectado: {myblob.name} ({myblob.length} bytes)")

    # Crear CSV si no existe
    ensure_csv_with_header(CSV_PATH)

    # Guardar el blob temporalmente
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        temp.write(myblob.read())
        temp_path = temp.name

    try:
        # Leer imagen
        frame = cv2.imread(temp_path)
        if frame is None:
            logging.error(f"❌ No se pudo leer la imagen: {temp_path}")
            return

        # Ejecutar YOLO solo para clase 'person'
        results = model.predict(
            source=frame,
            conf=CONF_THRESHOLD,
            classes=[PERSON_CLASS_ID],
            verbose=False
        )

        res = results[0]
        person_count = len(res.boxes) if hasattr(res, "boxes") else 0

        # Guardar resultado en CSV
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        append_csv(CSV_PATH, timestamp, person_count)

        logging.info(f"✅ {timestamp} → Personas detectadas: {person_count}")

    except Exception as e:
        logging.error(f"⚠️ Error procesando imagen: {e}")
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass
