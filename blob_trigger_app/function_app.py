import azure.functions as func
import logging
import os
import tempfile
import csv
from datetime import datetime
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

app = func.FunctionApp()

# --- CONFIGURACIÓN ---
VISION_ENDPOINT = "https://aiia-ia.services.ai.azure.com/"
VISION_KEY = "4f4OExuyPq3BK2Zz0dc7QPiMSkpTHWFoQCsqkGPlq0ZTdOSbXXr1JQQJ99BJAC5T7U2XJ3w3AAAAACOGI6zB"
CONF_THRESHOLD = 0.5
CSV_PATH = "person_counts.csv"

# --- LIMPIAR LOGS VERBOSOS DEL SDK DE AZURE ---
# Solo queremos ver nuestros mensajes, no los del cliente HTTP interno
for noisy_logger in [
    "azure.core.pipeline.policies.http_logging_policy",
    "azure.core.pipeline.policies._universal",
    "azure.core.pipeline.policies._retry",
]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def ensure_csv_with_header(csv_path: str):
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["Timestamp", "Person_Count"])
            writer.writeheader()


def append_csv(csv_path: str, ts: str, count: int):
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

    # Mensaje inicial limpio
    logging.info(f"Imagen subida detectada: {myblob.name}")

    ensure_csv_with_header(CSV_PATH)

    # Guardar el blob temporalmente
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        temp.write(myblob.read())
        temp_path = temp.name

    try:
        # Crear cliente de Azure Computer Vision
        client = ImageAnalysisClient(
            endpoint=VISION_ENDPOINT,
            credential=AzureKeyCredential(VISION_KEY)
        )

        with open(temp_path, "rb") as f:
            image_data = f.read()

        # Analizar personas
        result = client.analyze(
            image_data=image_data,
            visual_features=[VisualFeatures.PEOPLE],
            language="en"
        )

        person_count = 0
        if result.people is not None:
            for person in result.people.list:
                if person.confidence and person.confidence >= CONF_THRESHOLD:
                    person_count += 1

        # Guardar CSV y mostrar resumen
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        append_csv(CSV_PATH, timestamp, person_count)

        logging.info(f"Procesada '{myblob.name}' → {person_count} personas detectadas")

    except Exception as e:
        logging.error(f" Error procesando imagen '{myblob.name}': {e}")
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass
