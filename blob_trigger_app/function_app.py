import azure.functions as func
import logging
import os
import tempfile
import csv
from datetime import datetime
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient  # 🚀 for uploading
from PIL import Image, ImageDraw  # 🚀 for drawing boxes

app = func.FunctionApp()

VISION_ENDPOINT = "https://aiia-ia.services.ai.azure.com/"
VISION_KEY = "4f4OExuyPq3BK2Zz0dc7QPiMSkpTHWFoQCsqkGPlq0ZTdOSbXXr1JQQJ99BJAC5T7U2XJ3w3AAAAACOGI6zB"
CONF_THRESHOLD = 0.7
CSV_PATH = "person_counts.csv"

# 🚀 placeholders for storage upload
BLOB_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=aiiatask3;AccountKey=jbGC+C+gg/9Zb+vBo34Jyu72GIM6dW6lOdQzXFcalwQlYxSG5km4KaJXwaWoNIUCU6apHbgJJIGm+ASt8IHmxA==;EndpointSuffix=core.windows.net"
OUTPUT_CONTAINER = "$web"

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


# 🚀 Draw bounding boxes for each detected person
def draw_bounding_boxes(image_path: str, people, output_path: str):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for person in people.list:
        if person.confidence and person.confidence >= CONF_THRESHOLD:
            # Azure Vision gives bounding boxes as (x, y, width, height)
            box = person.bounding_box
            left = box.x
            top = box.y
            right = box.x + box.width
            bottom = box.y + box.height

            draw.rectangle([left, top, right, bottom], outline="red", width=4)
            draw.text((left, top - 10), f"{person.confidence:.2f}", fill="red")

    image.save(output_path)
    return output_path


# 🚀 Upload the processed image to Azure Blob Storage
def upload_to_blob(container_name: str, file_path: str, blob_name: str):
    blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(container_name)

    # Create the container if it doesn’t exist
    try:
        container_client.create_container()
    except Exception:
        pass  # container may already exist

    blob_client = container_client.get_blob_client(blob_name)
    with open(file_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)

    logging.info(f"Uploaded processed image to container '{container_name}' as '{blob_name}'")


@app.blob_trigger(
    arg_name="myblob",
    path="process/{name}",
    connection="AzureWebJobsStorage"
)
def ProcessUploadedImage(myblob: func.InputStream):
    logging.info(f"Imagen subida detectada: {myblob.name}")

    ensure_csv_with_header(CSV_PATH)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        temp.write(myblob.read())
        temp_path = temp.name

    try:
        client = ImageAnalysisClient(
            endpoint=VISION_ENDPOINT,
            credential=AzureKeyCredential(VISION_KEY)
        )

        with open(temp_path, "rb") as f:
            image_data = f.read()

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

        # 🚀 Draw bounding boxes and save locally
        if result.people and len(result.people.list) > 0:
            bounds_path = os.path.join(tempfile.gettempdir(), "bounds.jpg")
            draw_bounding_boxes(temp_path, result.people, bounds_path)
            upload_to_blob(OUTPUT_CONTAINER, bounds_path, "bounds.jpg")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        append_csv(CSV_PATH, timestamp, person_count)
        upload_to_blob(OUTPUT_CONTAINER, CSV_PATH, "person_counts.csv")

        logging.info(f"Procesada '{myblob.name}' → {person_count} personas detectadas")

    except Exception as e:
        logging.error(f"Error procesando imagen '{myblob.name}': {e}")
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass
