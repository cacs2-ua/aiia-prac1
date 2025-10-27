# frame_uploader.py
import cv2
import time
import os
from datetime import datetime, timezone
from azure.storage.blob import BlobServiceClient

def open_capture(source):
    try:
        source = int(source)
    except ValueError:
        pass
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")
    return cap

def main():
    blob_service_client = BlobServiceClient.from_connection_string("DefaultEndpointsProtocol=https;AccountName=aiiatask3;AccountKey=jbGC+C+gg/9Zb+vBo34Jyu72GIM6dW6lOdQzXFcalwQlYxSG5km4KaJXwaWoNIUCU6apHbgJJIGm+ASt8IHmxA==;EndpointSuffix=core.windows.net")
    container_client = blob_service_client.get_container_client("process")

    cap = open_capture("https://video2archives.earthcam.com/earthcamtv-vod/_definst_/mp4:archives/AbbeyRoadHD1/backup.mp4/playlist.m3u8")
    print("📸 Starting frame capture... Press Ctrl+C to stop.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Could not read frame, retrying...")
                time.sleep(1)
                continue

            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            filename = f"frame_{ts}.jpg"
            cv2.imwrite(filename, frame)

            # Upload to Azure
            blob_client = container_client.get_blob_client(filename)
            with open(filename, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
            print(f"⬆️ Uploaded: {filename}")

            os.remove(filename)
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    finally:
        cap.release()

if __name__ == "__main__":
    main()
