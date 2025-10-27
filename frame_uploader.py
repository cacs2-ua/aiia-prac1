# frame_uploader.py
import cv2
import time
import os
from datetime import datetime, timezone
from azure.storage.blob import BlobServiceClient
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Capture frames and upload to Azure Blob Storage.")
    parser.add_argument("--stream-url", type=str, default="0", help="Camera index or video stream URL")
    parser.add_argument("--interval", type=float, default=0.5, help="Time between captures (seconds)")
    parser.add_argument("--connection-string", type=str, required=True, help="Azure Storage connection string")
    parser.add_argument("--container", type=str, default="incoming-frames", help="Blob container name")
    return parser.parse_args()

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
    args = parse_args()
    blob_service_client = BlobServiceClient.from_connection_string(args.connection_string)
    container_client = blob_service_client.get_container_client(args.container)

    cap = open_capture(args.stream_url)
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
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
