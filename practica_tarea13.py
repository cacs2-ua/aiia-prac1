# practica_tarea13_cli.py
# Task 1.3 — Per-frame people counting with YOLOv8, CSV logging, and time plot (.jpg)
# Usage examples:
#   python practica_tarea13_cli.py --samples 30
#   python practica_tarea13_cli.py --samples 30 --stream-url "http://.../mjpg/video.mjpg" --interval 5 --plot-out person_count.jpg
#   python practica_tarea13_cli.py --samples 200 --stream-url 0 --interval 0 --show  # webcam 0, process EVERY frame

import os
import time
import csv
import argparse
from datetime import datetime
from typing import List, Union

import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt


# ============================
# ========= UTILITIES =========
# ============================

def ensure_parent_dir(path: str) -> None:
    """Create parent directory for a file path if it does not exist (no-op if path has no dir)."""
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def ensure_csv_with_header(csv_path: str) -> None:
    """Create CSV with header if it does not exist."""
    ensure_parent_dir(csv_path)
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["Timestamp", "Person_Count"])
            writer.writeheader()

def append_csv(csv_path: str, ts: str, count: int) -> None:
    """Append a single (timestamp, count) row to CSV."""
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Timestamp", "Person_Count"])
        writer.writerow({"Timestamp": ts, "Person_Count": count})

def parse_source(source_str: str) -> Union[int, str]:
    """
    Accept '0', '1', ... as webcam indices. Otherwise return the original string (URL or file path).
    """
    try:
        return int(source_str)
    except (ValueError, TypeError):
        return source_str

def open_capture(src: Union[int, str]) -> cv2.VideoCapture:
    """
    Open camera (int index) or stream (URL/path). Try FFmpeg first for URLs (useful for HLS),
    then fall back to default backend. Attempts to minimize internal buffering.
    """
    if isinstance(src, int):
        cap = cv2.VideoCapture(src)
    else:
        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(src)
    # Best-effort: shrink buffers (silently ignored if unsupported).
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {src}")
    return cap

def draw_boxes(frame: np.ndarray, boxes_xyxy: List[List[int]], confs: List[float]) -> np.ndarray:
    """Draw green boxes + confidences for detected persons."""
    for (x1, y1, x2, y2), score in zip(boxes_xyxy, confs):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"person {score:.2f}"
        cv2.putText(frame, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return frame

def plot_csv(csv_path: str, out_path: str) -> None:
    """
    Generate a JPG plot of Person_Count over samples using the CSV.
    X-axis uses sample index; tick labels show spaced timestamps for readability.
    """
    timestamps, counts = [], []
    if not os.path.exists(csv_path):
        print("CSV file not found: nothing to plot.")
        return

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(row["Timestamp"])
            counts.append(int(row["Person_Count"]))

    if not counts:
        print("Empty CSV: no data to plot.")
        return

    x = list(range(len(counts)))
    plt.figure(figsize=(10, 4.5))
    plt.plot(x, counts, marker="o")
    plt.title("People detected per sampled frame")
    plt.xlabel("Samples (over time)")
    plt.ylabel("Persons per frame")
    plt.grid(True, linestyle="--", alpha=0.4)

    # Space out timestamp tick labels
    N = max(1, len(timestamps) // 10)
    xticks_positions = list(range(0, len(x), N))
    xticks_labels = [timestamps[i] for i in xticks_positions]
    plt.xticks(xticks_positions, xticks_labels, rotation=45, ha="right")

    plt.tight_layout()
    # Ensure JPG extension
    if not out_path.lower().endswith((".jpg", ".jpeg")):
        out_path += ".jpg"

    ensure_parent_dir(out_path)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Plot saved to: {out_path}")


# ============================
# =========== MAIN ===========
# ============================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Count people with YOLOv8, save CSV, and generate a .jpg plot after N samples."
    )
    parser.add_argument("--samples", type=int, required=True,
                        help="Number of samples to take (sampled frames).")
    parser.add_argument(
        "--stream-url",
        type=str,
        default="https://video2archives.earthcam.com/earthcamtv-vod/_definst_/mp4:archives/AbbeyRoadHD1/backup.mp4/playlist.m3u8",
        help="Stream URL/path (HLS .m3u8, MJPEG, local file). Use '0' (or '1', ...) for a webcam index."
    )
    parser.add_argument("--interval", type=float, default=5.0,
                        help="Sampling interval in seconds. Use 0 to process EVERY frame.")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="YOLOv8 model path (auto-downloads if missing).")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="YOLO input size (higher = slower, potentially more accurate).")
    parser.add_argument("--device", type=str, default="",
                        help="Device for YOLO ('' autodetects; use 'cpu' or '0' for GPU 0 if available).")
    parser.add_argument("--csv-out", type=str, default="person_counts.csv",
                        help="Output CSV path.")
    parser.add_argument("--plot-out", type=str, default="person_count.jpg",
                        help="Output .jpg plot path.")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Detection confidence threshold.")
    parser.add_argument("--show", action="store_true",
                        help="Show a window with annotated frames.")
    parser.add_argument("--save-frame", action="store_true",
                        help="Save the last annotated frame as last_annotated_frame.jpg")
    return parser.parse_args()

def main():
    args = parse_args()

    SAMPLES_TARGET = max(1, args.samples)
    SOURCE = parse_source(args.stream_url)
    SAMPLE_EVERY_SECONDS = max(0.0, args.interval)
    YOLO_MODEL_PATH = args.model
    CSV_PATH = args.csv_out
    PLOT_PATH = args.plot_out
    CONF_THRESHOLD = float(args.conf)
    SHOW_WINDOW = bool(args.show)
    SAVE_ANNOTATED_FRAME = bool(args.save_frame)
    PERSON_CLASS_ID = 0  # COCO person class

    ensure_csv_with_header(CSV_PATH)

    print("Loading YOLOv8 model...")
    model = YOLO(YOLO_MODEL_PATH)  # auto-downloads if missing

    print(f"Opening source: {SOURCE}")
    cap = open_capture(SOURCE)
    print("Source opened successfully.")

    # Sample immediately on start (no initial delay).
    last_sample_time = time.time() - SAMPLE_EVERY_SECONDS
    taken = 0
    consecutive_failures = 0

    try:
        while taken < SAMPLES_TARGET:
            ok, frame = cap.read()
            if not ok or frame is None:
                consecutive_failures += 1
                wait_s = min(3 * consecutive_failures, 10)  # simple backoff up to 10s
                print(f"Invalid frame ({consecutive_failures}). Retrying in {wait_s}s...")
                try:
                    cap.release()
                except Exception:
                    pass
                time.sleep(wait_s)
                cap = open_capture(SOURCE)
                continue
            else:
                consecutive_failures = 0

            now = time.time()
            if (now - last_sample_time) < SAMPLE_EVERY_SECONDS:
                if SHOW_WINDOW:
                    cv2.imshow("Preview (not sampled)", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    # Be nice to the CPU when not sampling
                    time.sleep(0.01)
                continue

            # Time to sample
            last_sample_time = now

            # Run inference on the current frame (persons only)
            results = model.predict(
                source=frame,
                conf=CONF_THRESHOLD,
                classes=[PERSON_CLASS_ID],
                imgsz=args.imgsz,
                device=args.device if args.device is not None else "",
                verbose=False
            )
            res = results[0]

            boxes_xyxy, confs = [], []
            if getattr(res, "boxes", None) is not None and len(res.boxes) > 0:
                for b in res.boxes:
                    # Defensive access for different Ultralytics versions
                    cls_id = int(b.cls[0].item()) if getattr(b, "cls", None) is not None else -1
                    score = float(b.conf[0].item()) if getattr(b, "conf", None) is not None else 0.0
                    if cls_id == PERSON_CLASS_ID and score >= CONF_THRESHOLD:
                        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                        boxes_xyxy.append([x1, y1, x2, y2])
                        confs.append(score)

            person_count = len(boxes_xyxy)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            append_csv(CSV_PATH, timestamp, person_count)
            taken += 1
            print(f"[{taken}/{SAMPLES_TARGET}] {timestamp} → Persons: {person_count}")

            annotated = draw_boxes(frame.copy(), boxes_xyxy, confs) if boxes_xyxy else frame
            if SAVE_ANNOTATED_FRAME:
                try:
                    cv2.imwrite("last_annotated_frame.jpg", annotated)
                except Exception as e:
                    print(f"Failed to save annotated frame: {e}")

            if SHOW_WINDOW:
                cv2.imshow("YOLOv8 Person Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        try:
            cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()

    # Generate final .jpg plot
    try:
        plot_csv(CSV_PATH, PLOT_PATH)
    except Exception as e:
        print(f"Could not generate plot: {e}")

if __name__ == "__main__":
    main()
