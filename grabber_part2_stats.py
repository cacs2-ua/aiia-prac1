# grabber.py
# Capture frames from a webcam (device index like 0) OR a network stream (HLS .m3u8 / MJPEG),
# JPEG-encode them, and POST to the VM endpoint for YOLOv8 counting.
#
# Examples:
#   # Send ONE frame from laptop webcam (device 0)
#   python grabber.py --stream-url 0 --vm-url http://<VM_IP>:7001/upload --samples 1 --camera-id laptop_cam
#
#   # Send 30 frames from an MJPEG URL, one every 5s
#   python grabber.py --stream-url "http://.../mjpg/video.mjpg" --vm-url http://<VM_IP>:7001/upload --samples 30 --interval 5
#
#   # Run continuously from webcam 1
#   python grabber.py --stream-url 1 --vm-url http://<VM_IP>:7001/upload --samples -1 --interval 3

import argparse
import csv
import json
import os
import platform
import sys
import time
from datetime import datetime
from typing import Any, Optional

import cv2
import numpy as np
import requests


def ensure_parent_dir(path: str) -> None:
    """Create parent directory for a file path if it does not exist (no-op if path has no dir)."""
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def init_perf_csv(csv_path: str) -> None:
    """Create perf CSV with header if it does not exist."""
    ensure_parent_dir(csv_path)
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["ts", "e2e_ms", "processing_ms", "bytes_sent", "status"],
            )
            w.writeheader()


def parse_capture_source(s: str) -> Any:
    """
    Interpret --stream-url as either an integer device index (webcam) or a URL/path.
    - "0", "1", etc. -> int device index
    - anything else -> str (URL/file)
    """
    s = s.strip()
    try:
        # int("0") works; int("http://...") raises ValueError -> URL branch
        return int(s)
    except ValueError:
        return s


def open_capture(src: Any) -> cv2.VideoCapture:
    """
    Open a webcam device (int) or a URL/path (str).
    For webcam on Windows, try multiple backends for faster/robust open.
    For URLs, try FFMPEG first, then fallback.
    """
    if isinstance(src, int):
        if platform.system() == "Windows":
            # Try backends that often reduce webcam open latency on Windows
            for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
                cap = cv2.VideoCapture(src, backend)
                if cap.isOpened():
                    return cap
            return cv2.VideoCapture(src)
        else:
            return cv2.VideoCapture(src)
    else:
        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(src)
        return cap


def grab_frame_iter(src: Any, interval_s: float):
    """
    Yield a valid frame every 'interval_s' seconds.
    Reopen and retry if the capture fails.
    The FIRST frame is yielded immediately (no initial delay).
    """
    cap: Optional[cv2.VideoCapture] = None
    last = -interval_s  # ensures the very first frame is yielded immediately
    while True:
        try:
            if cap is None or not cap.isOpened():
                cap = open_capture(src)
                if not cap.isOpened():
                    print(f"[LOCAL] Could not open source: {src}. Retrying in 3s...")
                    time.sleep(3)
                    continue

            ok, frame = cap.read()
            if not ok or frame is None:
                print("[LOCAL] Invalid frame. Reopening in 3s...")
                try:
                    cap.release()
                except Exception:
                    pass
                cap = None
                time.sleep(3)
                continue

            now = time.time()
            if now - last >= interval_s:
                last = now
                yield frame
            else:
                # Small sleep to avoid busy-looping when interval is large
                time.sleep(min(0.05, max(0.0, interval_s - (now - last))))

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[LOCAL] Error: {e}. Retrying in 3s...")
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass
            cap = None
            time.sleep(3)

    if cap is not None:
        cap.release()


def encode_jpeg(frame: np.ndarray, quality: int = 90, max_width: Optional[int] = None) -> bytes:
    """Encode a frame to JPEG. Optionally resize if width exceeds 'max_width'."""
    img = frame
    if max_width and frame.shape[1] > max_width:
        h, w = frame.shape[:2]
        scale = max_width / float(w)
        new_size = (int(w * scale), int(h * scale))
        img = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("Failed to encode JPEG.")
    return buf.tobytes()


def send_to_vm(
    vm_url: str,
    jpg_bytes: bytes,
    camera_id: str,
    token: Optional[str],
    timeout_s: float = 10.0,
):
    """
    POST multipart/form-data to the VM:
      - file field 'frame' (image/jpeg)
      - form field 'meta' (JSON)
      - optional header 'X-API-Key'
    """
    headers = {}
    if token:
        headers["X-API-Key"] = token

    meta = {
        "timestamp_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "camera_id": camera_id,
    }

    files = {"frame": ("frame.jpg", jpg_bytes, "image/jpeg")}
    data = {"meta": json.dumps(meta, ensure_ascii=False)}

    # Use (connect_timeout, read_timeout) for better control
    resp = requests.post(vm_url, headers=headers, files=files, data=data, timeout=(5, timeout_s))
    return resp


def parse_args():
    p = argparse.ArgumentParser(
        description="Capture frames from webcam/stream and send them to a VM for YOLOv8 counting (Task 2.2 → Task 2.3)."
    )
    p.add_argument(
        "--stream-url",
        required=True,
        help="Webcam index (e.g., 0 or 1) OR stream URL (HLS .m3u8 / MJPEG).",
    )
    p.add_argument(
        "--vm-url",
        required=True,
        help="HTTP endpoint on the VM, e.g., http://IP:7001/upload",
    )
    p.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Seconds between samples (default 5.0). First frame is immediate.",
    )
    p.add_argument(
        "--samples",
        type=int,
        default=1,
        help="How many frames to send (default 1). Use -1 for infinite.",
    )
    p.add_argument(
        "--camera-id",
        default="cam01",
        help="Logical camera identifier to include in metadata.",
    )
    p.add_argument(
        "--jpeg-quality",
        type=int,
        default=90,
        help="JPEG quality (1-100).",
    )
    p.add_argument(
        "--max-width",
        type=int,
        default=1280,
        help="Resize if frame width exceeds this value (use 0 to disable).",
    )
    p.add_argument(
        "--token",
        default=None,
        help="Optional API key (sent as header X-API-Key).",
    )
    p.add_argument(
        "--http-timeout",
        type=float,
        default=10.0,
        help="Read timeout (seconds) for the POST request (connect timeout fixed at 5s).",
    )
    p.add_argument(
        "--perf-csv",
        type=str,
        default="perf_client.csv",
        help="Path to CSV where latency entries are appended.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    src = parse_capture_source(args.stream_url)

    print(f"[LOCAL] Opening source: {args.stream_url!r} (interpreted as {src!r})")
    print(f"[LOCAL] Sending frames to: {args.vm_url}")
    if args.samples == 0:
        print("[LOCAL] --samples 0 means no work; exiting.")
        return

    # Prepare latency CSV
    init_perf_csv(args.perf_csv)
    perf_file = open(args.perf_csv, "a", newline="", encoding="utf-8")
    perf_writer = csv.DictWriter(
        perf_file, fieldnames=["ts", "e2e_ms", "processing_ms", "bytes_sent", "status"]
    )

    sent = 0
    try:
        for frame in grab_frame_iter(src, max(0.0, args.interval)):
            jpg = encode_jpeg(
                frame,
                quality=args.jpeg_quality,
                max_width=(args.max_width if args.max_width > 0 else None),
            )

            # Measure end-to-end latency around the HTTP POST
            t0 = time.perf_counter()
            try:
                resp = send_to_vm(args.vm_url, jpg, args.camera_id, args.token, timeout_s=args.http_timeout)
            except requests.Timeout:
                # Log a timeout event without e2e_ms (unknown), use status 408-like marker
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                perf_writer.writerow({
                    "ts": ts,
                    "e2e_ms": "",
                    "processing_ms": "",
                    "bytes_sent": len(jpg),
                    "status": "TIMEOUT",
                })
                perf_file.flush()
                print(f"[LOCAL] ({sent+1}) VM TIMEOUT after {args.http_timeout}s")
                continue
            except requests.RequestException as rexc:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                perf_writer.writerow({
                    "ts": ts,
                    "e2e_ms": "",
                    "processing_ms": "",
                    "bytes_sent": len(jpg),
                    "status": f"REQ_ERR:{type(rexc).__name__}",
                })
                perf_file.flush()
                print(f"[LOCAL] ({sent+1}) VM REQUEST ERROR: {rexc}")
                continue
            t1 = time.perf_counter()

            e2e_ms = (t1 - t0) * 1000.0
            bytes_sent = len(jpg)
            status_code = resp.status_code

            processing_ms = None
            try:
                payload_json = resp.json()
                # server may return {"processing_ms": ..., "total_ms": ..., "count": ...}
                processing_ms = payload_json.get("processing_ms", None)
            except Exception:
                payload_json = None  # not JSON; ignore

            # Append latency row
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            perf_writer.writerow({
                "ts": ts,
                "e2e_ms": round(e2e_ms, 2),
                "processing_ms": (round(float(processing_ms), 2) if processing_ms is not None else ""),
                "bytes_sent": bytes_sent,
                "status": status_code,
            })
            perf_file.flush()

            sent += 1
            if resp.ok:
                if payload_json is None:
                    preview = resp.text[:300]
                else:
                    preview = payload_json
                print(f"[LOCAL] ({sent}) VM OK {status_code} | e2e_ms={e2e_ms:.2f} ms, "
                      f"processing_ms={processing_ms} | payload={preview}")
            else:
                print(f"[LOCAL] ({sent}) VM ERROR {status_code}: {resp.text[:300]} | "
                      f"e2e_ms={e2e_ms:.2f} ms, processing_ms={processing_ms}")

            if args.samples != -1 and sent >= args.samples:
                break

    except KeyboardInterrupt:
        print("[LOCAL] Interrupted by user.")
    except Exception as e:
        print(f"[LOCAL] Fatal error: {e}")
        sys.exit(2)
    finally:
        try:
            perf_file.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
