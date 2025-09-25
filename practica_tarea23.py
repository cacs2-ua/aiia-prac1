# grabber.py
# Captura frames de un stream (HLS .m3u8 o MJPEG) a intervalos regulares y los ENVÍA a la VM.
# La VM contará personas con YOLOv8 y guardará (timestamp, conteo) en CSV (Tarea 2.3).

import argparse
import time
from datetime import datetime
import json
import sys

import cv2
import requests
import numpy as np


def open_capture(url: str) -> cv2.VideoCapture:
    """Abre el stream con OpenCV: intenta FFmpeg (m3u8) y fallback sin flag."""
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(url)
    return cap


def grab_frame_iter(stream_url: str, interval_s: float):
    """
    Iterador que entrega un frame válido cada 'interval_s' segundos.
    Reabre y reintenta si el stream falla.
    """
    cap = None
    last = 0.0
    while True:
        try:
            if cap is None or not cap.isOpened():
                cap = open_capture(stream_url)
                if not cap.isOpened():
                    print(f"[LOCAL] No se pudo abrir el stream: {stream_url}. Reintentando en 3s...")
                    time.sleep(3)
                    continue

            ok, frame = cap.read()
            if not ok or frame is None:
                print("[LOCAL] Frame no válido. Reabriendo en 3s...")
                cap.release()
                cap = None
                time.sleep(3)
                continue

            now = time.time()
            if now - last >= interval_s:
                last = now
                yield frame

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[LOCAL] Error: {e}. Reintentando en 3s...")
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass
            cap = None
            time.sleep(3)

    if cap is not None:
        cap.release()


def encode_jpeg(frame: np.ndarray, quality: int = 90, max_width: int | None = None) -> bytes:
    """Codifica frame a JPEG. Opcionalmente redimensiona si excede 'max_width'."""
    img = frame
    if max_width and frame.shape[1] > max_width:
        h, w = frame.shape[:2]
        scale = max_width / float(w)
        new_size = (int(w * scale), int(h * scale))
        img = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Fallo al codificar JPEG.")
    return buf.tobytes()


def send_to_vm(vm_url: str, jpg_bytes: bytes, camera_id: str, token: str | None, timeout_s: float = 15.0):
    """Envía un JPEG a la VM por POST multipart/form-data: campo 'frame', y 'meta' en JSON."""
    headers = {}
    if token:
        headers["X-API-Key"] = token

    meta = {
        "timestamp_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "camera_id": camera_id,
    }
    files = {"frame": ("frame.jpg", jpg_bytes, "image/jpeg")}
    data = {"meta": json.dumps(meta, ensure_ascii=False)}

    resp = requests.post(vm_url, headers=headers, files=files, data=data, timeout=timeout_s)
    return resp


def parse_args():
    p = argparse.ArgumentParser(
        description="Captura frames de webcam y los envía a la VM para conteo con YOLOv8 (Tarea 2.3)."
    )
    p.add_argument("--stream-url", required=True, help="URL del stream (HLS .m3u8 o MJPEG).")
    p.add_argument("--vm-url", required=True, help="Endpoint HTTP en la VM, p.ej. http://IP:7001/upload")
    p.add_argument("--interval", type=float, default=5.0, help="Segundos entre muestras (p.ej. 5.0).")
    p.add_argument("--camera-id", default="cam01", help="Identificador lógico de la cámara.")
    p.add_argument("--jpeg-quality", type=int, default=90, help="Calidad JPEG (1-100).")
    p.add_argument("--max-width", type=int, default=1280, help="Redimensiona si el ancho supera este valor (0=sin cambio).")
    p.add_argument("--token", default=None, help="API key opcional (cabecera X-API-Key).")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"[LOCAL] Abriendo stream: {args.stream_url}")
    print(f"[LOCAL] Enviando frames a: {args.vm_url}")
    sent = 0
    try:
        for frame in grab_frame_iter(args.stream_url, args.interval):
            jpg = encode_jpeg(frame, quality=args.jpeg_quality,
                              max_width=(args.max_width if args.max_width > 0 else None))
            resp = send_to_vm(args.vm_url, jpg, args.camera_id, args.token)
            sent += 1
            if resp.ok:
                try:
                    payload = resp.json()
                except Exception:
                    payload = resp.text[:300]
                print(f"[LOCAL] ({sent}) VM OK {resp.status_code}: {payload}")
            else:
                print(f"[LOCAL] ({sent}) VM ERROR {resp.status_code}: {resp.text[:300]}")
    except KeyboardInterrupt:
        print("[LOCAL] Interrumpido por usuario.")
    except Exception as e:
        print(f"[LOCAL] Error fatal: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
