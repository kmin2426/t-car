#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO

"""
DEVICE 설정:
    0  -> GPU 0번
    1  -> GPU 1번
   -1  -> CPU
"""

DEVICE = 0  
IMG_DIR = "data/images" # input images directory
CONF_THRES = 0.40       # confidence threshold
IMGSZ = 640
TARGET_CLS_ID = 1       # traffic_light


def get_device():
    if DEVICE == -1:
        return "cpu"

    if torch.cuda.is_available():
        return str(DEVICE)

    print("⚠️ CUDA not available. Using CPU.")
    return "cpu"


def draw_boxes(img, boxes_xyxy, confs, thickness=2):
    for (x1, y1, x2, y2), c in zip(boxes_xyxy, confs):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness)

        txt = f"{c:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 6, y1), (0, 255, 0), -1)
        cv2.putText(img, txt, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    return img


def run_inference(weights_path, out_dir):
    device = get_device()

    model = YOLO(weights_path)
    img_dir = Path(IMG_DIR)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    img_paths = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in exts])

    print(f"\nUsing device: {device}")
    print(f"Images: {len(img_paths)}")

    for p in img_paths:
        results = model.predict(
            source=str(p),
            conf=CONF_THRES,
            imgsz=IMGSZ,
            device=device,
            verbose=False
        )

        r = results[0]
        img = cv2.imread(str(p))

        if r.boxes is None or len(r.boxes) == 0:
            cv2.imwrite(str(out_dir / p.name), img)
            continue

        xyxy = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy()

        keep = (cls == TARGET_CLS_ID)
        xyxy_k = xyxy[keep]
        conf_k = conf[keep]

        vis = img.copy()

        if len(xyxy_k) > 0:
            vis = draw_boxes(vis, xyxy_k, conf_k)

        cv2.imwrite(str(out_dir / p.name), vis)

        print(f"{p.name} → {len(xyxy_k)} TL boxes")

    print(f"\nSaved to: {out_dir.resolve()}")


if __name__ == "__main__":

    # before
    run_inference("weights/before.pt", "runs/viz/before")

    # after
    run_inference("weights/last.pt", "runs/viz/last")

    run_inference("weights/after.pt", "runs/viz/after")
