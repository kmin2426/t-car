#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import cv2

IMG_DIR = Path("runs/viz/compare")
OUT_MP4 = Path("runs/viz/compare.mp4")
FPS = 10

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def natural_sort_key(p):
    try:
        return int(p.stem)
    except:
        return p.stem

def main():
    imgs = sorted(
        [p for p in IMG_DIR.iterdir() if p.suffix.lower() in EXTS],
        key=natural_sort_key
    )

    if not imgs:
        raise FileNotFoundError(f"No images found in {IMG_DIR.resolve()}")

    first = cv2.imread(str(imgs[0]))
    if first is None:
        raise RuntimeError("Failed to read first image")

    h, w = first.shape[:2]

    OUT_MP4.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(OUT_MP4), fourcc, FPS, (w, h))

    for p in imgs:
        frame = cv2.imread(str(p))
        if frame is None:
            continue

        if frame.shape[0] != h or frame.shape[1] != w:
            frame = cv2.resize(frame, (w, h))

        vw.write(frame)

    vw.release()
    print("✅ Done:", OUT_MP4.resolve())

if __name__ == "__main__":
    main()
