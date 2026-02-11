#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import cv2
import numpy as np


BEFORE_DIR = Path("runs/viz/before")
AFTER_DIR  = Path("runs/viz/after")
OUT_PATH   = Path("runs/viz/compare.mp4")

FPS = 15
SEPARATOR_W = 8
FONT = cv2.FONT_HERSHEY_SIMPLEX


def natural_sort_key(p: Path):
    # 파일명 숫자 기준 정렬 (000001.jpg → 1)
    name = p.stem
    try:
        return int(name)
    except:
        return name


def put_label(img, text):
    x, y = 20, 40
    cv2.putText(img, text, (x, y), FONT, 1.1, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def resize_to_h(img, target_h):
    h, w = img.shape[:2]
    if h == target_h:
        return img
    new_w = int(w * (target_h / h))
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)


def pad_to_w(img, target_w):
    h, w = img.shape[:2]
    if w == target_w:
        return img
    pad = target_w - w
    return cv2.copyMakeBorder(img, 0, 0, 0, pad,
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))


def main():
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    before_imgs = sorted(
        [p for p in BEFORE_DIR.iterdir() if p.suffix.lower() in exts],
        key=natural_sort_key
    )

    if not before_imgs:
        raise FileNotFoundError("No images found in before folder.")

    pairs = []
    for bp in before_imgs:
        ap = AFTER_DIR / bp.name
        if ap.exists():
            pairs.append((bp, ap))
        else:
            print(f"[SKIP] missing after: {bp.name}")

    print(f"Total frames: {len(pairs)}")

    # 기준 프레임
    b0 = cv2.imread(str(pairs[0][0]))
    a0 = cv2.imread(str(pairs[0][1]))

    target_h = min(b0.shape[0], a0.shape[0])
    max_bw, max_aw = 0, 0

    for bp, ap in pairs:
        b = resize_to_h(cv2.imread(str(bp)), target_h)
        a = resize_to_h(cv2.imread(str(ap)), target_h)
        max_bw = max(max_bw, b.shape[1])
        max_aw = max(max_aw, a.shape[1])

    out_w = max_bw + SEPARATOR_W + max_aw
    out_h = target_h

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUT_PATH), fourcc, FPS, (out_w, out_h))

    sep = np.zeros((out_h, SEPARATOR_W, 3), dtype=np.uint8)

    for idx, (bp, ap) in enumerate(pairs):
        b = resize_to_h(cv2.imread(str(bp)), out_h)
        a = resize_to_h(cv2.imread(str(ap)), out_h)

        b = put_label(b, "BEFORE")
        a = put_label(a, "AFTER")

        b = pad_to_w(b, max_bw)
        a = pad_to_w(a, max_aw)

        frame = np.hstack([b, sep, a])
        writer.write(frame)

        if (idx + 1) % 50 == 0:
            print(f"{idx+1}/{len(pairs)} frames written...")

    writer.release()
    print("\n✅ Done.")
    print(f"Saved: {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
