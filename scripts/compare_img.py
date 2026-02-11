#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import cv2
import numpy as np

BEFORE_DIR = Path("runs/viz/before")
AFTER_DIR  = Path("runs/viz/after")
OUT_DIR    = Path("runs/viz/compare")

SEPARATOR_W = 8
DRAW_LABEL = True
FONT = cv2.FONT_HERSHEY_SIMPLEX


def natural_sort_key(p: Path):
    try:
        return int(p.stem)
    except:
        return p.stem


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
    return cv2.copyMakeBorder(img, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    before_imgs = sorted([p for p in BEFORE_DIR.iterdir() if p.suffix.lower() in exts],
                         key=natural_sort_key)

    if not before_imgs:
        raise FileNotFoundError(f"No images found in {BEFORE_DIR.resolve()}")

    # before 기준으로 after 매칭
    pairs = []
    for bp in before_imgs:
        ap = AFTER_DIR / bp.name
        if ap.exists():
            pairs.append((bp, ap))
        else:
            print(f"[SKIP] missing after: {bp.name}")

    if not pairs:
        raise RuntimeError("No matching pairs found.")

    # 높이 통일 + 최대 너비 계산(프레임 흔들림 방지)
    b0 = cv2.imread(str(pairs[0][0]))
    a0 = cv2.imread(str(pairs[0][1]))
    if b0 is None or a0 is None:
        raise RuntimeError("Failed to read first pair.")

    target_h = min(b0.shape[0], a0.shape[0])
    max_bw, max_aw = 0, 0

    for bp, ap in pairs:
        b = cv2.imread(str(bp))
        a = cv2.imread(str(ap))
        if b is None or a is None:
            print(f"[SKIP] read failed: {bp.name}")
            continue

        b = resize_to_h(b, target_h)
        a = resize_to_h(a, target_h)

        max_bw = max(max_bw, b.shape[1])
        max_aw = max(max_aw, a.shape[1])

    sep = np.zeros((target_h, SEPARATOR_W, 3), dtype=np.uint8)

    saved = 0
    for i, (bp, ap) in enumerate(pairs):
        b = cv2.imread(str(bp))
        a = cv2.imread(str(ap))
        if b is None or a is None:
            print(f"[SKIP] read failed: {bp.name}")
            continue

        b = resize_to_h(b, target_h)
        a = resize_to_h(a, target_h)

        if DRAW_LABEL:
            b = put_label(b, "BEFORE")
            a = put_label(a, "AFTER")

        b = pad_to_w(b, max_bw)
        a = pad_to_w(a, max_aw)

        merged = np.hstack([b, sep, a])
        out_path = OUT_DIR / bp.name
        cv2.imwrite(str(out_path), merged)

        saved += 1
        if (i + 1) % 50 == 0:
            print(f"[SAVE] {i+1}/{len(pairs)} ...")

    print(f"\n✅ Done. Saved {saved} images to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
