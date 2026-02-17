from pathlib import Path
import cv2
import numpy as np

BEFORE_DIR = Path("runs/viz/before")
AFTER_DIR  = Path("runs/viz/after")
LAST_DIR   = Path("runs/viz/last")
OUT_DIR    = Path("runs/viz/compare")

SEPARATOR_H = 8
DRAW_LABEL = True
FONT = cv2.FONT_HERSHEY_SIMPLEX

LABELS = ["BEFORE", "AFTER", "LAST"]
DIRS   = [BEFORE_DIR, AFTER_DIR, LAST_DIR]


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


def resize_to_w(img, target_w):
    h, w = img.shape[:2]
    if w == target_w:
        return img
    new_h = int(h * (target_w / w))
    return cv2.resize(img, (target_w, new_h), interpolation=cv2.INTER_AREA)


def pad_to_h(img, target_h):
    h, w = img.shape[:2]
    if h == target_h:
        return img
    pad = target_h - h
    return cv2.copyMakeBorder(img, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    base_imgs = sorted([p for p in BEFORE_DIR.iterdir() if p.suffix.lower() in exts],
                       key=natural_sort_key)

    triples = []
    for bp in base_imgs:
        ap = AFTER_DIR / bp.name
        lp = LAST_DIR / bp.name
        if ap.exists() and lp.exists():
            triples.append((bp, ap, lp))

    if not triples:
        raise RuntimeError("No matching triples found.")

    # 기준 width 계산
    imgs0 = [cv2.imread(str(p)) for p in triples[0]]
    target_w = min(im.shape[1] for im in imgs0)

    max_h = [0, 0, 0]

    # 최대 높이 계산 (세로 정렬 안정화)
    for tp in triples:
        for idx, p in enumerate(tp):
            im = cv2.imread(str(p))
            im = resize_to_w(im, target_w)
            max_h[idx] = max(max_h[idx], im.shape[0])

    sep = np.zeros((SEPARATOR_H, target_w, 3), dtype=np.uint8)

    saved = 0
    for i, tp in enumerate(triples):
        ims = []
        for idx, p in enumerate(tp):
            im = cv2.imread(str(p))
            im = resize_to_w(im, target_w)
            if DRAW_LABEL:
                im = put_label(im, LABELS[idx])
            im = pad_to_h(im, max_h[idx])
            ims.append(im)

        merged = np.vstack([ims[0], sep, ims[1], sep, ims[2]])
        out_path = OUT_DIR / tp[0].name
        cv2.imwrite(str(out_path), merged)

        saved += 1

    print(f"\n✅ Done. Saved {saved} images to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
