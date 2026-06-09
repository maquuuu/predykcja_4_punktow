from pathlib import Path
import csv
import random

from PIL import Image


SOURCE_ROOT = Path("D:/")
OUT_ROOT = Path("C:/tesla_footprint_crops")


PAD_X = 0.30
PAD_TOP = 0.20
PAD_BOTTOM = 0.45


TRAIN_CROPS_PER_OBJECT = 4
VAL_CROPS_PER_OBJECT = 1


JITTER_CENTER = 0.06
JITTER_SCALE = 0.12

MIN_CROP_W = 32
MIN_CROP_H = 32


def read_label_file(path):
    if not path.exists():
        return []

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            rows.append(parts)
    return rows


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def make_one_crop(
    img,
    image_name,
    split,
    obj_idx,
    aug_idx,
    xc,
    yc,
    bw,
    bh,
    keypoints,
    use_jitter,
):
    img_w, img_h = img.size

    x1 = (xc - bw / 2.0) * img_w
    y1 = (yc - bh / 2.0) * img_h
    x2 = (xc + bw / 2.0) * img_w
    y2 = (yc + bh / 2.0) * img_h

    box_w = x2 - x1
    box_h = y2 - y1

    if box_w < MIN_CROP_W or box_h < MIN_CROP_H:
        return None

    if use_jitter:
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        cx += random.uniform(-JITTER_CENTER, JITTER_CENTER) * box_w
        cy += random.uniform(-JITTER_CENTER, JITTER_CENTER) * box_h

        scale = 1.0 + random.uniform(-JITTER_SCALE, JITTER_SCALE)
        box_w *= scale
        box_h *= scale

        x1 = cx - box_w / 2.0
        x2 = cx + box_w / 2.0
        y1 = cy - box_h / 2.0
        y2 = cy + box_h / 2.0

    crop_x1 = x1 - PAD_X * box_w
    crop_x2 = x2 + PAD_X * box_w
    crop_y1 = y1 - PAD_TOP * box_h
    crop_y2 = y2 + PAD_BOTTOM * box_h

    crop_x1 = clamp(crop_x1, 0, img_w - 1)
    crop_y1 = clamp(crop_y1, 0, img_h - 1)
    crop_x2 = clamp(crop_x2, 1, img_w)
    crop_y2 = clamp(crop_y2, 1, img_h)

    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1

    if crop_w < MIN_CROP_W or crop_h < MIN_CROP_H:
        return None

    target = []

    for px_norm, py_norm, vis in keypoints:

        px = px_norm * img_w
        py = py_norm * img_h

        u = (px - crop_x1) / crop_w
        v = (py - crop_y1) / crop_h


        if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
            return None

        target.extend([u, v])

    crop = img.crop(
        (
            int(round(crop_x1)),
            int(round(crop_y1)),
            int(round(crop_x2)),
            int(round(crop_y2)),
        )
    )

    out_img_dir = OUT_ROOT / "crops" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)

    crop_name = f"{image_name}_obj{obj_idx:02d}_aug{aug_idx:02d}.jpg"
    crop_path = out_img_dir / crop_name
    crop.save(crop_path, quality=95)

    return crop_name, target


def process_split(split):
    images_dir = SOURCE_ROOT / "images" / split
    labels_dir = SOURCE_ROOT / "labels" / split

    out_rows = []
    image_paths = sorted(list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg")))

    crops_per_object = TRAIN_CROPS_PER_OBJECT if split == "train" else VAL_CROPS_PER_OBJECT

    for img_path in image_paths:
        label_path = labels_dir / f"{img_path.stem}.txt"
        rows = read_label_file(label_path)

        if not rows:
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print("Cannot open image:", img_path, e)
            continue

        for obj_idx, parts in enumerate(rows):


            if len(parts) != 17:
                print("Bad label length:", label_path, len(parts))
                continue

            cls_id = int(float(parts[0]))
            xc, yc, bw, bh = map(float, parts[1:5])

            raw_kpts = parts[5:]
            keypoints = []
            ok = True

            for i in range(0, 12, 3):
                x = float(raw_kpts[i])
                y = float(raw_kpts[i + 1])
                vis = int(float(raw_kpts[i + 2]))


                if vis == 0:
                    ok = False
                    break

                keypoints.append((x, y, vis))

            if not ok:
                continue

            for aug_idx in range(crops_per_object):
                use_jitter = split == "train" and aug_idx > 0

                result = make_one_crop(
                    img=img,
                    image_name=img_path.stem,
                    split=split,
                    obj_idx=obj_idx,
                    aug_idx=aug_idx,
                    xc=xc,
                    yc=yc,
                    bw=bw,
                    bh=bh,
                    keypoints=keypoints,
                    use_jitter=use_jitter,
                )

                if result is None:
                    continue

                crop_name, target = result
                out_rows.append([crop_name] + [f"{v:.8f}" for v in target])

    csv_path = OUT_ROOT / f"labels_{split}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "image",
                "x1",
                "y1",
                "x2",
                "y2",
                "x3",
                "y3",
                "x4",
                "y4",
            ]
        )
        writer.writerows(out_rows)

    print(f"{split}: saved {len(out_rows)} crop labels to {csv_path}")


def main():
    random.seed(123)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    process_split("train")
    process_split("val")


if __name__ == "__main__":
    main()
