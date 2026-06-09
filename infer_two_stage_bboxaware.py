from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision.models import resnet34

from ultralytics import YOLO


CONF = 0.25
DETECTOR_WEIGHTS = "yolo26m.pt"


REGRESSOR_WEIGHTS = "runs_footprint_regressor/best.pt"

SOURCE = "C:/tesla_projekt/test"
OUT_DIR = Path("runs_two_stage_predictions_bboxaware")

IMG_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


PAD_X = 0.30
PAD_TOP = 0.20
PAD_BOTTOM = 0.45


DRAW_REJECTED_BOXES = True


FOOTPRINT_CONF_THRESHOLD = 0.45


MIN_BOX_WIDTH_PX = 25
MIN_BOX_HEIGHT_PX = 25
MAX_BOX_WIDTH_RATIO = 0.75
MAX_BOX_HEIGHT_RATIO = 0.85
MAX_BOX_AREA_RATIO = 0.55
MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 6.0


MIN_FOOTPRINT_BOX_WIDTH_PX = 55
MIN_FOOTPRINT_BOX_HEIGHT_PX = 45
MIN_FOOTPRINT_BOX_AREA_PX = 3200


SKIP_BORDER_BOXES_FOR_FOOTPRINT = True
BOX_BORDER_MARGIN_PX = 5


ENABLE_OCCLUSION_FILTER = True
OCCLUDER_MIN_CONF = 0.25
FAR_BOX_HEIGHT_PX = 70
OCCLUSION_IOA_FAR = 0.15
OCCLUSION_IOA_NEAR = 0.30
OCCLUDER_MIN_BOTTOM_DELTA_RATIO = 0.02
OCCLUDER_MIN_SIZE_RATIO = 0.75


MIN_AREA_OF_DET_BOX = 0.035
MAX_AREA_OF_DET_BOX = 1.80
MIN_SPREAD_X_OF_DET_BOX = 0.18
MIN_SPREAD_Y_OF_DET_BOX = 0.10
CENTRAL_MARGIN_IN_DET_BOX = 0.22
MIN_EDGE_OF_DET_BOX = 0.04
MAX_CENTER_SHIFT_OF_DET_BOX = 0.75


MIN_OPPOSITE_EDGE_COS = -0.20

NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def box_touches_image_border(box_xyxy, img_w, img_h, margin=BOX_BORDER_MARGIN_PX):
    x1, y1, x2, y2 = map(float, box_xyxy)
    return x1 <= margin or y1 <= margin or x2 >= img_w - margin or y2 >= img_h - margin


def is_valid_car_box(box_xyxy, img_w, img_h):
    x1, y1, x2, y2 = map(float, box_xyxy)
    w = x2 - x1
    h = y2 - y1

    if w <= 0 or h <= 0:
        return False, "bad_box"
    if w < MIN_BOX_WIDTH_PX or h < MIN_BOX_HEIGHT_PX:
        return False, "small_box"
    if w > img_w * MAX_BOX_WIDTH_RATIO:
        return False, "wide_box"
    if h > img_h * MAX_BOX_HEIGHT_RATIO:
        return False, "tall_box"
    if (w * h) > (img_w * img_h * MAX_BOX_AREA_RATIO):
        return False, "large_box"

    aspect = w / h
    if aspect < MIN_ASPECT_RATIO or aspect > MAX_ASPECT_RATIO:
        return False, "aspect_box"

    return True, "ok"


def footprint_size_reason(box_xyxy):
    x1, y1, x2, y2 = map(float, box_xyxy)
    w = x2 - x1
    h = y2 - y1

    if (
        w < MIN_FOOTPRINT_BOX_WIDTH_PX
        or h < MIN_FOOTPRINT_BOX_HEIGHT_PX
        or (w * h) < MIN_FOOTPRINT_BOX_AREA_PX
    ):
        return False, "far_box"

    return True, "ok"


def _box_area(box_xyxy):
    x1, y1, x2, y2 = map(float, box_xyxy)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _intersection_area(box_a, box_b):
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def vehicle_occlusion_reason(box_idx, boxes, confs, clss):
    if not ENABLE_OCCLUSION_FILTER:
        return False, "ok"

    box = boxes[box_idx]
    x1, y1, x2, y2 = map(float, box)
    area = max(_box_area(box), 1.0)
    h = max(y2 - y1, 1.0)
    ioa_thr = OCCLUSION_IOA_FAR if h < FAR_BOX_HEIGHT_PX else OCCLUSION_IOA_NEAR

    for other_idx, (other_box, other_conf, other_cls) in enumerate(zip(boxes, confs, clss)):
        if other_idx == box_idx:
            continue
        if int(other_cls) not in {2, 5, 7}:
            continue
        if float(other_conf) < OCCLUDER_MIN_CONF:
            continue

        ox1, oy1, ox2, oy2 = map(float, other_box)
        other_area = max(_box_area(other_box), 1.0)
        other_h = max(oy2 - oy1, 1.0)
        inter = _intersection_area(box, other_box)
        if inter <= 0.0:
            continue

        ioa = inter / area
        bottom_delta = (oy2 - y2) / h
        size_ratio = max(other_area / area, other_h / h)
        looks_closer = (
            bottom_delta >= OCCLUDER_MIN_BOTTOM_DELTA_RATIO
            and size_ratio >= OCCLUDER_MIN_SIZE_RATIO
        )

        if looks_closer and ioa >= ioa_thr:
            return True, f"occluded{ioa:.2f}"

    return False, "ok"


def build_model():
    model = resnet34(weights=None)
    in_features = model.fc.in_features


    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, 9),
    )

    return model


def load_regressor():
    regressor = build_model().to(DEVICE)
    ckpt = torch.load(REGRESSOR_WEIGHTS, map_location=DEVICE)
    regressor.load_state_dict(ckpt["model"])
    regressor.eval()

    print(f"Loaded regressor: {REGRESSOR_WEIGHTS}")
    if isinstance(ckpt, dict):
        print("ckpt val_err_px:", ckpt.get("val_err_px"))
        print("ckpt uses_sigmoid_points:", ckpt.get("uses_sigmoid_points"))
        print("ckpt has_conf_head:", ckpt.get("has_conf_head"))
    return regressor


def make_crop_from_box(img_pil, box_xyxy):


    img_w, img_h = img_pil.size

    x1, y1, x2, y2 = map(float, box_xyxy)
    box_w = x2 - x1
    box_h = y2 - y1

    crop_x1 = x1 - PAD_X * box_w
    crop_x2 = x2 + PAD_X * box_w
    crop_y1 = y1 - PAD_TOP * box_h
    crop_y2 = y2 + PAD_BOTTOM * box_h

    crop_x1 = clamp(crop_x1, 0, img_w - 1)
    crop_y1 = clamp(crop_y1, 0, img_h - 1)
    crop_x2 = clamp(crop_x2, 1, img_w)
    crop_y2 = clamp(crop_y2, 1, img_h)

    ix1 = int(round(crop_x1))
    iy1 = int(round(crop_y1))
    ix2 = int(round(crop_x2))
    iy2 = int(round(crop_y2))

    ix1 = int(clamp(ix1, 0, img_w - 1))
    iy1 = int(clamp(iy1, 0, img_h - 1))
    ix2 = int(clamp(ix2, ix1 + 1, img_w))
    iy2 = int(clamp(iy2, iy1 + 1, img_h))

    crop_w = float(ix2 - ix1)
    crop_h = float(iy2 - iy1)

    det_u1 = (x1 - ix1) / crop_w
    det_v1 = (y1 - iy1) / crop_h
    det_u2 = (x2 - ix1) / crop_w
    det_v2 = (y2 - iy1) / crop_h

    det_box_uv = np.array(
        [
            clamp(det_u1, 0.0, 1.0),
            clamp(det_v1, 0.0, 1.0),
            clamp(det_u2, 0.0, 1.0),
            clamp(det_v2, 0.0, 1.0),
        ],
        dtype=np.float32,
    )

    crop = img_pil.crop((ix1, iy1, ix2, iy2))
    return crop, (float(ix1), float(iy1), float(ix2), float(iy2)), det_box_uv


def preprocess_crop(crop):
    img = TF.resize(crop, [IMG_SIZE, IMG_SIZE])
    img = TF.to_tensor(img)
    img = NORMALIZE(img)
    return img.unsqueeze(0)


def crop_points_to_image(points_uv, crop_box):
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1

    pts = []
    for u, v in points_uv:
        x = crop_x1 + float(u) * crop_w
        y = crop_y1 + float(v) * crop_h
        pts.append((int(round(x)), int(round(y))))
    return pts


def _unflip_points(points_uv):

    pts = np.asarray(points_uv, dtype=np.float32).reshape(4, 2).copy()
    pts[:, 0] = 1.0 - pts[:, 0]
    return pts[[1, 0, 3, 2]]


@torch.no_grad()
def predict_crop_tta(regressor, crop):
    x = preprocess_crop(crop).to(DEVICE)
    pred = regressor(x)[0]

    points = torch.sigmoid(pred[:8]).detach().cpu().numpy().reshape(4, 2)
    conf = torch.sigmoid(pred[8]).item()

    flipped_crop = TF.hflip(crop)
    xf = preprocess_crop(flipped_crop).to(DEVICE)
    pred_f = regressor(xf)[0]

    points_f = torch.sigmoid(pred_f[:8]).detach().cpu().numpy().reshape(4, 2)
    points_f = _unflip_points(points_f)
    conf_f = torch.sigmoid(pred_f[8]).item()

    points_avg = np.clip((points + points_f) * 0.5, 0.0, 1.0)
    conf_avg = (conf + conf_f) * 0.5

    return points_avg, conf_avg


def polygon_area(points):
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _orientation(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(a, b, c, d):
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    return (o1 * o2 < 0) and (o3 * o4 < 0)


def _cosine(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-6 or nb < 1e-6:
        return -1.0
    return float(np.dot(a, b) / (na * nb))


def footprint_plausibility_reason(points_uv, det_box_uv):

    pts = np.asarray(points_uv, dtype=np.float32).reshape(4, 2)
    det = np.asarray(det_box_uv, dtype=np.float32).reshape(4)

    if not np.isfinite(pts).all():
        return False, "nan"
    if (pts < 0.0).any() or (pts > 1.0).any():
        return False, "out"

    du1, dv1, du2, dv2 = det
    det_w = max(float(du2 - du1), 1e-6)
    det_h = max(float(dv2 - dv1), 1e-6)
    det_area = det_w * det_h

    x = pts[:, 0]
    y = pts[:, 1]
    spread_x = float(x.max() - x.min())
    spread_y = float(y.max() - y.min())
    area = polygon_area(pts)

    area_rel = area / det_area
    sx_rel = spread_x / det_w
    sy_rel = spread_y / det_h

    if area_rel < MIN_AREA_OF_DET_BOX:
        return False, f"area{area_rel:.2f}"
    if area_rel > MAX_AREA_OF_DET_BOX:
        return False, f"bigarea{area_rel:.2f}"
    if sx_rel < MIN_SPREAD_X_OF_DET_BOX:
        return False, f"sx{sx_rel:.2f}"
    if sy_rel < MIN_SPREAD_Y_OF_DET_BOX:
        return False, f"sy{sy_rel:.2f}"


    rel_x = (x - du1) / det_w
    rel_y = (y - dv1) / det_h
    inside_det = (rel_x >= 0.0) & (rel_x <= 1.0) & (rel_y >= 0.0) & (rel_y <= 1.0)
    if bool(inside_det.all()):
        d_det = np.minimum.reduce([rel_x, 1.0 - rel_x, rel_y, 1.0 - rel_y])
        if float(d_det.min()) > CENTRAL_MARGIN_IN_DET_BOX:
            return False, "central"


    c = pts.mean(axis=0)
    det_c = np.array([(du1 + du2) * 0.5, (dv1 + dv2) * 0.5], dtype=np.float32)
    shift_x = abs(float(c[0] - det_c[0])) / det_w
    shift_y = abs(float(c[1] - det_c[1])) / det_h
    if shift_x > MAX_CENTER_SHIFT_OF_DET_BOX or shift_y > MAX_CENTER_SHIFT_OF_DET_BOX:
        return False, f"shift{shift_x:.1f},{shift_y:.1f}"


    det_diag = float(np.sqrt(det_w * det_w + det_h * det_h))
    edge_lengths = np.sqrt(((pts - np.roll(pts, shift=-1, axis=0)) ** 2).sum(axis=1))
    if float(edge_lengths.min()) < MIN_EDGE_OF_DET_BOX * det_diag:
        return False, f"edge{edge_lengths.min() / det_diag:.2f}"


    if _segments_intersect(pts[0], pts[1], pts[2], pts[3]):
        return False, "cross"
    if _segments_intersect(pts[1], pts[2], pts[3], pts[0]):
        return False, "cross"


    e01 = pts[1] - pts[0]
    e32 = pts[2] - pts[3]
    e12 = pts[2] - pts[1]
    e03 = pts[3] - pts[0]
    if _cosine(e01, e32) < MIN_OPPOSITE_EDGE_COS:
        return False, "opp01"
    if _cosine(e12, e03) < MIN_OPPOSITE_EDGE_COS:
        return False, "opp12"

    return True, "ok"


@torch.no_grad()
def predict_points(regressor, img_pil, box_xyxy, img_w, img_h):
    if SKIP_BORDER_BOXES_FOR_FOOTPRINT and box_touches_image_border(box_xyxy, img_w, img_h):
        return None, 0.0, "border"

    crop, crop_box, det_box_uv = make_crop_from_box(img_pil, box_xyxy)
    points_uv, footprint_conf = predict_crop_tta(regressor, crop)

    if footprint_conf < FOOTPRINT_CONF_THRESHOLD:
        return None, footprint_conf, "lowfp"

    ok, reason = footprint_plausibility_reason(points_uv, det_box_uv)
    if not ok:
        return None, footprint_conf, reason

    pts = crop_points_to_image(points_uv, crop_box)
    return pts, footprint_conf, "ok"


def draw_prediction(img_bgr, box_xyxy, pts, conf, footprint_conf):
    x1, y1, x2, y2 = map(int, box_xyxy)
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for i, (x, y) in enumerate(pts):
        cv2.circle(img_bgr, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(img_bgr, str(i), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        cv2.line(img_bgr, pts[a], pts[b], (255, 0, 0), 2)

    cv2.putText(img_bgr, f"car {conf:.2f} fp {footprint_conf:.2f}", (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)


def draw_box_only(img_bgr, box_xyxy, conf, reason):
    x1, y1, x2, y2 = map(int, box_xyxy)
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 200, 255), 2)
    cv2.putText(img_bgr, f"car {conf:.2f} {reason}", (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    detector = YOLO(DETECTOR_WEIGHTS)
    regressor = load_regressor()

    results = detector.predict(source=SOURCE, imgsz=1280, conf=CONF, stream=True, verbose=True)

    for result in results:
        img_path = Path(result.path)
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print("Cannot read image:", img_path)
            continue

        img_pil = Image.open(img_path).convert("RGB")

        if result.boxes is None or len(result.boxes) == 0:
            continue

        boxes = result.boxes.xyxy.detach().cpu().numpy()
        confs = result.boxes.conf.detach().cpu().numpy()
        clss = result.boxes.cls.detach().cpu().numpy()

        img_h, img_w = img_bgr.shape[:2]
        drawn_any = False

        for box_idx, (box, conf, cls_id) in enumerate(zip(boxes, confs, clss)):

            if int(cls_id) not in {2, 5, 7}:
                continue

            valid_box, box_reason = is_valid_car_box(box, img_w, img_h)
            if not valid_box:
                if DRAW_REJECTED_BOXES:
                    draw_box_only(img_bgr, box, float(conf), box_reason)
                    drawn_any = True
                continue

            is_occluded, occlusion_reason = vehicle_occlusion_reason(box_idx, boxes, confs, clss)
            if is_occluded:
                if DRAW_REJECTED_BOXES:
                    draw_box_only(img_bgr, box, float(conf), occlusion_reason)
                    drawn_any = True
                continue

            footprint_big_enough, footprint_size_msg = footprint_size_reason(box)
            if not footprint_big_enough:
                if DRAW_REJECTED_BOXES:
                    draw_box_only(img_bgr, box, float(conf), footprint_size_msg)
                    drawn_any = True
                continue

            pts, footprint_conf, reason = predict_points(regressor, img_pil, box, img_w, img_h)
            if pts is None:
                if DRAW_REJECTED_BOXES:
                    draw_box_only(img_bgr, box, float(conf), f"{reason} {footprint_conf:.2f}")
                    drawn_any = True
                continue

            draw_prediction(img_bgr, box, pts, float(conf), footprint_conf)
            drawn_any = True

        if drawn_any:
            out_path = OUT_DIR / f"{img_path.stem}_two_stage.jpg"
            cv2.imwrite(str(out_path), img_bgr)
            print("Saved:", out_path)


if __name__ == "__main__":
    main()
