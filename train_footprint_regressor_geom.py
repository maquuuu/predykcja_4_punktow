from pathlib import Path
import csv
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision.models import resnet34, ResNet34_Weights


DATA_ROOT = Path("C:/tesla_footprint_crops")

IMG_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 120
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4


POINT_LOSS_WEIGHT = 1.0
CONF_LOSS_WEIGHT = 0.5


GEOM_LOSS_WEIGHT = 0.25
AREA_LOSS_WEIGHT = 1.0
SPREAD_LOSS_WEIGHT = 1.0
CENTRAL_LOSS_WEIGHT = 0.8
EDGE_LENGTH_LOSS_WEIGHT = 0.5

MIN_FOOTPRINT_AREA = 0.045
MIN_SPREAD_X = 0.32
MIN_SPREAD_Y = 0.12
CENTRAL_MARGIN = 0.18
MIN_EDGE_LENGTH = 0.06


DEFAULT_FOOTPRINT_CONF_THR = 0.70

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


FLIP_IDX = [1, 0, 3, 2]


class FootprintCropDataset(Dataset):
    def __init__(self, root, split, train=False):
        self.root = Path(root)
        self.split = split
        self.train = train

        csv_path = self.root / f"labels_{split}.csv"
        self.crop_dir = self.root / "crops" / split

        self.rows = []
        self.num_positive = 0
        self.num_negative = 0

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                image_name = row["image"]

                target = [
                    float(row["x1"]),
                    float(row["y1"]),
                    float(row["x2"]),
                    float(row["y2"]),
                    float(row["x3"]),
                    float(row["y3"]),
                    float(row["x4"]),
                    float(row["y4"]),
                ]

 
                has_points = float(row.get("has_points", 1.0))

                if has_points > 0.5:
                    self.num_positive += 1
                else:
                    self.num_negative += 1

                self.rows.append((image_name, target, has_points))

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        image_name, target, has_points = self.rows[idx]
        img_path = self.crop_dir / image_name

        img = Image.open(img_path).convert("RGB")
        target = torch.tensor(target, dtype=torch.float32).view(4, 2)
        has_points = torch.tensor(has_points, dtype=torch.float32)

        if has_points.item() > 0.5:
            target = target.clamp(0.0, 1.0)

        if self.train:
            if random.random() < 0.5:
                img = TF.hflip(img)

                if has_points.item() > 0.5:
                    target[:, 0] = 1.0 - target[:, 0]
                    target = target[FLIP_IDX]

            if random.random() < 0.5:
                brightness = random.uniform(0.80, 1.20)
                contrast = random.uniform(0.80, 1.20)
                saturation = random.uniform(0.80, 1.20)

                img = TF.adjust_brightness(img, brightness)
                img = TF.adjust_contrast(img, contrast)
                img = TF.adjust_saturation(img, saturation)

        img = TF.resize(img, [IMG_SIZE, IMG_SIZE])
        img = TF.to_tensor(img)
        img = self.normalize(img)

        return img, target.view(-1), has_points


def build_model():
    model = resnet34(weights=ResNet34_Weights.DEFAULT)
    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, 9),
    )

    return model


def polygon_area_torch(points):
    """
    points: Tensor [B, 4, 2], normalized [0, 1]
    Returns polygon area in normalized crop coordinates.
    """
    x = points[:, :, 0]
    y = points[:, :, 1]
    return 0.5 * torch.abs(
        (x * torch.roll(y, shifts=-1, dims=1)).sum(dim=1)
        - (y * torch.roll(x, shifts=-1, dims=1)).sum(dim=1)
    )


def geometry_losses(pred_points):
    """
    pred_points: Tensor [B, 8], after sigmoid, normalized [0, 1].

    Penalizes physically implausible footprints:
    - too small polygon area,
    - too small x/y spread,
    - all points strongly inside the crop,
    - collapsed neighboring points.
    """
    if pred_points.numel() == 0:
        zero = pred_points.sum() * 0.0
        return zero, zero, zero, zero, zero

    pts = pred_points.view(-1, 4, 2)
    x = pts[:, :, 0]
    y = pts[:, :, 1]

    area = polygon_area_torch(pts)
    area_loss = F.relu(MIN_FOOTPRINT_AREA - area).mean()

    spread_x = x.max(dim=1).values - x.min(dim=1).values
    spread_y = y.max(dim=1).values - y.min(dim=1).values
    spread_loss = (
        F.relu(MIN_SPREAD_X - spread_x).mean()
        + F.relu(MIN_SPREAD_Y - spread_y).mean()
    )


    d_to_edge = torch.stack([x, 1.0 - x, y, 1.0 - y], dim=-1).amin(dim=-1)
    closest_point_to_edge = d_to_edge.min(dim=1).values
    central_loss = F.relu(closest_point_to_edge - CENTRAL_MARGIN).mean()

    # Kara za zapadnięcie się sąsiednich punktów w jedno miejsce.
    rolled = torch.roll(pts, shifts=-1, dims=1)
    edge_lengths = torch.sqrt(((pts - rolled) ** 2).sum(dim=2) + 1e-9)
    edge_length_loss = F.relu(MIN_EDGE_LENGTH - edge_lengths).mean()

    total_geom_loss = (
        AREA_LOSS_WEIGHT * area_loss
        + SPREAD_LOSS_WEIGHT * spread_loss
        + CENTRAL_LOSS_WEIGHT * central_loss
        + EDGE_LENGTH_LOSS_WEIGHT * edge_length_loss
    )

    return total_geom_loss, area_loss, spread_loss, central_loss, edge_length_loss


@torch.no_grad()
def footprint_plausible_mask_torch(pred_points):
    """
    pred_points: Tensor [B, 8], normalized [0, 1].
    Returns bool Tensor [B] telling if the footprint passes basic geometry checks.
    """
    pts = pred_points.view(-1, 4, 2)
    x = pts[:, :, 0]
    y = pts[:, :, 1]

    area = polygon_area_torch(pts)
    spread_x = x.max(dim=1).values - x.min(dim=1).values
    spread_y = y.max(dim=1).values - y.min(dim=1).values

    d_to_edge = torch.stack([x, 1.0 - x, y, 1.0 - y], dim=-1).amin(dim=-1)
    closest_point_to_edge = d_to_edge.min(dim=1).values

    return (
        (area >= MIN_FOOTPRINT_AREA)
        & (spread_x >= MIN_SPREAD_X)
        & (spread_y >= MIN_SPREAD_Y)
        & (closest_point_to_edge <= CENTRAL_MARGIN)
    )


def footprint_is_plausible(points):
    """
    Inference helper.

    points can be a list/tuple or numpy array with shape [4, 2],
    normalized to crop coordinates [0, 1].

    Example:
        pts = pred_points.reshape(4, 2)
        fp_conf = sigmoid(logit)
        if fp_conf > 0.70 and footprint_is_plausible(pts):
            draw_points()
    """
    import numpy as np

    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    x = pts[:, 0]
    y = pts[:, 1]

    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    spread_x = float(x.max() - x.min())
    spread_y = float(y.max() - y.min())

    d_to_edge = np.minimum.reduce([x, 1.0 - x, y, 1.0 - y])
    closest_point_to_edge = float(d_to_edge.min())

    if area < MIN_FOOTPRINT_AREA:
        return False
    if spread_x < MIN_SPREAD_X:
        return False
    if spread_y < MIN_SPREAD_Y:
        return False
    if closest_point_to_edge > CENTRAL_MARGIN:
        return False

    return True


def corner_error_px(pred_points, target_points, has_points):
    """
    Średni błąd punktu w pikselach cropa 256x256,
    liczony tylko dla pozytywów.
    """
    mask = has_points > 0.5

    if mask.sum().item() == 0:
        return 0.0

    pred_points = pred_points[mask].view(-1, 4, 2)
    target_points = target_points[mask].view(-1, 4, 2)

    err = torch.sqrt(((pred_points - target_points) ** 2).sum(dim=2))
    return err.mean().item() * IMG_SIZE


def compute_loss(preds, targets, has_points, point_criterion, conf_criterion):
    # Sigmoid ogranicza punkty do cropa [0, 1].
    pred_points = torch.sigmoid(preds[:, :8])
    pred_conf_logits = preds[:, 8]

    # Confidence loss: dla wszystkich cropów.
    conf_loss = conf_criterion(pred_conf_logits, has_points)

    # Point + geometry loss: tylko dla cropów pozytywnych.
    positive_mask = has_points > 0.5

    if positive_mask.sum().item() > 0:
        pred_pos = pred_points[positive_mask]
        target_pos = targets[positive_mask].clamp(0.0, 1.0)

        point_loss = point_criterion(pred_pos, target_pos)

        geom_loss, area_loss, spread_loss, central_loss, edge_length_loss = geometry_losses(pred_pos)
    else:
        zero = preds.sum() * 0.0
        point_loss = zero
        geom_loss = zero
        area_loss = zero
        spread_loss = zero
        central_loss = zero
        edge_length_loss = zero

    total_loss = (
        POINT_LOSS_WEIGHT * point_loss
        + CONF_LOSS_WEIGHT * conf_loss
        + GEOM_LOSS_WEIGHT * geom_loss
    )

    losses = {
        "total": total_loss,
        "point": point_loss.detach(),
        "conf": conf_loss.detach(),
        "geom": geom_loss.detach(),
        "area": area_loss.detach(),
        "spread": spread_loss.detach(),
        "central": central_loss.detach(),
        "edge_len": edge_length_loss.detach(),
    }

    return losses, pred_points, pred_conf_logits


def train_one_epoch(model, loader, optimizer, point_criterion, conf_criterion):
    model.train()

    sums = {
        "total": 0.0,
        "point": 0.0,
        "conf": 0.0,
        "geom": 0.0,
        "err": 0.0,
    }
    total_n = 0

    for imgs, targets, has_points in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        has_points = has_points.to(DEVICE, non_blocking=True)

        preds = model(imgs)

        losses, pred_points, _ = compute_loss(
            preds,
            targets,
            has_points,
            point_criterion,
            conf_criterion,
        )

        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        optimizer.step()

        bs = imgs.size(0)
        sums["total"] += losses["total"].item() * bs
        sums["point"] += losses["point"].item() * bs
        sums["conf"] += losses["conf"].item() * bs
        sums["geom"] += losses["geom"].item() * bs
        sums["err"] += corner_error_px(pred_points.detach(), targets, has_points) * bs
        total_n += bs

    return (
        sums["total"] / total_n,
        sums["point"] / total_n,
        sums["conf"] / total_n,
        sums["geom"] / total_n,
        sums["err"] / total_n,
    )


@torch.no_grad()
def validate(model, loader, point_criterion, conf_criterion):
    model.eval()

    sums = {
        "total": 0.0,
        "point": 0.0,
        "conf": 0.0,
        "geom": 0.0,
        "err": 0.0,
    }
    total_n = 0

    correct_conf = 0
    total_conf = 0

    plausible_positive = 0
    total_positive = 0

    for imgs, targets, has_points in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        has_points = has_points.to(DEVICE, non_blocking=True)

        preds = model(imgs)

        losses, pred_points, pred_conf_logits = compute_loss(
            preds,
            targets,
            has_points,
            point_criterion,
            conf_criterion,
        )

        probs = torch.sigmoid(pred_conf_logits)
        pred_valid = probs > 0.5
        gt_valid = has_points > 0.5

        correct_conf += (pred_valid == gt_valid).sum().item()
        total_conf += has_points.numel()

        positive_mask = has_points > 0.5
        if positive_mask.sum().item() > 0:
            plausible = footprint_plausible_mask_torch(pred_points[positive_mask])
            plausible_positive += plausible.sum().item()
            total_positive += positive_mask.sum().item()

        bs = imgs.size(0)
        sums["total"] += losses["total"].item() * bs
        sums["point"] += losses["point"].item() * bs
        sums["conf"] += losses["conf"].item() * bs
        sums["geom"] += losses["geom"].item() * bs
        sums["err"] += corner_error_px(pred_points, targets, has_points) * bs
        total_n += bs

    conf_acc = correct_conf / max(total_conf, 1)
    plausible_rate = plausible_positive / max(total_positive, 1)

    return (
        sums["total"] / total_n,
        sums["point"] / total_n,
        sums["conf"] / total_n,
        sums["geom"] / total_n,
        sums["err"] / total_n,
        conf_acc,
        plausible_rate,
    )


def main():
    train_ds = FootprintCropDataset(DATA_ROOT, "train", train=True)
    val_ds = FootprintCropDataset(DATA_ROOT, "val", train=False)

    print("Train crops:", len(train_ds), f"(pos={train_ds.num_positive}, neg={train_ds.num_negative})")
    print("Val crops:", len(val_ds), f"(pos={val_ds.num_positive}, neg={val_ds.num_negative})")

    if train_ds.num_negative == 0 or val_ds.num_negative == 0:
        print(
            "WARNING: Brak negatywnych cropów has_points=0. "
            "Confidence head będzie uczył się prawie zawsze fp=1.00. "
            "Dodaj cropy aut uciętych/niepełnych z has_points=0."
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = build_model().to(DEVICE)

    point_criterion = nn.SmoothL1Loss(beta=0.03)
    conf_criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
    )

    out_dir = Path("runs_footprint_regressor")
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_err = math.inf

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_point_loss, train_conf_loss, train_geom_loss, train_err = train_one_epoch(
            model,
            train_loader,
            optimizer,
            point_criterion,
            conf_criterion,
        )

        (
            val_loss,
            val_point_loss,
            val_conf_loss,
            val_geom_loss,
            val_err,
            val_conf_acc,
            val_plausible_rate,
        ) = validate(
            model,
            val_loader,
            point_criterion,
            conf_criterion,
        )

        scheduler.step()

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"train_loss={train_loss:.5f} "
            f"train_point={train_point_loss:.5f} "
            f"train_conf={train_conf_loss:.5f} "
            f"train_geom={train_geom_loss:.5f} "
            f"train_err={train_err:.2f}px | "
            f"val_loss={val_loss:.5f} "
            f"val_point={val_point_loss:.5f} "
            f"val_conf={val_conf_loss:.5f} "
            f"val_geom={val_geom_loss:.5f} "
            f"val_err={val_err:.2f}px "
            f"val_conf_acc={val_conf_acc:.3f} "
            f"val_plausible={val_plausible_rate:.3f}"
        )


        if val_err < best_val_err:
            best_val_err = val_err
            ckpt = {
                "model": model.state_dict(),
                "img_size": IMG_SIZE,
                "flip_idx": FLIP_IDX,
                "val_err_px": best_val_err,
                "has_conf_head": True,
                "uses_sigmoid_points": True,
                "default_footprint_conf_thr": DEFAULT_FOOTPRINT_CONF_THR,
                "geometry_thresholds": {
                    "min_footprint_area": MIN_FOOTPRINT_AREA,
                    "min_spread_x": MIN_SPREAD_X,
                    "min_spread_y": MIN_SPREAD_Y,
                    "central_margin": CENTRAL_MARGIN,
                    "min_edge_length": MIN_EDGE_LENGTH,
                },
                "loss_weights": {
                    "point": POINT_LOSS_WEIGHT,
                    "conf": CONF_LOSS_WEIGHT,
                    "geom": GEOM_LOSS_WEIGHT,
                    "area": AREA_LOSS_WEIGHT,
                    "spread": SPREAD_LOSS_WEIGHT,
                    "central": CENTRAL_LOSS_WEIGHT,
                    "edge_length": EDGE_LENGTH_LOSS_WEIGHT,
                },
            }
            torch.save(ckpt, out_dir / "best.pt")
            print(f"Saved best.pt, val_err={best_val_err:.2f}px")

    print("Done. Best val corner error:", best_val_err)


if __name__ == "__main__":
    main()
