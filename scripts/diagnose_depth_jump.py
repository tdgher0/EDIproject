import sys
from pathlib import Path

import numpy as np
import scipy.io
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.segmentation.inference import predict_wall_mask


THRESHOLDS = [None, 0.25, 0.10, 0.05, 0.025, 0.0125, 0.00625]


def load_metadata(mat_path):
    return scipy.io.loadmat(str(mat_path))["SUNRGBDMeta"][0]


def convert_depth(raw_depth):
    depth_uint16 = np.array(raw_depth, dtype=np.uint16)
    depth = (depth_uint16 >> 3) | (depth_uint16 << 13)
    depth = depth.astype(np.float32) / 1000.0
    depth[depth > 8] = 8.0
    return depth


def count_triangles(mask, depth, threshold):
    wall = mask > 0
    h, w = depth.shape

    candidate = 0
    accepted = 0
    rejected = 0

    for y in range(h - 1):
        for x in range(w - 1):
            p00 = wall[y, x]
            p01 = wall[y, x + 1]
            p10 = wall[y + 1, x]
            p11 = wall[y + 1, x + 1]

            triangles = []

            if p00 and p01 and p10:
                triangles.append((
                    depth[y, x],
                    depth[y, x + 1],
                    depth[y + 1, x],
                ))

            if p01 and p10 and p11:
                triangles.append((
                    depth[y, x + 1],
                    depth[y + 1, x],
                    depth[y + 1, x + 1],
                ))

            for z_values in triangles:
                candidate += 1
                depth_jump = max(z_values) - min(z_values)

                if threshold is None or depth_jump <= threshold:
                    accepted += 1
                else:
                    rejected += 1

    return candidate, accepted, rejected


def main():
    project_root = Path(__file__).resolve().parent.parent

    depth_dir = project_root / "data/raw/sunrgbd_train_depth"
    rgb_dir = project_root / "data/raw/sunrgbd"
    gt_dir = project_root / "data/raw/sunrgbd_labels"
    mat_path = project_root / "SUNRGBDtoolbox/SUNRGBDtoolbox/Metadata/SUNRGBDMeta.mat"
    checkpoint_path = project_root / "models/checkpoints/best_model.pt"
    indices_path = project_root / "outputs/validation/validation_indices_100.txt"

    indices = [int(x) for x in indices_path.read_text().split()]
    meta = load_metadata(mat_path)

    totals = {
        threshold: {
            "gt_candidate": 0,
            "gt_accepted": 0,
            "pred_candidate": 0,
            "pred_accepted": 0,
        }
        for threshold in THRESHOLDS
    }

    for index in indices:
        rgb_path = rgb_dir / f"img-{index:06d}.jpg"
        depth_path = depth_dir / f"{index}.png"
        gt_path = gt_dir / f"img-{5050 + index:06d}.png"

        with Image.open(depth_path) as depth_img:
            depth = convert_depth(depth_img)

        with Image.open(gt_path) as gt_img:
            gt_mask = np.array(gt_img) == 1

        pred_mask = predict_wall_mask(str(rgb_path), str(checkpoint_path))

        for threshold in THRESHOLDS:
            gt_candidate, gt_accepted, _ = count_triangles(
                gt_mask, depth, threshold
            )
            pred_candidate, pred_accepted, _ = count_triangles(
                pred_mask, depth, threshold
            )

            totals[threshold]["gt_candidate"] += gt_candidate
            totals[threshold]["gt_accepted"] += gt_accepted
            totals[threshold]["pred_candidate"] += pred_candidate
            totals[threshold]["pred_accepted"] += pred_accepted

        print(f"Processed {index}")

    print("\nDepth-jump triangle diagnostic")
    print("=" * 78)
    print(
        f"{'Threshold':>12} "
        f"{'GT reject %':>12} "
        f"{'Pred reject %':>14} "
        f"{'GT accepted':>14} "
        f"{'Pred accepted':>16}"
    )
    print("-" * 78)

    for threshold in THRESHOLDS:
        data = totals[threshold]

        gt_reject_pct = (
            0.0
            if data["gt_candidate"] == 0
            else (1 - data["gt_accepted"] / data["gt_candidate"]) * 100
        )

        pred_reject_pct = (
            0.0
            if data["pred_candidate"] == 0
            else (1 - data["pred_accepted"] / data["pred_candidate"]) * 100
        )

        label = "None" if threshold is None else f"{threshold:.5f} m"

        print(
            f"{label:>12} "
            f"{gt_reject_pct:12.2f} "
            f"{pred_reject_pct:14.2f} "
            f"{data['gt_accepted']:14,d} "
            f"{data['pred_accepted']:16,d}"
        )


if __name__ == "__main__":
    main()
