import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from scipy.io import loadmat
from scipy.stats import pearsonr, spearmanr

from src.depth.midas import load_midas, predict_depth




INDICES = [
    57, 71, 965,
    2020, 2486, 2642,
    3145, 3499, 4502,
    4854, 4926, 5059,
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RGB_DIR = PROJECT_ROOT / "data/raw/sunrgbd"
DEPTH_DIR = PROJECT_ROOT / "data/raw/sunrgbd_train_depth"
GT_DIR = PROJECT_ROOT / "data/raw/sunrgbd_labels"
META_PATH = (
    PROJECT_ROOT
    / "SUNRGBDtoolbox/SUNRGBDtoolbox/Metadata/SUNRGBDMeta.mat"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs/validation/midas_feasibility"
DEPTH_OUTPUT_DIR = OUTPUT_DIR / "relative_depth"
RESULTS_PATH = OUTPUT_DIR / "midas_metric_depth_comparison.csv"


def convert_depth(raw_depth: np.ndarray) -> np.ndarray:
    """Convert SUN RGB-D encoded depth to metres."""
    depth_uint16 = np.array(raw_depth, dtype=np.uint16)

    depth = (depth_uint16 >> 3) | (depth_uint16 << 13)
    depth = depth.astype(np.float32) / 1000.0
    depth[depth > 8.0] = 8.0

    return depth


def load_metadata():
    """Load SUN RGB-D metadata."""
    mat = loadmat(META_PATH)
    return mat["SUNRGBDMeta"][0]


def fit_inverse_depth(x: np.ndarray, y: np.ndarray):
    """
    Fit metric depth using:

        y ~= a / x

    where x is the MiDaS relative-depth output.
    """
    valid = np.abs(x) > 1e-8

    x_valid = x[valid]
    y_valid = y[valid]

    feature = 1.0 / x_valid

    denominator = np.sum(feature * feature)

    if denominator <= 0:
        return np.nan

    a = np.sum(feature * y_valid) / denominator

    return float(a)


def fit_inverse_affine(x: np.ndarray, y: np.ndarray):
    """
    Fit metric depth using:

        y ~= 1 / (a*x + b)

    This provides a more flexible inverse-depth model.
    """
    valid = np.abs(y) > 1e-8

    x_valid = x[valid]
    y_valid = y[valid]

    target = 1.0 / y_valid

    A = np.column_stack([x_valid, np.ones_like(x_valid)])

    a, b = np.linalg.lstsq(A, target, rcond=None)[0]

    return float(a), float(b)


def calculate_metrics(predicted: np.ndarray, actual: np.ndarray):
    """Calculate MAE and RMSE."""
    error = predicted - actual

    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))

    return mae, rmse


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEPTH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading SUN RGB-D metadata...")
    meta = load_metadata()

    print("Loading MiDaS...")
    model, transform, device = load_midas()

    rows = []

    for index in INDICES:
        print(f"\nProcessing index {index}...")

        rgb_path = RGB_DIR / f"img-{index:06d}.jpg"
        depth_path = DEPTH_DIR / f"{index}.png"
        gt_path = GT_DIR / f"img-{5050 + index:06d}.png"

        image = cv2.imread(
            str(rgb_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise FileNotFoundError(
                f"Could not load RGB: {rgb_path}"
            )

        raw_depth = cv2.imread(
            str(depth_path),
            cv2.IMREAD_UNCHANGED,
        )

        if raw_depth is None:
            raise FileNotFoundError(
                f"Could not load depth: {depth_path}"
            )

        gt_label = cv2.imread(
            str(gt_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if gt_label is None:
            raise FileNotFoundError(
                f"Could not load GT: {gt_path}"
            )

        metric_depth = convert_depth(raw_depth)
        gt_wall = gt_label == 1

        relative_depth = predict_depth(
            image,
            model,
            transform,
            device,
        ).astype(np.float32)

        np.save(
            DEPTH_OUTPUT_DIR / f"{index:06d}.npy",
            relative_depth,
        )

        record_idx = 5050 + index - 1
        record = meta[record_idx]

        K = record["K"]

        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])

        if relative_depth.shape != metric_depth.shape:
            raise ValueError(
                f"Depth shape mismatch for {index}: "
                f"MiDaS={relative_depth.shape}, "
                f"metric={metric_depth.shape}"
            )

        valid = (
            gt_wall
            & np.isfinite(relative_depth)
            & np.isfinite(metric_depth)
            & (metric_depth > 0.0)
            & (metric_depth <= 8.0)
            & (np.abs(relative_depth) > 1e-8)
        )

        x = relative_depth[valid].astype(np.float64)
        y = metric_depth[valid].astype(np.float64)

        if len(x) < 10:
            print("  Too few valid wall pixels; skipping.")
            continue

        pearson = float(pearsonr(x, y).statistic)
        spearman = float(spearmanr(x, y).statistic)

        # Model 1:
        # Z_metric ~= a / MiDaS
        inverse_a = fit_inverse_depth(x, y)
        inverse_prediction = inverse_a / x
        inverse_mae, inverse_rmse = calculate_metrics(
            inverse_prediction,
            y,
        )

        # Model 2:
        # Z_metric ~= 1 / (a * MiDaS + b)
        affine_a, affine_b = fit_inverse_affine(x, y)

        denominator = affine_a * x + affine_b
        valid_prediction = np.abs(denominator) > 1e-8

        inverse_affine_prediction = np.full_like(
            y,
            np.nan,
        )

        inverse_affine_prediction[valid_prediction] = (
            1.0 / denominator[valid_prediction]
        )

        finite_prediction = np.isfinite(
            inverse_affine_prediction
        )

        affine_mae, affine_rmse = calculate_metrics(
            inverse_affine_prediction[finite_prediction],
            y[finite_prediction],
        )

        rows.append(
            {
                "index": index,
                "sensor_type": str(record["sensorType"][0]),
                "width": image.shape[1],
                "height": image.shape[0],
                "gt_wall_pixels": int(
                    np.count_nonzero(gt_wall)
                ),
                "valid_wall_pixels": int(len(x)),
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "midas_min": float(np.min(x)),
                "midas_median": float(np.median(x)),
                "midas_max": float(np.max(x)),
                "metric_min_m": float(np.min(y)),
                "metric_median_m": float(np.median(y)),
                "metric_max_m": float(np.max(y)),
                "pearson_r": pearson,
                "spearman_r": spearman,
                "inverse_scale_a": inverse_a,
                "inverse_scale_mae_m": inverse_mae,
                "inverse_scale_rmse_m": inverse_rmse,
                "inverse_affine_a": affine_a,
                "inverse_affine_b": affine_b,
                "inverse_affine_mae_m": affine_mae,
                "inverse_affine_rmse_m": affine_rmse,
            }
        )

        print(
            f"  Sensor: {record['sensorType'][0]}"
        )
        print(
            f"  Valid wall pixels: {len(x):,}"
        )
        print(
            f"  Pearson r: {pearson:.4f}"
        )
        print(
            f"  Spearman r: {spearman:.4f}"
        )
        print(
            f"  Inverse scale a: {inverse_a:.6f}"
        )
        print(
            f"  Inverse scale MAE: {inverse_mae:.4f} m"
        )
        print(
            f"  Inverse-affine a: {affine_a:.6f}"
        )
        print(
            f"  Inverse-affine b: {affine_b:.6f}"
        )
        print(
            f"  Inverse-affine MAE: {affine_mae:.4f} m"
        )

    if not rows:
        raise RuntimeError(
            "No valid experiment results were produced."
        )

    import csv

    fieldnames = list(rows[0].keys())

    with RESULTS_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nExperiment complete.")
    print(f"Results: {RESULTS_PATH}")
    print(
        f"MiDaS depth maps: {DEPTH_OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
