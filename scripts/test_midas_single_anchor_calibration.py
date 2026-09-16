import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import cv2
import numpy as np
from scipy.io import loadmat

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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs/validation/midas_feasibility"
)
RESULTS_PATH = (
    OUTPUT_DIR
    / "midas_single_anchor_calibration.csv"
)


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


def fit_inverse_affine(
    x: np.ndarray,
    y: np.ndarray,
):
    """
    Fit the oracle model:

        y ~= 1 / (a*x + b)

    using all valid wall pixels.
    """
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (np.abs(y) > 1e-8)
    )

    x_valid = x[valid]
    y_valid = y[valid]

    target = 1.0 / y_valid
    A = np.column_stack(
        [x_valid, np.ones_like(x_valid)]
    )

    a, b = np.linalg.lstsq(
        A,
        target,
        rcond=None,
    )[0]

    return float(a), float(b)


def fit_single_scale(
    midas_values: np.ndarray,
    metric_values: np.ndarray,
) -> float:
    """
    Fit Z ~= a / D using the supplied anchor samples only.
    """
    feature = 1.0 / midas_values

    denominator = np.sum(feature * feature)

    if denominator <= 0:
        raise ValueError(
            "Cannot fit scale: invalid anchor MiDaS values."
        )

    a = np.sum(feature * metric_values) / denominator

    return float(a)


def calculate_metrics(
    predicted: np.ndarray,
    actual: np.ndarray,
):
    """Calculate MAE, RMSE, and median absolute error."""
    error = predicted - actual
    absolute_error = np.abs(error)

    mae = float(np.mean(absolute_error))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    median_abs_error = float(
        np.median(absolute_error)
    )

    return mae, rmse, median_abs_error


def nearest_valid_anchor(
    requested_x: float,
    requested_y: float,
    valid_mask: np.ndarray,
):
    """
    Find the valid pixel nearest to the requested image coordinate.
    """
    ys, xs = np.where(valid_mask)

    if len(xs) == 0:
        return None

    distances_squared = (
        (xs.astype(np.float64) - requested_x) ** 2
        + (ys.astype(np.float64) - requested_y) ** 2
    )

    position = int(np.argmin(distances_squared))

    return int(xs[position]), int(ys[position])


def evaluate_scale(
    scale_a: float,
    midas_values: np.ndarray,
    metric_values: np.ndarray,
):
    """Evaluate Z ~= a / D over all valid wall pixels."""
    prediction = scale_a / midas_values

    valid_prediction = (
        np.isfinite(prediction)
        & np.isfinite(metric_values)
    )

    if not np.any(valid_prediction):
        return np.nan, np.nan, np.nan, 0

    mae, rmse, median_abs_error = calculate_metrics(
        prediction[valid_prediction],
        metric_values[valid_prediction],
    )

    return (
        mae,
        rmse,
        median_abs_error,
        int(np.count_nonzero(valid_prediction)),
    )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

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

        y_pixels, x_pixels = np.where(valid)

        if len(x_pixels) < 10:
            print(
                "  Too few valid wall pixels; skipping."
            )
            continue

        x = relative_depth[valid].astype(np.float64)
        y = metric_depth[valid].astype(np.float64)

        oracle_a, oracle_b = fit_inverse_affine(x, y)

        oracle_denominator = (
            oracle_a * x + oracle_b
        )

        oracle_valid = (
            np.isfinite(oracle_denominator)
            & (np.abs(oracle_denominator) > 1e-8)
        )

        oracle_prediction = np.full_like(
            y,
            np.nan,
        )

        oracle_prediction[oracle_valid] = (
            1.0 / oracle_denominator[oracle_valid]
        )

        oracle_finite = np.isfinite(
            oracle_prediction
        )

        oracle_mae, oracle_rmse, _ = (
            calculate_metrics(
                oracle_prediction[oracle_finite],
                y[oracle_finite],
            )
        )

        height, width = metric_depth.shape

        requested_anchors = [
            (
                "center",
                [(width - 1) * 0.50, (height - 1) * 0.50],
            ),
            (
                "left_25pct",
                [(width - 1) * 0.25, (height - 1) * 0.50],
            ),
            (
                "right_75pct",
                [(width - 1) * 0.75, (height - 1) * 0.50],
            ),
        ]

        anchor_results = {}

        for strategy, coordinates in requested_anchors:
            requested_x, requested_y = coordinates

            anchor = nearest_valid_anchor(
                requested_x,
                requested_y,
                valid,
            )

            if anchor is None:
                anchor_results[strategy] = None
                continue

            anchor_x, anchor_y = anchor

            anchor_results[strategy] = (
                anchor_x,
                anchor_y,
                float(relative_depth[anchor_y, anchor_x]),
                float(metric_depth[anchor_y, anchor_x]),
            )

        for strategy, _ in requested_anchors:
            result = anchor_results[strategy]

            if result is None:
                rows.append(
                    {
                        "index": index,
                        "sensor_type": str(
                            meta[5050 + index - 1]["sensorType"][0]
                        ),
                        "anchor_strategy": strategy,
                        "anchor_count": 0,
                        "anchor_coordinates": "",
                        "valid_wall_pixels": int(len(x)),
                        "anchor_valid_count": 0,
                        "anchor_depth_values_m": "",
                        "anchor_midas_values": "",
                        "scale_a": np.nan,
                        "mae_m": np.nan,
                        "rmse_m": np.nan,
                        "median_abs_error_m": np.nan,
                        "oracle_inverse_affine_mae_m": oracle_mae,
                        "oracle_inverse_affine_rmse_m": oracle_rmse,
                    }
                )
                continue

            anchor_x, anchor_y, anchor_d, anchor_z = result

            scale_a = fit_single_scale(
                np.array([anchor_d]),
                np.array([anchor_z]),
            )

            mae, rmse, median_abs_error, valid_count = (
                evaluate_scale(
                    scale_a,
                    x,
                    y,
                )
            )

            rows.append(
                {
                    "index": index,
                    "sensor_type": str(
                        meta[5050 + index - 1]["sensorType"][0]
                    ),
                    "anchor_strategy": strategy,
                    "anchor_count": 1,
                    "anchor_coordinates": (
                        f"{anchor_x},{anchor_y}"
                    ),
                    "valid_wall_pixels": int(len(x)),
                    "anchor_valid_count": 1,
                    "anchor_depth_values_m": (
                        f"{anchor_z:.6f}"
                    ),
                    "anchor_midas_values": (
                        f"{anchor_d:.6f}"
                    ),
                    "scale_a": scale_a,
                    "mae_m": mae,
                    "rmse_m": rmse,
                    "median_abs_error_m": median_abs_error,
                    "oracle_inverse_affine_mae_m": oracle_mae,
                    "oracle_inverse_affine_rmse_m": oracle_rmse,
                }
            )

        three_anchors = []

        for strategy, _ in requested_anchors:
            result = anchor_results[strategy]

            if result is not None:
                anchor_x, anchor_y, anchor_d, anchor_z = result

                three_anchors.append(
                    (
                        anchor_x,
                        anchor_y,
                        anchor_d,
                        anchor_z,
                    )
                )

        if three_anchors:
            anchor_midas = np.array(
                [item[2] for item in three_anchors],
                dtype=np.float64,
            )
            anchor_depth = np.array(
                [item[3] for item in three_anchors],
                dtype=np.float64,
            )

            scale_a = fit_single_scale(
                anchor_midas,
                anchor_depth,
            )

            three_mae, three_rmse, three_median_abs_error, three_valid_count = (
                evaluate_scale(
                    scale_a,
                    x,
                    y,
                )
            )

            coordinates = ";".join(
                f"{item[0]},{item[1]}"
                for item in three_anchors
            )

            depth_values = ";".join(
                f"{item[3]:.6f}"
                for item in three_anchors
            )

            midas_values = ";".join(
                f"{item[2]:.6f}"
                for item in three_anchors
            )

            rows.append(
                {
                    "index": index,
                    "sensor_type": str(
                        meta[5050 + index - 1]["sensorType"][0]
                    ),
                    "anchor_strategy": "three_anchors",
                    "anchor_count": len(three_anchors),
                    "anchor_coordinates": coordinates,
                    "valid_wall_pixels": int(len(x)),
                    "anchor_valid_count": len(three_anchors),
                    "anchor_depth_values_m": depth_values,
                    "anchor_midas_values": midas_values,
                    "scale_a": scale_a,
                    "mae_m": mae,
                    "rmse_m": rmse,
                    "median_abs_error_m": median_abs_error,
                    "oracle_inverse_affine_mae_m": oracle_mae,
                    "oracle_inverse_affine_rmse_m": oracle_rmse,
                }
            )

        print(
            f"  Valid wall pixels: {len(x):,}"
        )
        print(
            f"  Oracle inverse-affine MAE: "
            f"{oracle_mae:.4f} m"
        )

        for strategy, _ in requested_anchors:
            result = anchor_results[strategy]

            if result is None:
                print(
                    f"  {strategy}: no valid anchor"
                )
                continue

            anchor_x, anchor_y, anchor_d, anchor_z = result

            scale_a = fit_single_scale(
                np.array([anchor_d]),
                np.array([anchor_z]),
            )

            mae, rmse, median_abs_error, _ = (
                evaluate_scale(
                    scale_a,
                    x,
                    y,
                )
            )

            print(
                f"  {strategy}: "
                f"pixel=({anchor_x},{anchor_y}), "
                f"Z={anchor_z:.3f} m, "
                f"MAE={mae:.4f} m"
            )

        if three_anchors:
            print(
                f"  three_anchors: "
                f"count={len(three_anchors)}, "
                f"MAE={three_mae:.4f} m"
            )

    if not rows:
        raise RuntimeError(
            "No valid experiment results were produced."
        )

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

    print("\nExperiment file created and ready.")
    print(f"Results will be saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
