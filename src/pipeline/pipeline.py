from pathlib import Path

import cv2
import numpy as np
import yaml

from src.segmentation.inference import predict_wall_mask
from src.segmentation.opening_inference import predict_opening_masks
from src.depth.midas import load_midas, predict_depth
from src.calibration.scale import apply_calibration
from src.measurement.wall_area import calculate_wall_area_from_depth
from src.estimation.paint import paint_cost
from src.estimation.primer import primer_cost
from src.estimation.putty import putty_cost
from src.estimation.labour import labour_cost

__all__ = ["run_pipeline"]

_ESTIMATION_CONFIG_CACHE: dict[str, dict] = {}


def _load_estimation_config(estimation_config_path) -> dict:
    config_path = Path(estimation_config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Estimation config not found: {config_path}"
        )

    cache_key = str(config_path.resolve())
    cached = _ESTIMATION_CONFIG_CACHE.get(cache_key)
    if cached is not None:
        return cached

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Estimation config must contain a mapping")

    _ESTIMATION_CONFIG_CACHE[cache_key] = config
    return config


def _merge_cfg(base, overrides, allowed_keys, name):
    cfg = base.copy()
    if overrides is None:
        return cfg
    if not isinstance(overrides, dict):
        raise TypeError(f"{name}_overrides must be a dict")
    for k in allowed_keys:
        if k in overrides:
            v = overrides[k]
            if not isinstance(v, (int, float, np.integer, np.floating)):
                raise TypeError(f"{name}_overrides[{k}] must be numeric")
            if not np.isfinite(v):
                raise ValueError(f"{name}_overrides[{k}] must be finite")
            cfg[k] = v
    return cfg


def run_pipeline(
    image_path,
    calibration_factor,
    camera_fx,
    camera_fy,
    camera_cx,
    camera_cy,
    wall_checkpoint_path="models/checkpoints/best_model.pt",
    opening_checkpoint_path="models/checkpoints/opening_augmented_best_model.pt",
    estimation_config_path="configs/estimation_config.yaml",
    max_depth_jump=0.025,
    paint_overrides=None,
    primer_overrides=None,
    putty_overrides=None,
    labour_overrides=None,
):
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    numeric_inputs = {
        "calibration_factor": calibration_factor,
        "camera_fx": camera_fx,
        "camera_fy": camera_fy,
        "camera_cx": camera_cx,
        "camera_cy": camera_cy,
    }

    for name, value in numeric_inputs.items():
        if not isinstance(value, (int, float, np.integer, np.floating)):
            raise TypeError(f"{name} must be a numeric scalar")
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")

    if calibration_factor <= 0:
        raise ValueError("calibration_factor must be positive")
    if camera_fx <= 0:
        raise ValueError("camera_fx must be positive")
    if camera_fy <= 0:
        raise ValueError("camera_fy must be positive")
    if max_depth_jump is not None and max_depth_jump < 0:
        raise ValueError("max_depth_jump must be non-negative")

    wall_mask = predict_wall_mask(
        image_path,
        checkpoint_path=wall_checkpoint_path,
    )

    door_mask, window_mask = predict_opening_masks(
        image_path,
        checkpoint_path=opening_checkpoint_path,
    )

    if wall_mask.shape != door_mask.shape or wall_mask.shape != window_mask.shape:
        raise ValueError("Wall and opening masks must have the same shape")

    # A_paintable = A_wall - A_openings
    paintable_mask = wall_mask & ~(door_mask | window_mask)

    if not np.any(paintable_mask):
        raise ValueError("Paintable mask contains no pixels")

    midas_model, midas_transform, midas_device = load_midas()

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    relative_depth = predict_depth(
        image,
        midas_model,
        midas_transform,
        midas_device,
    )

    metric_depth = apply_calibration(
        relative_depth,
        calibration_factor,
    )

    valid_depth = np.isfinite(metric_depth) & (metric_depth > 0)
    if not np.any(valid_depth):
        raise ValueError("Metric depth contains no valid positive values")

    paintable_area_m2 = calculate_wall_area_from_depth(
        paintable_mask,
        metric_depth,
        camera_fx,
        camera_fy,
        camera_cx,
        camera_cy,
        max_depth_jump=max_depth_jump,
    )

    config = _load_estimation_config(estimation_config_path)

    effective_paint_config = _merge_cfg(
        config["paint"],
        paint_overrides,
        ("coats", "coverage_m2_per_litre", "price_per_litre", "wastage_percent"),
        "paint",
    )
    effective_primer_config = _merge_cfg(
        config["primer"],
        primer_overrides,
        ("coats", "coverage_m2_per_litre", "price_per_litre", "wastage_percent"),
        "primer",
    )
    effective_putty_config = _merge_cfg(
        config["putty"],
        putty_overrides,
        ("coats", "coverage_m2_per_kg", "price_per_kg", "wastage_percent"),
        "putty",
    )
    effective_labour_config = _merge_cfg(
        config["labour"],
        labour_overrides,
        ("rate_per_m2",),
        "labour",
    )

    paint_quantity_l, paint_cost_value = paint_cost(
        paintable_area_m2,
        effective_paint_config["coats"],
        effective_paint_config["coverage_m2_per_litre"],
        effective_paint_config["price_per_litre"],
        effective_paint_config["wastage_percent"],
    )

    primer_quantity_l, primer_cost_value = primer_cost(
        paintable_area_m2,
        effective_primer_config["coats"],
        effective_primer_config["coverage_m2_per_litre"],
        effective_primer_config["price_per_litre"],
        effective_primer_config["wastage_percent"],
    )

    putty_quantity_kg, putty_cost_value = putty_cost(
        paintable_area_m2,
        effective_putty_config["coats"],
        effective_putty_config["coverage_m2_per_kg"],
        effective_putty_config["price_per_kg"],
        effective_putty_config["wastage_percent"],
    )

    labour_cost_value = labour_cost(
        paintable_area_m2,
        effective_labour_config["rate_per_m2"],
    )

    total_cost_value = (
        paint_cost_value
        + primer_cost_value
        + putty_cost_value
        + labour_cost_value
    )

    return {
        "wall_mask": wall_mask,
        "door_mask": door_mask,
        "window_mask": window_mask,
        "paintable_mask": paintable_mask,
        "relative_depth": relative_depth,
        "metric_depth": metric_depth,
        "paintable_area_m2": paintable_area_m2,
        "calibration_factor": calibration_factor,
        "paint_quantity_l": paint_quantity_l,
        "paint_cost": paint_cost_value,
        "primer_quantity_l": primer_quantity_l,
        "primer_cost": primer_cost_value,
        "putty_quantity_kg": putty_quantity_kg,
        "putty_cost": putty_cost_value,
        "labour_cost": labour_cost_value,
        "total_cost": total_cost_value,
    }
