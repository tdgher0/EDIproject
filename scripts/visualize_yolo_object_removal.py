import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from PIL import Image

from src.segmentation.inference import predict_wall_mask
from src.detection.yolo import YOLODetector
from src.detection.object_mask import remove_objects_from_wall

indices = [57, 71, 965, 2020, 2486, 2642, 3145, 3499, 4502, 4854, 4926, 5059]

output_dir = Path("outputs/validation/yolo_object_removal")
output_dir.mkdir(parents=True, exist_ok=True)

detector = YOLODetector()

for index in indices:
    image_path = Path(f"data/raw/sunrgbd/img-{index:06d}.jpg")
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    wall_mask = predict_wall_mask(str(image_path))
    detections = detector.predict(image)
    usable_wall = remove_objects_from_wall(wall_mask, detections)

    object_wall_mask = wall_mask & ~usable_wall

    # U-Net wall mask visualization.
    wall_visual = np.zeros_like(image_rgb)
    wall_visual[wall_mask] = 255

    # Actual YOLO object masks over the original image.
    yolo_visual = image_rgb.copy()

    for detection in detections:
        mask = detection["mask"]
        class_name = detection["class_name"]
        confidence = detection["confidence"]
        x1, y1, x2, y2 = detection["bbox"]

        # Draw object mask boundary.
        mask_uint8 = (mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(
            mask_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        for contour in contours:
            cv2.polylines(
                yolo_visual,
                [contour],
                True,
                (255, 0, 0),
                2,
            )

        label = f"{class_name} {confidence:.2f}"
        cv2.putText(
            yolo_visual,
            label,
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

    # Final usable wall mask.
    usable_visual = np.zeros_like(image_rgb)
    usable_visual[usable_wall] = 255

    # Highlight only object pixels removed from the wall.
    overlap_visual = image_rgb.copy()
    overlap_visual[object_wall_mask] = [255, 0, 0]

    top = np.hstack([image_rgb, wall_visual])
    bottom = np.hstack([yolo_visual, overlap_visual])
    sheet = np.vstack([top, bottom])

    output_path = output_dir / f"img-{index:06d}_yolo_removal.jpg"
    Image.fromarray(sheet).save(output_path)

    print(
        f"{index}: detections={len(detections)}, "
        f"wall={int(wall_mask.sum())}, "
        f"object_overlap={int(object_wall_mask.sum())}, "
        f"usable={int(usable_wall.sum())}"
    )

print(f"\nSaved diagnostic images to: {output_dir}")
