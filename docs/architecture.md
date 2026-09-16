# EDIproject System Architecture

## 1. Overview

EDIproject is organized as a modular computer-vision pipeline for estimating paintable wall area and painting cost from an indoor RGB image.

The system separates the workflow into independent modules for:

1. Image preprocessing
2. Wall segmentation
3. Door and window segmentation
4. Depth estimation
5. Metric-depth calibration
6. 3D geometric measurement
7. Material estimation
8. Cost estimation
9. Visualization
10. Interactive application

The main production flow is:

```text
RGB Image
    |
    v
Preprocessing
    |
    +---------------------+
    |                     |
    v                     v
Wall Segmentation   Opening Segmentation
    |                     |
    v                     v
Wall Mask          Door / Window Mask
    |                     |
    +----------+----------+
               |
               v
      Paintable Wall Mask
               |
               v
        MiDaS Depth
               |
               v
       Relative Depth
               |
               v
   Single-Anchor Calibration
               |
               v
        Metric Depth
               |
               v
      3D Backprojection
               |
               v
   Surface Triangulation
               |
               v
     Physical Wall Area
               |
               v
       Material Estimation
               |
               v
        Cost Estimation
```

## 2. Main Source Modules

The implementation is organized under `src/`.

```text
src/
├── app/
├── calibration/
├── depth/
├── detection/
├── estimation/
├── measurement/
├── pipeline/
├── processing/
├── segmentation/
└── visualization/
```

Each module has a specific responsibility.

## 3. Application Layer

### Location

```text
src/app/app_final.py
```

The Streamlit application provides the user interface for the production pipeline.

The application allows the user to:

- Upload an indoor RGB image.
- Enter the metric calibration factor.
- Enter camera intrinsic parameters.
- Start the estimation pipeline.
- View intermediate processing results.
- View the final paintable wall area.
- View material quantities.
- View labour cost.
- View total estimated cost.

The application calls the central pipeline rather than implementing the individual computer-vision algorithms itself.

## 4. Pipeline Layer

### Location

```text
src/pipeline/pipeline.py
```

The pipeline module coordinates the complete production workflow.

Its responsibilities include:

1. Loading the input image.
2. Running wall segmentation.
3. Running opening segmentation.
4. Creating the paintable wall mask.
5. Running MiDaS depth estimation.
6. Converting relative depth to metric depth.
7. Performing 3D wall-area measurement.
8. Estimating paint quantity.
9. Estimating primer quantity.
10. Estimating putty quantity.
11. Estimating labour cost.
12. Calculating total cost.

The pipeline also validates important numerical inputs such as calibration factor, focal lengths, and depth-jump threshold.

## 5. Segmentation Layer

### Location

```text
src/segmentation/
```

This layer contains the segmentation models and supporting functionality.

Important modules include:

```text
model.py
dataset.py
evaluate.py
inference.py
opening_dataset.py
opening_inference.py
train.py
train_opening.py
```

### Wall Segmentation

The production wall model uses:

```text
U-Net
    +
ResNet34 encoder
```

Primary checkpoint:

```text
models/checkpoints/best_model.pt
```

The production prediction threshold is:

```text
0.80
```

The threshold was selected using downstream physical wall-area validation.

### Opening Segmentation

A separate U-Net with a ResNet34 encoder predicts:

```text
0 = Background
1 = Door
2 = Window
```

Preferred checkpoint:

```text
models/checkpoints/opening_augmented_best_model.pt
```

The opening model includes augmentation during training.

## 6. Processing Layer

### Location

```text
src/processing/
```

This layer contains image-processing operations used by the project.

Important modules include:

```text
preprocessing.py
morphology.py
inpainting.py
```

Preprocessing prepares image data for model inference.

Morphological processing provides mask-processing operations.

Inpainting functionality exists for object-removal experiments and is not a required step in the primary production pipeline.

## 7. Depth Layer

### Location

```text
src/depth/
```

Important module:

```text
midas.py
```

MiDaS is used to estimate relative depth from the RGB image.

The output is not directly interpreted as metres.

Instead, the relative depth is passed to the calibration layer.

## 8. Calibration Layer

### Location

```text
src/calibration/
```

Important modules include:

```text
reference_object.py
scale.py
aruco.py
vanishing_points.py
```

The production pipeline uses the single-anchor inverse-scale calibration approach.

For a reference point:

```text
k = Z_reference × D_reference
```

Metric depth is then calculated as:

```text
Z = k / D
```

where:

- `Z` is metric depth.
- `D` is MiDaS relative depth.
- `k` is the calibration scale.

Camera intrinsic parameters are also required for 3D reconstruction:

```text
fx
fy
cx
cy
```

The calibration factor and camera intrinsics are camera/reference-specific.

ArUco and vanishing-point functionality represents additional calibration experimentation and is not required for the current primary production flow.

## 9. Measurement Layer

### Location

```text
src/measurement/
```

Important modules include:

```text
pixel_area.py
wall_area.py
wall_dimensions.py
opening_area.py
```

The measurement layer provides both basic pixel-area functionality and physical 3D area measurement.

The primary wall-area calculation uses metric depth and camera geometry.

For a pixel `(u, v)`:

```text
X = (u - cx) × Z / fx
Y = (v - cy) × Z / fy
```

This produces a 3D point:

```text
P = (X, Y, Z)
```

Neighbouring points are combined into triangular surface elements.

For vertices `P1`, `P2`, and `P3`:

```text
Area = 0.5 × |(P2 - P1) × (P3 - P1)|
```

The valid triangle areas are summed to obtain physical wall area.

## 10. Depth-Jump Filtering

The physical-area calculation includes a depth-discontinuity filter.

Production value:

```text
max_depth_jump = 0.025 m
```

This prevents neighbouring points with excessive depth differences from forming invalid surface triangles.

The filter is particularly relevant near:

- Doors
- Windows
- Objects
- Segmentation boundaries
- Abrupt depth changes

## 11. Detection Layer

### Location

```text
src/detection/
```

Important modules include:

```text
object_mask.py
yolo.py
```

Object-mask functionality was developed to investigate removal of objects that may interfere with wall-area estimation.

YOLO-based detection and object-removal functionality is experimental.

It is not part of the current primary production pipeline.

The production pipeline instead relies on wall segmentation and opening segmentation to define the paintable region.

## 12. Estimation Layer

### Location

```text
src/estimation/
```

Modules include:

```text
paint.py
primer.py
putty.py
labour.py
total_cost.py
```

This layer converts physical paintable area into material quantities and cost.

The general material formula is:

```text
Quantity =
(Area × Coats / Coverage)
×
(1 + Wastage / 100)
```

Cost is:

```text
Cost = Quantity × Price
```

Labour is:

```text
Labour Cost = Area × Labour Rate
```

The total cost combines:

```text
Paint
+
Primer
+
Putty
+
Labour
```

Configuration is stored in:

```text
configs/estimation_config.yaml
```

The current values are illustrative assumptions and can be replaced with project-specific values.

## 13. Visualization Layer

### Location

```text
src/visualization/
```

Important modules include:

```text
dashboard_data.py
depth.py
masks.py
```

The visualization layer supports inspection of:

- Segmentation masks
- Depth maps
- Intermediate pipeline outputs
- Data used by the application dashboard

This allows the individual processing stages to be inspected instead of displaying only the final numerical result.

## 14. Model Configuration

### Location

```text
configs/model_config.yaml
```

The model configuration specifies the main wall segmentation setup, including:

- U-Net architecture
- ResNet34 encoder
- Input channels
- Output channels
- Training image size
- Batch size
- Number of epochs
- Learning rate
- Optimizer
- Loss functions
- Validation metric

The primary training configuration uses BCEWithLogitsLoss together with Dice loss.

## 15. Model Checkpoints

The primary checkpoints are stored under:

```text
models/checkpoints/
```

Production wall model:

```text
models/checkpoints/best_model.pt
```

Preferred production opening model:

```text
models/checkpoints/opening_augmented_best_model.pt
```

Additional checkpoints are retained for experimental comparison and are not automatically used by the production pipeline.

## 16. Data and Validation Layer

The project uses SUN RGB-D data for model development and validation.

Relevant directories include:

```text
data/raw/
data/processed/
outputs/validation/
```

Validation scripts are stored under:

```text
scripts/
```

Important validation functionality includes:

```text
validate_wall_area.py
validate_wall_area_tversky.py
test_midas_metric_feasibility.py
test_midas_single_anchor_calibration.py
```

The validation outputs are retained under:

```text
outputs/validation/
```

This separates experimental evidence from the production source code.

## 17. Test Architecture

Automated tests are stored under:

```text
tests/
```

The tests cover:

- Calibration
- Depth
- Detection
- Estimation
- Image processing
- Measurement
- Opening-area measurement
- Pipeline validation
- Segmentation

The current test suite contains 88 tests.

The complete suite has been verified with all 88 tests passing.

## 18. Production Data Flow

The production data flow can be summarized as:

```text
Input RGB Image
       |
       v
Wall Model --------------------+
       |                       |
       v                       |
Wall Mask                      |
                               |
Opening Model                  |
       |                       |
       v                       |
Door/Window Mask               |
       |                       |
       +-----------+-----------+
                   |
                   v
          Paintable Mask
                   |
                   v
              MiDaS
                   |
                   v
          Relative Depth
                   |
                   v
       Single-Anchor Calibration
                   |
                   v
            Metric Depth
                   |
                   v
       Camera Backprojection
                   |
                   v
       Triangular Surface Mesh
                   |
                   v
          Wall Surface Area
                   |
                   v
         Material Estimation
                   |
                   v
           Cost Estimation
```

## 19. Production vs Experimental Components

The current production architecture intentionally excludes several experimental components.

### Production

```text
Wall U-Net + ResNet34
Opening U-Net + ResNet34
MiDaS relative depth
Single-anchor inverse-scale calibration
Pinhole 3D backprojection
Triangular surface-area calculation
Depth-jump filtering
Material estimation
Cost estimation
Streamlit application
```

### Experimental

```text
Tversky wall checkpoint
YOLO object detection
Object-removal/inpainting workflow
ArUco calibration
Vanishing-point calibration
Inverse-affine MiDaS calibration
```

The experimental components remain in the repository for research comparison but are not required for the main production workflow.

## 20. Architectural Design Principle

The central architectural decision is to separate:

```text
Perception
    ↓
Depth
    ↓
Calibration
    ↓
Geometry
    ↓
Estimation
```

This separation allows each stage to be evaluated independently.

For example:

- Segmentation quality can be evaluated using IoU and Dice.
- Depth calibration can be evaluated using metric-depth error.
- Wall measurement can be evaluated using physical-area error.
- Cost estimation can be tested independently using known input areas and configuration values.

This modular structure also makes future improvements possible without replacing the entire system.

## 21. Summary

EDIproject follows a modular architecture in which computer vision, depth estimation, geometric reconstruction, and cost estimation are connected through a central production pipeline.

The main production chain is:

```text
RGB
 ↓
Segmentation
 ↓
Paintable Mask
 ↓
Relative Depth
 ↓
Metric Calibration
 ↓
3D Reconstruction
 ↓
Physical Wall Area
 ↓
Material Quantities
 ↓
Painting Cost
```

This architecture provides a clear separation between machine-learning prediction, geometric measurement, and final quantity/cost estimation.
