# EDIproject
## Automated Wall Area Measurement and Painting Cost Estimation from Indoor RGB Images

EDIproject is a computer-vision-based system designed to estimate the paintable wall area of an indoor scene and calculate the corresponding requirements and cost for painting-related materials and labour.

The system uses semantic segmentation to identify walls and openings such as doors and windows, monocular depth estimation to obtain relative depth, metric calibration, and 3D geometric reconstruction to estimate physical wall area. The resulting paintable area is used to estimate paint, primer, putty, labour, and total cost.

The project is implemented in Python and includes a Streamlit-based application for interactive estimation.

---

## 1. Project Objectives

1. Detect and segment wall regions from indoor RGB images.
2. Detect doors and windows so non-paintable regions can be excluded.
3. Estimate relative depth from a monocular RGB image.
4. Convert relative depth into metric depth using single-anchor calibration.
5. Calculate physical wall area using 3D geometry rather than pixel area alone.
6. Calculate paintable wall area after removing detected openings.
7. Estimate quantities of paint, primer, and putty.
8. Estimate labour cost and total painting cost.
9. Provide an interactive Streamlit application.
10. Validate the system using SUN RGB-D data.

---

## 2. System Architecture

```text
RGB Image
    |
    +--------------------+
    |                    |
    v                    v
Wall Segmentation    Opening Segmentation
U-Net + ResNet34     U-Net + ResNet34
    |                    |
    v                    v
Wall Mask            Door / Window Mask
    |                    |
    +---------+----------+
              |
              v
       Paintable Wall Mask
              |
              v
       MiDaS Depth Model
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
   +----------+----------+
   |          |          |
   v          v          v
 Paint      Primer     Putty
   |          |          |
   +----------+----------+
              |
              v
         Labour Cost
              |
              v
          Total Cost
```

---

## 3. Processing Pipeline

### 3.1 Wall Segmentation

Wall regions are detected using a U-Net architecture with a ResNet34 encoder.

Primary wall checkpoint:

```text
models/checkpoints/best_model.pt
```

Training configuration:

- Image size: 512 × 512
- Batch size: 4
- Epochs: 20
- Learning rate: 0.0001
- Optimizer: Adam
- BCEWithLogitsLoss
- Dice loss

Production wall threshold:

```text
0.80
```

The threshold was selected using downstream physical wall-area validation rather than segmentation metrics alone.

### 3.2 Door and Window Segmentation

A separate U-Net with ResNet34 is used to identify opening regions.

Classes:

```text
0 = Background
1 = Door
2 = Window
```

Preferred opening checkpoint:

```text
models/checkpoints/opening_augmented_best_model.pt
```

The augmented model uses horizontal-flip and rotation augmentation.

### 3.3 Paintable Wall Mask

The final paintable region is calculated conceptually as:

```text
paintable_mask = wall_mask AND NOT(opening_mask)
```

Detected doors and windows are therefore excluded from the wall area used for material estimation.

---

## 4. Monocular Depth Estimation

The project uses MiDaS for monocular depth estimation.

MiDaS produces relative depth rather than direct metric distance. The production system therefore applies single-anchor inverse-scale calibration.

The relationship is:

```text
Z = k / D
```

where:

- `Z` = metric depth
- `D` = MiDaS relative depth
- `k` = calibration scale

For a reference point with known physical depth:

```text
k = Z_reference × D_reference
```

The calibration factor is then applied to the depth map.

---

## 5. Camera Calibration

Metric reconstruction uses the camera intrinsic parameters:

```text
fx = focal length in x direction
fy = focal length in y direction
cx = principal point x coordinate
cy = principal point y coordinate
```

These parameters are camera-specific.

The calibration factor is also not a universal constant. It depends on the reference/calibration procedure and camera setup.

The values shown in the application are example/default values and should be replaced with values appropriate for the camera and calibration setup being used.

---

## 6. 3D Wall Area Calculation

Wall area is not estimated from pixel count alone.

Valid wall pixels are projected into 3D using the pinhole camera model.

For pixel `(u, v)` and depth `Z`:

```text
X = (u - cx) × Z / fx
Y = (v - cy) × Z / fy
```

This produces:

```text
P = (X, Y, Z)
```

Neighbouring 3D points are connected into triangular surface elements.

For triangle vertices `P1`, `P2`, and `P3`:

```text
Area = 0.5 × |(P2 - P1) × (P3 - P1)|
```

The areas of valid triangles are summed to obtain physical wall area.

---

## 7. Depth Discontinuity Filtering

Abrupt depth changes can occur around doors, windows, objects, and segmentation boundaries.

The production geometry uses a maximum depth-jump threshold of:

```text
0.025 m
```

Neighbouring points with excessive depth differences are prevented from forming valid surface elements. This reduces artificial surface patches across strong depth discontinuities.

---

## 8. Material and Cost Estimation

Material quantities are calculated from the estimated paintable area.

General quantity formula:

```text
Quantity = (Area × Coats / Coverage) × (1 + Wastage / 100)
```

Cost formula:

```text
Cost = Quantity × Price
```

Labour:

```text
Labour Cost = Area × Labour Rate
```

Total:

```text
Total Cost =
Paint Cost
+ Primer Cost
+ Putty Cost
+ Labour Cost
```

The current estimation configuration contains illustrative assumptions for coverage, prices, wastage, coats, and labour rate.

These values are not universal market rates and should be replaced with appropriate local or project-specific values for real-world estimation.

---

## 9. Validation Results

### 9.1 Wall Area Validation

A validation run was performed on SUN RGB-D training indices 1–100 using the production wall configuration.

Results:

- 91 wall-containing cases suitable for wall-area comparison.
- 8 images had no wall pixels in the ground-truth mask.
- 4 of those 8 no-wall images were correctly predicted as empty.
- One image produced no predicted wall pixels and therefore no valid area estimate.

For the 91 wall-containing cases:

```text
Mean Absolute Error:   0.4821 m²
Mean Percentage Error: 21.54%
```

Validation output:

```text
outputs/validation/wall_area_validation.csv
```

The 0.80 production threshold was selected because downstream physical-area validation showed better performance than lower tested thresholds.

### 9.2 Wall Segmentation

Primary model results:

```text
IoU       = 0.5651
Dice      = 0.6630
Precision = 0.6614
Recall    = 0.8032
```

The model can over-segment walls in difficult indoor scenes, so downstream physical-area validation is important in addition to conventional segmentation metrics.

### 9.3 Opening Segmentation

Results for the augmented opening model:

```text
Class       IoU       Dice      Precision   Recall

Background  0.9673    0.9834    0.9751      0.9918
Door        0.2594    0.4120    0.7178      0.2889
Window      0.3472    0.5155    0.5958      0.4542
```

Mean metrics:

```text
Mean IoU  = 0.5247
Mean Dice = 0.6369
```

Door and window segmentation is more challenging than background segmentation. Opening-area valid-pair statistics should therefore be interpreted carefully because missed opening predictions are not represented in a valid-pair-only error calculation.

---

## 10. End-to-End Pipeline

Production pipeline:

```text
src/pipeline/pipeline.py
```

The pipeline performs:

1. Image loading.
2. Wall segmentation.
3. Opening segmentation.
4. Paintable-mask generation.
5. MiDaS depth estimation.
6. Metric-depth calibration.
7. 3D backprojection.
8. Surface triangulation.
9. Depth-jump filtering.
10. Paintable wall-area estimation.
11. Paint estimation.
12. Primer estimation.
13. Putty estimation.
14. Labour estimation.
15. Total-cost calculation.

Intermediate results are returned so the application can visualize the processing stages.

---

## 11. Streamlit Application

Application:

```text
src/app/app_final.py
```

The application provides:

- RGB image upload
- Calibration-factor input
- Camera intrinsic input
- Pipeline execution
- Wall segmentation visualization
- Opening segmentation visualization
- Paintable-wall visualization
- Relative-depth visualization
- Metric-depth visualization
- 3D wall-area explanation
- Paint quantity
- Primer quantity
- Putty quantity
- Labour cost
- Total cost

Run the application with:

```powershell
.\venv\Scripts\Activate.ps1
python -m streamlit run src/app/app_final.py
```

---

## 12. Installation

Create and activate the Python virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

### PyTorch / CUDA

The development environment uses a CUDA-enabled PyTorch build. The application automatically uses a CUDA GPU when available and falls back to CPU otherwise.

If installing on a different system, install a PyTorch build compatible with your hardware from the official PyTorch installation instructions, then install the remaining project dependencies.

### Required model files

The trained model files are not included in this repository because of their large size.

The production pipeline requires:

```text
models/checkpoints/best_model.pt
models/checkpoints/opening_augmented_best_model.pt
yolo11n-seg.pt
```

Place the model files at the paths shown above before running the application.

Large datasets, trained checkpoints, and generated outputs are intentionally excluded from Git.

### Run the application

```powershell
python -m streamlit run src/app/app_final.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## 13. Testing

The project contains automated tests for:

- Calibration
- Depth processing
- Detection
- Estimation
- Image processing
- Measurement
- Opening-area measurement
- Pipeline
- Segmentation

The test suite contains 88 tests and was previously verified with:

```powershell
python -m pytest -v
```

Result:

```text
88 passed
```

---

## 14. Project Structure

```text
EDIproject/
│
├── configs/
│   ├── estimation_config.yaml
│   └── model_config.yaml
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
│
├── models/
│   ├── checkpoints/
│   │   ├── best_model.pt
│   │   └── opening_augmented_best_model.pt
│   └── checkpoints_tversky/
│
├── notebooks/
│
├── outputs/
│   └── validation/
│
├── scripts/
│
├── src/
│   ├── app/
│   ├── calibration/
│   ├── depth/
│   ├── detection/
│   ├── estimation/
│   ├── measurement/
│   ├── pipeline/
│   ├── processing/
│   ├── segmentation/
│   └── visualization/
│
├── tests/
│
├── SUNRGBDtoolbox/
│
├── requirements.txt
└── README.md
```

---

## 15. Important Production Components

### Wall Model

```text
models/checkpoints/best_model.pt
```

### Opening Model

```text
models/checkpoints/opening_augmented_best_model.pt
```

### Pipeline

```text
src/pipeline/pipeline.py
```

### Streamlit Application

```text
src/app/app_final.py
```

### Wall Validation

```text
scripts/validate_wall_area.py
```

### Estimation Configuration

```text
configs/estimation_config.yaml
```

### Model Configuration

```text
configs/model_config.yaml
```

---

## 16. Experimental Components

The following components were investigated during development but are not part of the primary production pipeline.

### Tversky Wall Model

A Tversky-loss-based wall segmentation checkpoint was evaluated experimentally. The primary production checkpoint remains:

```text
models/checkpoints/best_model.pt
```

### YOLO Object Removal

YOLO-based object detection and object-removal/inpainting experiments were performed. They remain experimental and are not integrated into the primary production pipeline.

### ArUco Calibration

ArUco calibration experiments are present but are not required by the current primary pipeline.

### Alternative MiDaS Calibration

Inverse-affine MiDaS calibration was investigated experimentally. The production methodology remains single-anchor inverse-scale calibration.

---

## 17. Limitations

### Wall Segmentation

The wall model can over-segment around furniture, cabinets, appliances, windows, and visually similar surfaces. Such errors can increase estimated wall area.

### Opening Segmentation

Missed doors or windows can cause non-paintable regions to remain inside the estimated paintable area.

### Monocular Depth

MiDaS provides relative depth, so metric reconstruction depends on calibration.

### Camera Parameters

Incorrect camera intrinsic parameters can introduce errors into 3D reconstruction.

### Calibration

Single-anchor calibration depends on a suitable reference depth. Calibration errors affect metric depth and therefore physical area.

### Dataset

SUN RGB-D contains varied indoor scenes and annotations that may not perfectly represent every downstream wall-area measurement situation.

### Cost Assumptions

Material coverage, prices, wastage, coats, and labour rates are configurable assumptions and should be adapted to the actual application.

---

## 18. Future Improvements

1. Improve wall segmentation using a larger and more diverse dataset.
2. Improve door and window segmentation, especially recall.
3. Add more robust multi-point depth calibration.
4. Improve automatic camera calibration.
5. Use RGB-D sensors where direct metric depth is available.
6. Improve geometric surface reconstruction.
7. Improve handling of curved or irregular wall surfaces.
8. Improve segmentation around furniture and appliances.
9. Add confidence estimation for wall-area predictions.
10. Add real-world regional material pricing.
11. Support multiple rooms and complete-building estimation.
12. Improve automatic handling of images with no valid wall region.

---

## 19. Conclusion

EDIproject demonstrates an end-to-end approach for automated wall-area measurement and painting cost estimation from indoor imagery.

The system combines:

```text
Semantic Segmentation
        +
Opening Detection
        +
Monocular Depth Estimation
        +
Metric Calibration
        +
3D Geometric Reconstruction
        +
Material Estimation
        +
Cost Estimation
```

The key design principle is that wall area is estimated from a reconstructed 3D surface rather than from pixel counts alone.

The current research prototype provides:

- Wall segmentation
- Door/window segmentation
- Paintable-mask generation
- Monocular depth estimation
- Single-anchor metric calibration
- 3D wall-area measurement
- Material estimation
- Labour estimation
- Total cost estimation
- Streamlit interface
- Automated tests
- Validation scripts
- Experimental evaluation

The project provides a foundation for further improvements in segmentation, calibration, metric depth estimation, geometric reconstruction, and real-world cost estimation.
