# EDIproject Methodology

## 1. Introduction

EDIproject is a computer-vision-based system for estimating the paintable wall area of an indoor scene from an RGB image and converting that measured area into material quantities and painting cost.

The methodology combines semantic segmentation, relative-depth estimation, metric calibration, camera geometry, 3D surface-area computation, and quantity/cost estimation.

The production processing chain is:

```text
RGB image
   |
   v
Wall segmentation
   |
   v
Door/window segmentation
   |
   v
Paintable wall mask
   |
   v
MiDaS relative depth
   |
   v
Single-anchor metric calibration
   |
   v
3D wall-area measurement
   |
   v
Paintable area
   |
   v
Material quantity estimation
   |
   v
Labour and total cost
```

The objective is not simply to count wall pixels. Pixel area changes with perspective and depth, so the system estimates a 3D surface area using depth and camera geometry.

---

## 2. Problem Definition

Given an indoor RGB image, the system estimates:

1. The pixels belonging to walls.
2. The pixels belonging to doors and windows.
3. The paintable portion of the wall.
4. The metric depth of valid wall pixels.
5. The physical 3D surface area of the paintable wall.
6. The required quantities of paint, primer, and putty.
7. The associated labour cost and total estimated painting cost.

The central measurement problem is:

```text
RGB image + depth + camera calibration
                    |
                    v
          3D wall surface reconstruction
                    |
                    v
             wall surface area
```

---

## 3. Input and Output

### 3.1 Input

The production application accepts:

- An indoor RGB image.
- A metric calibration factor.
- Camera focal lengths `fx` and `fy`.
- Camera principal-point coordinates `cx` and `cy`.

The application also uses trained model checkpoints and the estimation configuration.

### 3.2 Output

The production pipeline provides:

- Wall segmentation mask.
- Door segmentation mask.
- Window segmentation mask.
- Paintable-wall mask.
- MiDaS relative-depth map.
- Metric-depth map.
- Paintable wall area in square metres.
- Paint quantity in litres.
- Primer quantity in litres.
- Putty quantity in kilograms.
- Labour cost.
- Total estimated cost.

Intermediate results are also returned by the pipeline so that the processing stages can be inspected.

---

## 4. Overall System Methodology

The production system is divided into the following major stages:

### Stage 1: Wall segmentation

A U-Net model with a ResNet34 encoder predicts the wall region from the RGB image.

### Stage 2: Opening segmentation

A second U-Net with a ResNet34 encoder predicts:

```text
0 = background
1 = door
2 = window
```

### Stage 3: Paintable-wall extraction

Door and window regions are removed from the predicted wall region.

### Stage 4: Relative-depth estimation

MiDaS produces a relative-depth representation from the RGB image.

### Stage 5: Metric-depth calibration

The relative depth is converted into metric depth using a single known reference and an inverse-scale relationship.

### Stage 6: 3D area calculation

Wall pixels are backprojected into 3D using the camera intrinsics. A triangular mesh is then used to estimate the wall surface area.

### Stage 7: Depth discontinuity filtering

Triangles crossing sufficiently large depth discontinuities are rejected to reduce unreliable surface-area contributions.

### Stage 8: Material and cost estimation

The measured paintable area is converted into paint, primer, putty, labour, and total cost using configurable assumptions.

---

## 5. Wall Segmentation

### 5.1 Model

The production wall-segmentation model is:

```text
U-Net + ResNet34 encoder
```

The trained checkpoint is:

```text
models/checkpoints/best_model.pt
```

The model was trained using:

- 512 × 512 image size
- Batch size 4
- 20 epochs
- Learning rate `1e-4`
- Adam optimizer
- BCEWithLogits loss combined with Dice loss

The best validation IoU during baseline model training was approximately `0.5723` at epoch 18. This is a model-training segmentation metric and is distinct from the downstream physical wall-area validation reported below.

### 5.2 Production threshold

The final production probability threshold is:

```text
0.80
```

The threshold was selected using downstream wall-area measurement rather than segmentation metrics alone. This separates the model-training segmentation evaluation from the physical area-measurement objective.

A controlled threshold comparison on wall-containing validation images showed that the threshold of `0.80` produced the lowest measured wall-area mean absolute error among the evaluated thresholds.

The evaluated thresholds were:

```text
0.50
0.60
0.70
0.80
```

At threshold `0.80`, the wall-containing validation set produced approximately:

```text
Mean absolute area error: 0.4821 m²
Median absolute area error: 0.1923 m²
Mean percentage error: 21.54%
Median percentage error: 6.44%
```

These values describe the evaluated validation subset and should not be interpreted as universal performance for arbitrary indoor images.

---

## 6. Opening Segmentation

Wall area cannot be treated as completely paintable because doors and windows normally occupy portions of a wall.

Therefore, the system uses a separate semantic segmentation model for openings.

### 6.1 Model

The production opening model is:

```text
U-Net + ResNet34 encoder
```

The preferred checkpoint is:

```text
models/checkpoints/opening_augmented_best_model.pt
```

The model predicts three classes:

```text
0 = background
1 = door
2 = window
```

### 6.2 Data augmentation

The augmented opening model used:

- 50% horizontal flipping
- Random rotation between -10° and +10°

The selected augmented model improved validation IoU for both opening classes compared with the original, non-augmented opening model.

The best augmented validation results were approximately:

```text
Background IoU: 0.9637
Door IoU:       0.3487
Window IoU:     0.3807
```

### 6.3 Opening segmentation test performance

On the opening test set, the production-style evaluation produced:

```text
Class       IoU       Dice      Precision   Recall
----------------------------------------------------
Background  0.9673    0.9834    0.9751      0.9918
Door        0.2594    0.4120    0.7178      0.2889
Window      0.3472    0.5155    0.5958      0.4542
```

The mean IoU was approximately:

```text
0.5247
```

and mean Dice was approximately:

```text
0.6369
```

The relatively lower door and window performance indicates that opening segmentation remains an important source of uncertainty.

---

## 7. Paintable-Wall Mask

The paintable wall is calculated by combining the wall mask with the opening masks.

Conceptually:

```text
paintable_mask =
    wall_mask AND NOT(door_mask OR window_mask)
```

Therefore:

- Wall pixels remain candidates for painting.
- Door pixels are removed.
- Window pixels are removed.
- Background pixels are removed.

This stage converts the semantic understanding of the scene into the specific region whose physical area is required for painting estimation.

---

## 8. MiDaS Relative Depth

MiDaS is used to estimate depth information from the RGB image.

The model provides relative depth rather than directly guaranteed metric distance.

Therefore, the raw MiDaS output cannot by itself be interpreted as metres.

The production methodology treats MiDaS output as a relative-depth representation and performs a separate metric calibration step.

---

## 9. Single-Anchor Metric Calibration

### 9.1 Motivation

For physical wall-area measurement, depth must be expressed in metric units.

A known reference object or reference distance can provide one metric depth anchor.

If:

```text
Z_ref = known metric reference depth
D_ref = MiDaS relative depth at the reference
```

then the production scale parameter is:

```text
k = Z_ref × D_ref
```

### 9.2 Inverse-scale relationship

The production metric-depth relationship is:

```text
Z = k / D
```

where:

- `Z` is metric depth.
- `k` is the calibrated scale parameter.
- `D` is MiDaS relative depth.

Therefore:

```text
metric_depth = k / MiDaS_relative_depth
```

This inverse relationship is the production methodology.

### 9.3 Calibration limitation

The calibration factor is camera/setup dependent.

The current application displays an example/default calibration factor and camera intrinsics, but these values must not be treated as universal constants.

A different camera or calibration setup requires appropriate calibration parameters.

---

## 10. Camera Model

The system uses a pinhole-camera model to convert image coordinates and metric depth into 3D coordinates.

The camera intrinsic matrix is:

```text
K =
[ fx   0   cx ]
[  0  fy   cy ]
[  0   0    1 ]
```

where:

- `fx` = focal length in pixels along the x-axis.
- `fy` = focal length in pixels along the y-axis.
- `cx` = x-coordinate of the principal point.
- `cy` = y-coordinate of the principal point.

These parameters are supplied to the production pipeline.

---

## 11. 3D Backprojection

For an image pixel `(u, v)` with metric depth `Z`, the pinhole model gives:

```text
X = (u - cx) × Z / fx

Y = (v - cy) × Z / fy

Z = Z
```

Thus, each valid paintable-wall pixel can be represented by a 3D point:

```text
P = (X, Y, Z)
```

The collection of these points forms an approximate 3D representation of the visible paintable wall.

---

## 12. Triangular Mesh Surface Area

A wall is a surface rather than a flat collection of pixels. Because depth varies across the image, directly converting pixel count into area would not correctly account for perspective.

The system therefore forms local triangles from neighbouring valid 3D points.

For three 3D points:

```text
P1, P2, P3
```

the triangle area is:

```text
A_triangle = 0.5 × ||(P2 - P1) × (P3 - P1)||
```

where `×` denotes the vector cross product.

The total visible paintable-wall area is obtained by summing the valid triangle areas:

```text
A_wall = Σ A_triangle
```

This produces an estimate of physical surface area in square metres when the depth calibration and camera parameters are metric.

---

## 13. Depth-Jump Filtering

Depth discontinuities can occur around:

- Doors
- Windows
- Furniture
- Object boundaries
- Segmentation boundaries
- Other regions where neighbouring pixels do not belong to the same continuous surface

If a triangle connects points with a large depth difference, treating the triangle as a continuous wall surface can create an artificial surface.

The production measurement therefore uses a depth-jump filter.

The current production implementation uses:

```text
max_depth_jump = 0.025
```

This is the configured depth-jump threshold used by the production implementation; it is not a universal or camera-independent optimal threshold.

Triangles crossing a sufficiently large depth discontinuity are excluded from the area calculation.

This improves the physical validity of the reconstructed surface by preventing unreliable triangles from contributing to the measured area.

---

## 14. Wall Area Measurement

The complete wall-area calculation is therefore:

```text
RGB image
   |
   +--> wall segmentation
   |
   +--> opening segmentation
   |
   v
paintable mask
   |
   v
MiDaS relative depth
   |
   v
metric depth
   |
   v
camera backprojection
   |
   v
3D points
   |
   v
valid local triangles
   |
   v
depth-jump filtering
   |
   v
sum of triangle areas
   |
   v
paintable wall area (m²)
```

This is the central scientific measurement stage of the project.

---

## 15. Material Quantity Estimation

Once the paintable area `A` is known, the system estimates material quantities using configurable coverage, coat count, and wastage assumptions.

The general quantity equation is:

```text
Quantity = (A × C / R) × (1 + W / 100)
```

where:

- `A` = paintable area in m².
- `C` = number of coats.
- `R` = material coverage in m² per unit.
- `W` = wastage percentage.

The unit depends on the material:

- Litres for paint.
- Litres for primer.
- Kilograms for putty.

---

## 16. Paint Estimation

For paint:

```text
Paint quantity (L) =
    (Area × Paint coats / Paint coverage)
    × (1 + wastage / 100)
```

The current example configuration uses:

```text
Coverage: 12.0 m²/L
Price: ₹30/L
Wastage: 5%
Coats: 2
```

These are illustrative project assumptions, not universal market prices.

Paint cost is:

```text
Paint cost = Paint quantity × Price per litre
```

---

## 17. Primer Estimation

For primer:

```text
Primer quantity (L) =
    (Area × Primer coats / Primer coverage)
    × (1 + wastage / 100)
```

The current example configuration uses:

```text
Coverage: 15.0 m²/L
Price: ₹18/L
Wastage: 5%
Coats: 1
```

Primer cost is:

```text
Primer cost = Primer quantity × Price per litre
```

---

## 18. Putty Estimation

For putty:

```text
Putty quantity (kg) =
    (Area × Putty coats / Putty coverage)
    × (1 + wastage / 100)
```

The current example configuration uses:

```text
Coverage: 4.0 m²/kg
Price: ₹25/kg
Wastage: 5%
Coats: 1
```

Putty cost is:

```text
Putty cost = Putty quantity × Price per kilogram
```

---

## 19. Labour Estimation

Labour cost is estimated directly from measured area:

```text
Labour cost = Area × Labour rate per m²
```

The current example configuration uses:

```text
Labour rate: ₹20/m²
```

As with material prices, this is an illustrative configurable assumption.

---

## 20. Total Cost

The total painting cost is:

```text
Total cost =
    Paint cost
  + Primer cost
  + Putty cost
  + Labour cost
```

This makes the cost-estimation stage modular because material prices, coverage rates, wastage, coats, and labour rates can be changed without changing the computer-vision measurement methodology.

---

## 21. Validation Methodology

The project validates the system at multiple levels rather than relying only on the final cost output. These evaluations are intended to characterize the implemented system on the evaluated datasets and subsets, rather than provide an end-to-end accuracy guarantee for arbitrary real-world rooms.

### 21.1 Segmentation validation

Wall and opening segmentation are evaluated using metrics such as:

- IoU
- Dice score
- Precision
- Recall

### 21.2 Physical area validation

The wall-area pipeline is evaluated against reference wall areas using absolute and percentage errors.

For the controlled wall-containing validation subset with the production threshold of `0.80`, the measured performance was approximately:

```text
Mean absolute error:    0.4821 m²
Median absolute error:  0.1923 m²
Mean percentage error:  21.54%
Median percentage error: 6.44%
```

### 21.3 Geometry isolation

The project also compared area calculation using the predicted wall mask against area calculation using the reference wall mask.

This isolates the contribution of segmentation errors from the geometry calculation itself.

The reference-mask experiment showed that the geometry component itself contributed very little error compared with the predicted-mask case on the evaluated subset.

---

## 22. Opening-Area Validation

Opening segmentation and opening-area estimation were evaluated separately.

For valid ground-truth/prediction pairs in the opening-area validation:

```text
Door valid pairs:      658
Door MAE:              0.6201 m²
Door median AE:        0.2461 m²

Window valid pairs:    782
Window MAE:            0.6260 m²
Window median AE:      0.2904 m²

Combined valid pairs:  1482
Combined MAE:          0.6636 m²
Combined median AE:    0.2970 m²
```

These figures are based on valid evaluation pairs and therefore should be interpreted together with the opening detection/segmentation coverage and missing-pair information.

---

## 23. Production and Experimental Methodology

The project deliberately separates the production methodology from research experiments.

### Production

The production system uses:

```text
Baseline U-Net ResNet34 wall model
        +
Augmented U-Net ResNet34 opening model
        +
MiDaS relative depth
        +
Single-anchor inverse-scale calibration
        +
Pinhole backprojection
        +
Triangular mesh area
        +
Depth-jump filtering
```

### Experimental

The project also contains experiments involving:

- Tversky-loss wall segmentation
- Alternative MiDaS inverse-affine calibration
- ArUco calibration
- Vanishing-point calibration
- YOLO object detection/removal
- Inpainting-based object removal

These approaches are retained for investigation and comparison but are not automatically part of the production pipeline.

---

## 24. Why the Production Methodology Uses Single-Anchor Calibration

Experiments showed that an inverse-affine calibration fitted using ground-truth depth can provide lower depth error than the production single-anchor inverse-scale method on the evaluated samples.

However, ground-truth-fitted calibration is not equivalent to the intended deployment procedure because it depends on depth information that would not normally be available during ordinary inference.

The production method therefore remains the explicit single-anchor inverse-scale calibration approach because it matches the intended calibration workflow and keeps the inference process independent of per-image ground-truth depth.

---

## 25. Important Assumptions

The methodology relies on several assumptions:

1. The input is an indoor RGB image.
2. The wall segmentation model identifies the visible wall region sufficiently well.
3. Door and window regions can be identified and removed from the paintable region.
4. MiDaS relative depth preserves useful scene-depth structure.
5. The supplied calibration factor correctly converts the relative depth into metric depth for the camera/setup.
6. The camera intrinsic parameters are appropriate for the input camera.
7. Valid neighbouring 3D points approximately represent continuous surface regions.
8. The depth-jump threshold removes major discontinuity artefacts without removing excessive valid wall surface.
9. Material coverage and price assumptions represent the intended application context.

---

## 26. Limitations

### 26.1 Segmentation errors

The largest practical source of wall-area error is wall-mask quality.

The validation analysis showed a tendency toward wall over-segmentation in difficult scenes.

Examples include areas around:

- Cabinets
- Refrigerators
- Microwaves
- Windows and blinds
- Other visually similar surfaces

### 26.2 Opening segmentation

Door and window segmentation has substantially lower IoU than background segmentation.

Missed or incorrectly segmented openings can therefore affect the paintable area.

### 26.3 Relative-depth estimation

MiDaS produces relative depth rather than directly measured metric depth.

A single-anchor calibration does not guarantee perfect metric depth throughout the scene.

### 26.4 Camera calibration

Incorrect camera intrinsics or an unsuitable calibration factor can affect 3D reconstruction and therefore physical area.

### 26.5 Visible surface only

The system measures the visible paintable surface represented in the input image. It does not automatically infer hidden wall regions outside the camera view.

### 26.6 Cost assumptions

Material prices, coverage, wastage, coat counts, and labour rates are configurable assumptions. The resulting cost is therefore an estimate rather than a fixed market quotation.

---

## 27. Reproducibility

The methodology is implemented through the project source tree.

Important modules include:

```text
src/segmentation/
src/depth/
src/calibration/
src/measurement/
src/estimation/
src/pipeline/
src/app/
```

Production checkpoints are stored under:

```text
models/checkpoints/
```

Configuration is stored under:

```text
configs/
```

Validation results are stored under:

```text
outputs/validation/
```

The automated test suite is stored under:

```text
tests/
```

The current verified test suite contains 88 tests, with the latest full run passing all 88 tests.

---

## 28. End-to-End Mathematical Summary

The methodology can be summarized mathematically.

### Step 1: Semantic masks

Let:

```text
W(u,v) = wall mask
D(u,v) = door mask
O(u,v) = window mask
```

Then:

```text
P(u,v) = W(u,v) ∧ ¬(D(u,v) ∨ O(u,v))
```

where `P` is the paintable-wall mask.

### Step 2: Metric depth

Given MiDaS relative depth `d(u,v)` and calibrated scale `k`:

```text
Z(u,v) = k / d(u,v)
```

for valid depth values.

### Step 3: 3D backprojection

For each valid paintable pixel:

```text
X = (u - cx) Z / fx
Y = (v - cy) Z / fy
Z = Z
```

### Step 4: Triangle area

For each valid triangle:

```text
A_i = 0.5 ||(P2 - P1) × (P3 - P1)||
```

after applying the depth-jump validity condition.

### Step 5: Total paintable area

```text
A = Σ A_i
```

### Step 6: Material quantity

For a material:

```text
Q = (A × C / R) × (1 + W / 100)
```

### Step 7: Material cost

```text
Cost_material = Q × Price
```

### Step 8: Labour

```text
Cost_labour = A × Labour_rate
```

### Step 9: Total

```text
Cost_total =
    Cost_paint
  + Cost_primer
  + Cost_putty
  + Cost_labour
```

---

## 29. Complete Methodology Flow

```text
                    INPUT RGB IMAGE
                           |
             +-------------+-------------+
             |                           |
             v                           v
       WALL SEGMENTATION         OPENING SEGMENTATION
             |                           |
             |                    Door + Window masks
             |                           |
             +-------------+-------------+
                           |
                           v
                  PAINTABLE WALL MASK
                           |
                           v
                   MiDaS RELATIVE DEPTH
                           |
                           v
                SINGLE-ANCHOR CALIBRATION
                           |
                           v
                    METRIC DEPTH MAP
                           |
                           v
                 PINHOLE BACKPROJECTION
                           |
                           v
                      3D WALL POINTS
                           |
                           v
                    TRIANGULAR MESH
                           |
                           v
                 DEPTH-JUMP FILTERING
                           |
                           v
                PAINTABLE AREA (m²)
                           |
             +-------------+-------------+
             |             |             |
             v             v             v
           PAINT         PRIMER        PUTTY
             |             |             |
             +-------------+-------------+
                           |
                           v
                      LABOUR COST
                           |
                           v
                      TOTAL COST
```

---

## 30. Conclusion

The EDIproject methodology converts an indoor RGB image into a physical paintable-wall area by combining semantic segmentation with calibrated monocular depth and 3D camera geometry.

The key methodological distinction is that the system does not treat wall pixels as directly proportional to physical area. Instead, it:

1. Identifies the wall.
2. Removes doors and windows.
3. Estimates relative depth.
4. Converts relative depth to metric depth using a calibrated inverse-scale model.
5. Backprojects valid pixels into 3D.
6. Constructs local triangular surfaces.
7. Rejects triangles crossing excessive depth discontinuities.
8. Sums the remaining triangle areas.
9. Converts the resulting physical area into material quantities and cost.

This provides a complete, modular pipeline from RGB image understanding to metric wall-area and painting-cost estimation under the stated calibration and assumptions, while keeping experimental alternatives separate from the production methodology.
