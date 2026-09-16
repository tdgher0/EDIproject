# EDIproject Results

## 1. Overview

This document presents the experimental, validation, and verification results obtained during development of EDIproject.

The results are reported separately for:

- Wall segmentation
- Wall-area measurement
- Opening segmentation
- Opening-area measurement
- MiDaS metric-depth calibration feasibility
- Error analysis
- Automated software tests
- Streamlit application verification

The purpose of this document is to report the measured evidence for the implemented system without presenting experimental results as universal real-world accuracy.

---

## 2. Final Production Configuration

The final production configuration used for the reported wall-area pipeline is:

```text
Wall model:
U-Net + ResNet34
Checkpoint:
models/checkpoints/best_model.pt
Wall threshold:
0.80

Opening model:
U-Net + ResNet34
Checkpoint:
models/checkpoints/opening_augmented_best_model.pt

Opening classes:
0 = background
1 = door
2 = window

Depth:
MiDaS relative depth

Metric calibration:
Single-anchor inverse-scale

Area measurement:
Pinhole backprojection
+
Triangular 3D mesh

Depth-jump threshold:
0.025
```

Experimental alternatives were retained separately and were not substituted into the final production pipeline without controlled comparison.

---

## 3. Wall Segmentation Results

### 3.1 Baseline training

The baseline wall-segmentation model used:

```text
Architecture: U-Net with ResNet34 encoder
Input size: 512 × 512
Batch size: 4
Epochs: 20
Learning rate: 1e-4
Optimizer: Adam
Loss: BCEWithLogits + Dice
```

The best validation IoU during baseline training was:

```text
0.5723
```

at approximately epoch 18.

This value is a model-training validation metric and should be distinguished from the downstream physical wall-area validation results.

---

## 4. Wall Threshold Evaluation

The production segmentation threshold was selected by evaluating its effect on the downstream wall-area measurement.

The evaluated probability thresholds were:

```text
0.50
0.60
0.70
0.80
```

For the 91 wall-containing validation images, the approximate wall-area results were:

| Threshold | Mean absolute area error (m²) | Median absolute error (m²) |
|---:|---:|---:|
| 0.50 | 0.5838 | 0.1968 |
| 0.60 | 0.5282 | 0.1818 |
| 0.70 | 0.4920 | 0.1698 |
| 0.80 | **0.4822** | 0.1923 |

The threshold `0.80` produced the lowest mean absolute wall-area error among the evaluated thresholds and was therefore selected for production.

The selection was based on the physical measurement objective rather than segmentation metrics alone.

---

## 5. Wall Segmentation Metrics at Production Threshold

Using the exact production-style segmentation evaluation on the 91 wall-containing validation images, the results at threshold `0.80` were approximately:

```text
Mean IoU:       0.7093
Mean Dice:      0.8069
Precision:      0.8559
Recall:         0.8022
```

For comparison, the same evaluation across thresholds produced:

| Threshold | IoU | Dice | Precision | Recall |
|---:|---:|---:|---:|---:|
| 0.50 | 0.7125 | 0.8144 | 0.7965 | 0.8652 |
| 0.60 | 0.7154 | 0.8159 | 0.8153 | 0.8496 |
| 0.70 | 0.7150 | 0.8140 | 0.8349 | 0.8293 |
| 0.80 | 0.7093 | 0.8069 | 0.8559 | 0.8022 |

This demonstrates an important project result: the threshold with the best segmentation metric was not necessarily the threshold with the best downstream physical-area error.

---

## 6. Full Wall-Area Validation

A validation run was performed for indices 1–100.

The run produced:

```text
99 CSV rows
```

because index 95 failed validation due to:

```text
wall_mask contains no wall pixels
```

The validation set contained:

```text
8 ground-truth empty-wall images, of which 4 were correctly predicted as empty.
```

The main wall-area statistics for the 91 wall-containing valid cases were:

```text
Mean absolute error:       0.4821 m²
Median absolute error:     0.1923 m²
Mean percentage error:     21.54%
Median percentage error:    6.44%
```

The corresponding validation output is stored in:

```text
outputs/validation/wall_area_validation.csv
```

These statistics apply to the evaluated validation subset and should not be interpreted as an end-to-end accuracy guarantee for arbitrary indoor scenes.

---

## 7. Wall-Area Error Analysis

The validation analysis showed a tendency toward wall over-segmentation.

For the 91 wall-containing cases, the approximate pixel statistics were:

```text
Mean predicted wall pixels: 72,498
Mean ground-truth wall pixels: 67,042

Mean extra predicted pixels: 11,257
Mean missed pixels:           5,801

Total extra pixels:  1,024,351
Total missed pixels:   527,872

Extra / missed ratio:  1.9405
```

The ratio greater than 1 indicates that the evaluated model produced more extra wall pixels than missed wall pixels.

This is consistent with the observed tendency of the wall model to include visually similar non-wall regions.

---

## 8. Worst Wall-Area Cases

Several difficult images produced substantially larger errors.

Representative worst cases from the validation analysis include:

| Index | IoU | Extra pixels | Missed pixels | Area error (m²) | Percentage error |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.1735 | 80,442 | 3,842 | 1.2221 | 332.86% |
| 30 | 0.2956 | 66,768 | 4,380 | 11.4996 | 196.87% |
| 27 | 0.3329 | 34,942 | 283 | 5.7000 | 202.48% |
| 54 | 0.4146 | 30,706 | 1,207 | 0.5487 | 115.90% |
| 57 | 0.5568 | 28,671 | 2,681 | 2.0401 | 49.56% |

Visual diagnostics indicated over-segmentation around objects and structures such as:

- Refrigerators
- Microwaves
- Cabinets
- Windows
- Blinds
- Other visually similar surfaces

These cases show that segmentation quality remains a major practical limitation of the wall-area system.

---

## 9. Segmentation Error Versus Geometry Error

The project separately evaluated the wall-area geometry using the predicted wall mask and a reference ground-truth wall mask.

For the 91 wall-containing cases:

```text
Predicted-mask area MAE: approximately 0.5838 m²
Ground-truth-mask area MAE: approximately 0.0003 m²
```

The corresponding geometry error reduction when using the reference wall mask was approximately:

```text
99.95%
```

This experiment is important because it isolates the source of error.

It indicates that, on the evaluated subset, the implemented metric-depth and 3D geometry calculation contributed very little error when the wall region itself was correct.

Therefore, wall segmentation quality is a dominant contributor to the observed wall-area error.

---

## 10. Opening Segmentation Results

The production opening model is:

```text
U-Net + ResNet34
```

using:

```text
models/checkpoints/opening_augmented_best_model.pt
```

The augmented training strategy used:

```text
50% horizontal flipping
Random rotation from -10° to +10°
```

The best augmented validation results were:

```text
Background IoU: 0.9637
Door IoU:       0.3487
Window IoU:     0.3807
```

The augmented model improved validation IoU for both opening classes compared with the original non-augmented opening model.

---

## 11. Opening Segmentation Test Results

The production-style opening evaluation produced:

| Class | IoU | Dice | Precision | Recall |
|---|---:|---:|---:|---:|
| Background | 0.9673 | 0.9834 | 0.9751 | 0.9918 |
| Door | 0.2594 | 0.4120 | 0.7178 | 0.2889 |
| Window | 0.3472 | 0.5155 | 0.5958 | 0.4542 |

Overall:

```text
Mean IoU:   0.5247
Mean Dice:  0.6369
```

The test set contained approximately:

```text
5050 images
1307 images containing door class
1350 images containing window class
```

Opening pixels represented approximately 4% of the total pixels.

The relatively low door and window IoU compared with background IoU shows that opening segmentation remains an important uncertainty in the paintable-area calculation.

---

## 12. Opening-Area Validation

Opening-area validation was performed using the official SUN RGB-D depth conversion and associated metadata.

For valid ground-truth/prediction pairs, the measured results were:

| Opening type | Valid pairs | MAE (m²) | Median absolute error (m²) |
|---|---:|---:|---:|
| Door | 658 | 0.6201 | 0.2461 |
| Window | 782 | 0.6260 | 0.2904 |
| Combined | 1482 | 0.6636 | 0.2970 |

These results are reported for valid evaluation pairs only.

Consequently, they should be interpreted together with the missing-pair and segmentation-coverage information. They should not be treated as a complete recall-aware end-to-end opening-detection performance measure.

---

## 13. Paintable-Area Opening Effect

A separate ground-truth paintable-area experiment evaluated the effect of removing openings from wall area.

Across the evaluated 19 images:

```text
Mean wall area:          4.1003 m²
Mean paintable area:     3.8709 m²
Mean opening effect:     0.2294 m²
```

The opening effect was approximately:

```text
5.6%
```

of the mean wall area in this evaluated sample.

This demonstrates why door and window removal is necessary for painting-area estimation.

The result is sample-specific and should not be interpreted as a universal proportion of openings in indoor walls.

---

## 14. MiDaS Calibration Feasibility

A feasibility experiment evaluated the production single-anchor inverse-scale calibration against an inverse-affine calibration fitted using ground-truth depth.

The results were:

| Image index | Sensor | Inverse-scale MAE (m) | Inverse-affine MAE (m) |
|---:|---|---:|---:|
| 57 | kv2 | 0.720 | 0.252 |
| 71 | kv2 | 1.045 | 0.102 |
| 965 | kv2 | 1.634 | 0.239 |
| 2020 | kv1 | 0.849 | 0.128 |
| 2486 | kv1 | 0.546 | 0.265 |
| 2642 | kv1 | 0.071 | 0.037 |
| 3145 | xtion | 6.738 | 0.623 |
| 3499 | xtion | 0.950 | 0.064 |
| 4502 | — | 0.053 | 0.019 |
| 4854 | realsense | 0.788 | 0.228 |
| 4926 | — | 0.478 | 0.141 |
| 5059 | — | 0.336 | 0.045 |

The inverse-affine experiment generally produced lower depth MAE.

However, it used ground-truth depth to fit the calibration relationship. This is not equivalent to the intended deployment process.

Therefore, the inverse-affine method remains a research/experimental result, while the production pipeline retains explicit single-anchor inverse-scale calibration.

---

## 15. Tversky Wall Model Experiment

A Tversky-loss wall checkpoint was evaluated as an experimental alternative.

The fair 100-image benchmark produced approximately:

```text
Tversky wall model MAE:  0.2566 m²
Baseline wall model MAE: 0.2642 m²
```

The Tversky model therefore showed a modest improvement in the evaluated benchmark.

However, the baseline wall model remains the primary production model because production selection considers the complete validated pipeline and the established production configuration rather than replacing the system based on a single experimental comparison.

The experimental checkpoint is:

```text
models/checkpoints_tversky/best_model.pt
```

---

## 16. Streamlit Application Verification

The Streamlit application was successfully executed and visually verified.

The application demonstrated the complete processing sequence:

```text
Input image
→ Wall segmentation
→ Opening segmentation
→ Paintable wall
→ MiDaS relative depth
→ Metric depth
→ 3D explanation
→ Area and cost results
```

A successful application run displayed results including:

```text
Paintable area
Total cost
Labour cost
Paint quantity
Paint cost
Primer quantity
Primer cost
Putty quantity
Putty cost
```

One verified application demonstration using `img-000004.jpg` produced:

```text
Paintable area: 8.43 m²
Total cost: ₹278.80
Labour: ₹168.60
Paint: 1.48 L
Paint cost: ₹44.26
Primer: 0.59 L
Primer cost: ₹10.62
Putty: 2.21 kg
Putty cost: ₹55.25
```

These values are specific to that demonstration image and the configured calibration and estimation assumptions.

They should not be compared directly with results from a different input image.

---

## 17. Independent Pipeline Geometry Check

The production pipeline output was independently checked by recomputing the 3D area directly from the returned paintable mask and metric-depth output.

For the tested pipeline run:

```text
Pipeline paintable area:
11.97907829284668 m²

Independent geometry recomputation:
11.97907829284668 m²

Difference:
0.0 m²
```

This confirms that the reported pipeline area is consistent with the implemented geometry calculation for that tested case.

The test image was:

```text
data/raw/sunrgbd/img-000050.jpg
```

with the tested calibration parameters:

```text
Calibration factor: 1059
fx: 529.5
fy: 529.5
cx: 365
cy: 265
```

This is an implementation-consistency check, not an accuracy measurement against independent physical ground truth.

---

## 18. Automated Test Results

The project includes automated tests covering:

- Calibration
- Depth
- Detection
- Estimation
- Image processing
- Measurement
- Opening-area measurement
- Pipeline behaviour
- Segmentation

The latest full test run produced:

```text
88 tests collected
88 passed
```

Execution time was approximately:

```text
10.90 seconds
```

This provides software-level verification of the implemented modules and pipeline behaviour.

Passing automated tests do not by themselves establish real-world measurement accuracy.

---

## 19. Key Findings

The experimental and validation results lead to several important findings.

### Finding 1: Physical area is a better threshold-selection objective

The threshold comparison showed that the segmentation threshold giving the best segmentation metric was not necessarily the one producing the lowest physical wall-area error.

The production threshold of `0.80` was therefore selected using downstream area error.

### Finding 2: Segmentation is a major source of area error

The geometry-isolation experiment showed approximately `0.0003 m²` MAE when the reference wall mask was used, compared with approximately `0.5838 m²` for the predicted wall-mask evaluation.

This indicates that improving wall segmentation has substantial potential to improve the final area estimate.

### Finding 3: Opening segmentation remains difficult

Background segmentation was strong, while door and window IoU were substantially lower.

This can affect the final paintable area because missed openings may remain incorrectly classified as paintable wall.

### Finding 4: Relative depth requires calibration

MiDaS provides relative depth, so metric wall-area estimation requires a calibration relationship.

The production implementation uses explicit single-anchor inverse-scale calibration.

### Finding 5: Ground-truth-fitted depth calibration is not directly deployable

The inverse-affine experiment produced lower depth error on the evaluated samples, but it relied on ground-truth depth.

It is therefore retained as an experimental feasibility result rather than being used as the production calibration method.

### Finding 6: Cost estimation is configurable

The final cost is driven by the measured area and configurable assumptions for coverage, coats, wastage, prices, and labour.

Therefore, the cost model can be adapted without changing the computer-vision measurement system.

---

## 20. Overall Performance Summary

The principal measured results are summarized below.

| Component | Result |
|---|---|
| Wall baseline best validation IoU | 0.5723 |
| Production wall threshold | 0.80 |
| Wall-area MAE, 91 wall-containing cases | 0.4821 m² |
| Wall-area median absolute error | 0.1923 m² |
| Wall-area mean percentage error | 21.54% |
| Wall-area median percentage error | 6.44% |
| Production-threshold wall IoU | 0.7093 |
| Production-threshold wall Dice | 0.8069 |
| Door IoU | 0.2594 |
| Window IoU | 0.3472 |
| Opening mean IoU | 0.5247 |
| Opening mean Dice | 0.6369 |
| Door-area MAE | 0.6201 m² |
| Window-area MAE | 0.6260 m² |
| Combined opening-area MAE | 0.6636 m² |
| Ground-truth-mask geometry MAE | ~0.0003 m² |
| Tversky experimental wall MAE | 0.2566 m² |
| Baseline fair-benchmark wall MAE | 0.2642 m² |
| Automated tests | 88 / 88 passed |

---

## 21. Limitations of the Results

The reported results should be interpreted within the evaluation conditions used during development.

1. The wall-area results are based on a controlled validation subset.
2. The opening-area results are based on valid evaluation pairs.
3. Dataset annotations may contain incomplete or imperfect wall labels.
4. MiDaS depth remains relative and depends on the calibration procedure.
5. Camera intrinsics are camera dependent.
6. Segmentation errors can dominate the final physical-area error.
7. Opening segmentation has lower performance than background segmentation.
8. Cost results depend on configurable assumptions rather than fixed market prices.
9. The system measures visible paintable surfaces and does not automatically infer hidden surfaces outside the image.
10. The results do not constitute a guarantee of performance for arbitrary real-world rooms.

---

## 22. Final Production Result

The final EDIproject production methodology is:

```text
                    RGB IMAGE
                        |
                        v
              U-Net ResNet34
              Wall Segmentation
                 threshold 0.80
                        |
                        +----------------+
                        |                |
                        v                v
                Wall mask       Opening U-Net
                                  ResNet34
                                     |
                              Door + Window
                                  masks
                        |                |
                        +-------+--------+
                                |
                                v
                       Paintable wall mask
                                |
                                v
                       MiDaS relative depth
                                |
                                v
                  Single-anchor inverse scale
                                |
                                v
                         Metric depth
                                |
                                v
                     Pinhole backprojection
                                |
                                v
                         3D wall points
                                |
                                v
                       Triangular mesh
                                |
                                v
                     Depth-jump filtering
                          threshold 0.025
                                |
                                v
                     Paintable area (m²)
                                |
                                v
                 +--------------+--------------+
                 |              |              |
                 v              v              v
               Paint         Primer          Putty
                 |              |              |
                 +--------------+--------------+
                                |
                                v
                           Labour cost
                                |
                                v
                           Total cost
```

The validated implementation demonstrates that the project successfully connects computer-vision segmentation, monocular depth, metric calibration, 3D surface measurement, and configurable painting-cost estimation into a single working pipeline.

The principal remaining technical limitation is segmentation quality, particularly wall over-segmentation and door/window detection. The validation evidence indicates that improving segmentation is likely to provide greater benefit to final wall-area accuracy than changing the already-consistent 3D geometry calculation.
