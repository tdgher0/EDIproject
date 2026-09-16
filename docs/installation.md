# EDIproject Installation Guide

## 1. Overview

This document describes how to set up and run EDIproject on a Windows system.

EDIproject is a Python-based computer-vision application for automated wall-area measurement and painting-cost estimation from indoor RGB images.

The production pipeline uses:

- Wall segmentation using U-Net with a ResNet34 encoder
- Door/window segmentation using a second U-Net with a ResNet34 encoder
- MiDaS relative depth estimation
- Single-anchor inverse-scale metric depth calibration
- Pinhole-camera backprojection and triangular-mesh wall-area measurement
- Depth-jump filtering
- Paint, primer, putty, labour, and total-cost estimation
- Streamlit for the user interface

---

## 2. System Requirements

The current project was developed and tested on:

- Windows
- Python 3.14.6
- NVIDIA RTX 4060 Laptop GPU
- CUDA-enabled PyTorch
- PowerShell

A GPU is recommended because the segmentation and depth-estimation models are computationally intensive.

The project can contain experimental components that have additional dependencies, but the production application uses the dependencies listed in `requirements.txt`.

---

## 3. Project Structure

The main project structure is:

```text
EDIproject/
├── configs/
├── data/
├── docs/
├── models/
├── notebooks/
├── outputs/
├── scripts/
├── src/
├── tests/
├── SUNRGBDtoolbox/
├── requirements.txt
└── README.md
```

The Python virtual environment is normally kept separately as:

```text
venv/
```

Generated caches and temporary files should not be considered part of the core application structure.

---

## 4. Create the Python Environment

Open PowerShell and move to the project directory:

```powershell
cd C:\EDIproject
```

Create a virtual environment:

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

After activation, PowerShell should show `(venv)` at the beginning of the command prompt.

Example:

```text
(venv) PS C:\EDIproject>
```

If PowerShell does not allow the activation script to run, the project's Python environment or system execution-policy configuration should be checked rather than changing security settings unnecessarily.

---

## 5. Install Project Dependencies

With the virtual environment activated, install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

The current production requirements are:

```text
torch
torchvision
segmentation-models-pytorch
numpy
pillow
opencv-python
PyYAML
streamlit
```

`pytest` is used for development/testing but is not required by the production application requirements.

Experimental components such as YOLO may require additional packages when they are specifically used.

---

## 6. Verify the Python Environment

Check the Python version:

```powershell
python --version
```

Check the installed PyTorch version:

```powershell
python -c "import torch; print(torch.__version__)"
```

Check CUDA availability:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

If CUDA is available, the result should be:

```text
True
```

The project was developed with CUDA-enabled PyTorch.

---

## 7. Required Model Checkpoints

The production pipeline requires the trained segmentation checkpoints.

### Wall segmentation

Primary wall checkpoint:

```text
models/checkpoints/best_model.pt
```

This is the production wall-segmentation model.

The production wall threshold is:

```text
0.80
```

### Opening segmentation

Preferred production checkpoint:

```text
models/checkpoints/opening_augmented_best_model.pt
```

It predicts:

```text
0 = background
1 = door
2 = window
```

The original opening checkpoint is retained as an experimental/reference model:

```text
models/checkpoints/opening_best_model.pt
```

The Tversky wall model is also experimental:

```text
models/checkpoints_tversky/best_model.pt
```

Experimental checkpoints should not replace the production checkpoints without a controlled comparison.

---

## 8. Dataset and Data Files

The project contains data-processing and validation scripts for the SUN RGB-D dataset.

The project uses SUN RGB-D images and associated depth/metadata for development and validation.

The SUN RGB-D dataset is required for training and validation, but is not required for normal inference through the Streamlit application when the trained checkpoints are already available.

Dataset preparation scripts are located under:

```text
scripts/
```

Relevant project modules include:

```text
src/segmentation/dataset.py
src/segmentation/opening_dataset.py
```

Validation outputs are stored under:

```text
outputs/validation/
```

---

## 9. Running the Streamlit Application

Activate the virtual environment:

```powershell
cd C:\EDIproject
.\venv\Scripts\Activate.ps1
```

Start the Streamlit application:

```powershell
streamlit run src\app\app_final.py
```

Streamlit will start a local web server and display the application address in the terminal.

Open the displayed local address in a web browser.

---

## 10. Using the Application

The application accepts an indoor RGB image.

The user also provides the camera calibration information required for metric wall-area estimation.

The current application provides inputs for:

- Calibration factor
- Camera focal length `fx`
- Camera focal length `fy`
- Principal point `cx`
- Principal point `cy`

The default/example values currently shown by the application are:

```text
Calibration factor: 1059.0
fx: 529.5
fy: 529.5
cx: 365.0
cy: 265.0
```

These values are example/default values from the current project configuration. They should not be interpreted as universal values for every camera.

Camera intrinsics and the metric calibration factor are camera/calibration dependent. For deployment with another camera or calibration setup, these values must be determined through the project's calibration procedure.

---

## 11. Production Processing Pipeline

After an image and calibration parameters are supplied, the application follows this sequence:

```text
Input RGB image
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
Metric depth calibration
       |
       v
3D wall-area measurement
       |
       v
Paint / primer / putty / labour estimation
       |
       v
Total cost
```

The paintable-wall mask is computed as:

```text
paintable_mask =
    wall_mask AND NOT(door_mask OR window_mask)
```

---

## 12. Main Production Entry Point

The production pipeline is implemented in:

```text
src/pipeline/pipeline.py
```

The pipeline accepts:

- RGB image path
- Calibration factor
- Camera intrinsics
- Wall-model checkpoint
- Opening-model checkpoint
- Configuration path
- Maximum allowed depth jump

The production default depth-jump threshold is:

```text
0.025
```

The pipeline returns the measured paintable area and material/cost estimates together with intermediate processing results.

---

## 13. Configuration

Material and labour assumptions are stored in:

```text
configs/estimation_config.yaml
```

The current example configuration contains:

```yaml
paint:
  coverage_m2_per_litre: 12.0
  price_per_litre: 30.0
  wastage_percent: 5.0
  coats: 2

primer:
  coverage_m2_per_litre: 15.0
  price_per_litre: 18.0
  wastage_percent: 5.0
  coats: 1

putty:
  coverage_m2_per_kg: 4.0
  price_per_kg: 25.0
  wastage_percent: 5.0
  coats: 1

labour:
  rate_per_m2: 20.0
```

These values are illustrative assumptions used by the project and should be replaced with appropriate real-world material coverage, prices, wastage assumptions, coat counts, and labour rates when the system is deployed for actual estimation.

---

## 14. Running Tests

The project contains automated tests under:

```text
tests/
```

After activating the virtual environment, run:

```powershell
python -m pytest -v
```

The current verified test suite contains 88 tests, with the latest full run passing all 88 tests.

A successful result ends with a summary similar to:

```text
88 passed
```

Tests cover areas including:

- Calibration
- Depth processing
- Detection
- Estimation
- Image processing
- Measurement
- Opening-area measurement
- Pipeline behaviour
- Segmentation

---

## 15. Important Development Note

The project contains both production and experimental components.

Production components should be kept stable while experimental approaches are evaluated separately.

Examples of experimental components include:

- Tversky-loss wall segmentation
- Alternative MiDaS calibration experiments
- ArUco-based calibration
- Vanishing-point calibration
- YOLO-based object detection/removal
- Inpainting-based object removal

These experiments should not automatically be inserted into the production pipeline.

---

## 16. Calibration Considerations

The metric depth stage uses a single-anchor inverse-scale relationship.

Conceptually:

```text
metric_depth = k / MiDaS_relative_depth
```

where the scale factor is obtained from a known reference depth.

The project therefore requires appropriate calibration information for physically meaningful wall-area measurement.

The calibration factor is not a universal constant.

For a different camera or calibration setup, the calibration factor and camera intrinsics should be determined again.

---

## 17. Troubleshooting

### Python command is not found

Verify that Python is installed:

```powershell
python --version
```

If the command is unavailable, install/configure Python before creating the virtual environment.

### Virtual environment does not activate

Use:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks the script, inspect the local execution-policy configuration rather than disabling security controls globally.

### PyTorch cannot load

Check:

```powershell
python -c "import torch; print(torch.__version__)"
```

If Windows reports an application-control or Code Integrity policy error while loading a PyTorch DLL, this is an operating-system security-policy issue rather than evidence that the project code itself is incorrect.

Do not blindly reinstall or replace the working CUDA-enabled PyTorch environment.

### Streamlit does not start

Verify that the environment is activated and Streamlit is installed:

```powershell
python -c "import streamlit; print(streamlit.__version__)"
```

Then run:

```powershell
streamlit run src\app\app_final.py
```

### Model checkpoint cannot be found

Confirm that the required files exist:

```text
models/checkpoints/best_model.pt
models/checkpoints/opening_augmented_best_model.pt
```

Run the application from the project root:

```text
C:\EDIproject
```

This keeps the project's relative paths consistent.

---

## 18. Recommended Installation Order

For a fresh setup, use this order:

1. Obtain the EDIproject source directory.
2. Open PowerShell in `C:\EDIproject`.
3. Create the Python virtual environment.
4. Activate the virtual environment.
5. Install `requirements.txt`.
6. Verify Python and PyTorch.
7. Verify the required model checkpoints.
8. Run the automated tests.
9. Start the Streamlit application.
10. Provide an RGB image and appropriate calibration parameters.

---

## 19. Quick Start

### First-time setup

For a fresh project environment, use:

```powershell
cd C:\EDIproject
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -v
streamlit run src\app\app_final.py
```

The dependency installation step only needs to be performed when setting up the environment or when the project dependencies change.

### Normal subsequent use

Once the environment and dependencies are already prepared:

```powershell
cd C:\EDIproject
.\venv\Scripts\Activate.ps1
streamlit run src\app\app_final.py
```

---

## 20. Summary

EDIproject is currently organized as a Windows/Python computer-vision application with a Streamlit interface.

The production system requires:

- A configured Python environment
- The production segmentation checkpoints
- The production Python dependencies
- Appropriate camera calibration parameters
- An indoor RGB image

The complete processing chain combines semantic segmentation, opening removal, relative-depth estimation, metric calibration, 3D wall-area measurement, and material/cost estimation.

The project includes automated tests and validation outputs to support development and evaluation.
