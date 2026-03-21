# rPPG Cross-Domain BP and HR

This repository contains a notebook-based workflow for remote photoplethysmography (rPPG) experiments on facial video. The current project focuses on:

- loading and scanning the UBFC dataset structure
- detecting facial landmarks and defining multiple facial regions of interest (ROIs)
- extracting RGB traces from ROIs
- generating rPPG signals with CHROM-style processing
- comparing extracted signals against ground-truth physiological traces

The main implementation lives in [rppg.ipynb](./rppg.ipynb).

## Project Status

This is an experimental research notebook rather than a packaged library. The notebook includes several exploration blocks and repeated processing sections as the pipeline evolves.

At the moment, the repository is best suited for:

- reproducing ROI-based rPPG experiments
- testing cross-subject or cross-dataset signal extraction ideas
- visual inspection of ROI masks and signal quality
- using extracted signals as a base for future heart-rate and blood-pressure modeling

## Repository Structure

```text
.
|-- rppg.ipynb           # Main notebook for dataset processing, ROI extraction, and rPPG analysis
|-- requirements.txt     # Python dependencies used by the notebook
|-- roi_results/         # Saved sample outputs (.npz) from ROI/rPPG processing
`-- .gitignore
```

## Features

- GPU-aware runtime check using PyTorch
- UBFC dataset traversal and subject discovery
- MediaPipe Face Mesh landmark detection
- multi-ROI facial segmentation
- ROI visualization on detected faces
- CHROM-based rPPG extraction
- bandpass filtering and signal normalization
- comparison between extracted rPPG and ground-truth PPG/HR traces
- saved intermediate outputs in NumPy `.npz` format

## Dataset

The notebook is currently configured around the **UBFC dataset** and expects a local dataset root path to be set inside the notebook.

Example from the notebook:

```python
root = r"G:\.shortcut-targets-by-id\...\UBFC_DATASET"
```

Before running the notebook, update the dataset path so it points to your local copy.

Expected structure is similar to:

```text
UBFC_DATASET/
|-- DATASET_1/
|   |-- 10-gt/
|   |-- 11-gt/
|   `-- ...
`-- DATASET_2/
    |-- subject1/
    |-- subject10/
    `-- ...
```

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If you are using Jupyter directly:

```bash
pip install notebook
jupyter notebook
```

## Requirements

Core dependencies used in the project:

- `jupyter`
- `ipykernel`
- `numpy`
- `pandas`
- `opencv-python`
- `matplotlib`
- `tqdm`
- `mediapipe`
- `scipy`
- `torch`

## How To Run

1. Open `rppg.ipynb` in Jupyter Notebook or JupyterLab.
2. Update the dataset root path to your local UBFC dataset.
3. Run the environment and GPU-check cells.
4. Run the ROI visualization section to confirm landmarks and ROI placement.
5. Run the dataset processing sections to extract ROI traces and rPPG signals.
6. Inspect plots and saved results in `roi_results/`.

## Outputs

The repository already includes sample saved outputs inside `roi_results/`, such as:

- ROI processing results
- extracted rPPG arrays
- subject-level `.npz` files for selected recordings

These files are useful for quick inspection without rerunning the full pipeline.

## Notes

- The notebook is currently research-oriented and not yet refactored into reusable modules.
- Some sections appear iterative or duplicated because they capture different experiment stages.
- The current repo is centered more on **signal extraction and evaluation** than on a finalized blood-pressure prediction model.

## Future Improvements

- refactor notebook code into Python modules
- add a clean training and evaluation pipeline
- separate preprocessing, ROI extraction, and signal modeling
- add explicit heart-rate and blood-pressure estimation metrics
- add configuration files for dataset paths and experiment settings
- include example plots and benchmark results in the repo

## License

No license file is included yet. If you plan to share or reuse this project publicly, add a license that matches how you want the code and outputs to be used.
