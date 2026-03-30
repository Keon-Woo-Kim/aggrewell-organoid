# aggrewell-organoid

Automated organoid area measurement from AggreWell plate images using a two-stage YOLOv8 pipeline.

## Overview

This tool detects and measures organoid areas in AggreWell microwell plates. It uses two pre-trained YOLOv8 models in sequence:

1. **Well detector** (YOLOv8n) — locates microwells via low-confidence detection + RANSAC grid fitting
2. **Organoid segmentor** (YOLOv8n-seg) — segments the organoid in each well crop

Area is computed as: `organoid_mask_pixels / image_area_pixels * 10000`

Relative volume is computed as: `area ^ 1.5` (spherical approximation)

![Sample output](docs/sample_output.png?raw=true)

## Installation

We recommend using a separate conda environment:

```bash
conda create -n aggrewell python=3.10
conda activate aggrewell
```

Then install from the cloned repository:

```bash
git clone https://github.com/Keon-Woo-Kim/aggrewell-organoid.git
cd aggrewell-organoid
pip install .
```

This installs all dependencies and registers the `aggrewell-organoid` command.

GPU is auto-detected — if CUDA is available, it will be used automatically.

## Usage

Place your plate images (`.jpg`, `.png`, `.tiff`) in a folder, then run:

```bash
aggrewell-organoid path/to/your/images
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--exclude` | none | Well to exclude from analysis (e.g. `r3c0`) |
| `--output` | `results_<foldername>/` next to input | Output directory |

### Examples

```bash
# All wells included
aggrewell-organoid data/my_experiment

# Exclude a specific well
aggrewell-organoid data/my_experiment --exclude r3c0

# Custom output directory
aggrewell-organoid data/my_experiment --output /path/to/output
```

## Output

Results are saved to the output directory:

| File | Description |
|------|-------------|
| `<name>.xlsx` | Sheet "data" (per-image averages) + sheet "data_raw" (all organoids) |
| `<name>.csv` | Per-image averaged areas and volumes |
| `<name>_raw.csv` | Individual organoid areas and volumes |
| `*_organoids.jpg` | Annotated plate images with organoid overlays |

Intermediate outputs:
- `crops/` — cropped well images
- `well_preview/` — grid visualization for QC

## How It Works

### Step 1: Well Detection + Cropping

- YOLOv8n detects well candidates at low confidence, filtered by size
- A 4×6 affine grid is fitted via RANSAC; spurious edge detections are auto-filtered
- Misaligned grids are detected by edge crop aspect ratio and corrected via strip-and-redetect
- Each well is cropped at native resolution with padding

### Step 2: Organoid Segmentation

- YOLOv8n-seg runs instance segmentation on each well crop
- Maximum 1 organoid per well (closest to center if multiple detected)
- Organoid area computed relative to whole image area; exported to CSV/XLSX with overlay images

## Package Structure

```
aggrewell-organoid/
├── pyproject.toml                          # Package config + dependencies
├── README.md
├── .gitignore
├── docs/
│   └── sample_output.jpg                   # Sample output image
└── src/aggrewell_organoid/
    ├── __init__.py
    ├── cli.py                              # CLI entry point (argparse)
    ├── crop_wells.py                       # Well detection + RANSAC grid fitting
    ├── infer_organoids.py                  # Organoid segmentation + export
    └── models/
        ├── well_detector.pt                # YOLOv8n (bbox)
        └── organoid_detector.pt            # YOLOv8n-seg (instance segmentation)
```

## Notes

- Fixed grid: 4 rows x 6 columns (24 wells per image; only AggreWell 24-well plates supported)
- Supported image formats: `.jpg`, `.jpeg`, `.png`, `.tiff`
- Images should be AggreWell plate photos (tested on 2880x2048)
- Runs on CPU by default; GPU is used automatically if available

## Changelog

**v1.2.0** (2026-03-30) — Improved well cropping robustness; relative volume output; organoid model retrained with atypical morphologies

**v1.1.0** (2026-03-03) — Models retrained with hard-case images; max 1 organoid per well enforced; 24-well only

**v1.0.0** (2026-02-21) — Initial release
