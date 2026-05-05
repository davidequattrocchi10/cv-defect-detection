# CV Defect Detection System — Project Summary & Runbook

## Project Goal
Build a production-ready Computer Vision system that detects surface 
defects in industrial leather products using YOLOv8, deployed on Azure 
with a FastAPI backend and MLflow experiment tracking.

## Target Profile
Portfolio project for CV/ML Engineer roles (remote, Europe).
Demonstrates: end-to-end ML pipeline, cloud deployment, MLOps basics.

## Tech Stack
| Component | Technology |
|-----------|-----------|
| Language | Python 3.11 |
| Detection Model | YOLOv8 (Ultralytics) |
| Image Processing | OpenCV |
| Deep Learning | PyTorch |
| API Backend | FastAPI |
| Containerization | Docker |
| Cloud Platform | Azure ML + Azure Container Apps |
| Experiment Tracking | MLflow |
| CI/CD | GitHub Actions |
| Dataset | MVTec AD — Leather category |

---

## Environment Setup (One-Time)

### Prerequisites
- macOS with Homebrew installed
- Python 3.11 via Homebrew
- Docker Desktop (Apple Silicon version)
- Node.js (for Claude Code CLI)
- Git configured with GitHub remote

### First-Time Setup
```bash
# Clone repository
git clone https://github.com/davidequattrocchi10/cv-defect-detection.git
cd cv-defect-detection

# Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install Kaggle CLI
pip install kaggle
# Place kaggle.json at ~/.kaggle/kaggle.json
# chmod 600 ~/.kaggle/kaggle.json
```

### Every Session
```bash
cd ~/Documents/Projects/cv-defect-project
source .venv/bin/activate
```

---

## Dataset

### Source
- Name: MVTec AD — Leather Category
- URL: https://www.mvtec.com/company/research/datasets/mvtec-ad
- License: CC BY-NC-SA 4.0 (non-commercial use only)
- Download: Manual — download leather.tar.xz and extract to data/raw/

### Extract
```bash
tar -xf ~/Downloads/leather.tar.xz -C data/raw/
```

### Structure After Extraction
data/raw/leather/
├── train/
│   └── good/          # 245 defect-free training images (1024x1024)
├── test/
│   ├── good/          # 32 defect-free test images
│   ├── color/         # 19 color defect images
│   ├── cut/           # 19 cut defect images
│   ├── fold/          # 17 fold defect images
│   ├── glue/          # 19 glue defect images
│   └── poke/          # 18 poke defect images
└── ground_truth/
├── color/         # Binary masks for color defects
├── cut/           # Binary masks for cut defects
├── fold/          # Binary masks for fold defects
├── glue/          # Binary masks for glue defects
└── poke/          # Binary masks for poke defects

### Key Properties (from exploration notebook)
- Image dimensions: 1024×1024 pixels, 3 channels (RGB)
- Pixel range: min≈32, max≈255
- Mean pixel value: ≈73, std≈21 (dark leather texture)
- Ground truth masks: binary (0 or 255)
- Class distribution: 245 train / 92 defective test images

---

## Data Pipeline

### Step 1 — Inspect and Validate Dataset
```bash
python src/data/download_dataset.py
```
Verifies all expected folders exist and prints image counts per category.

### Step 2 — Preprocess to YOLO Format
```bash
python src/data/preprocess.py
```

What this script does:
1. Creates output directory structure under data/processed/
2. Resizes all images from 1024×1024 → 640×640 (YOLOv8 optimal input)
3. Copies good images to train split with empty label files
4. Converts ground truth masks → YOLO bounding box annotations
5. Splits defective images 80/20 into test/val splits
6. Writes dataset.yaml required by YOLOv8

### Output Structure
data/processed/
├── images/
│   ├── train/    # 245 images (defect-free)
│   ├── val/      # 19 images (defective, 20% split)
│   └── test/     # 73 images (defective, 80% split)
├── labels/
│   ├── train/    # 245 empty .txt files
│   ├── val/      # 19 YOLO annotation files
│   └── test/     # 73 YOLO annotation files
└── dataset.yaml  # YOLOv8 training configuration

### YOLO Annotation Format
Each .txt label file contains one line per defect:
class_id center_x center_y width height
All values normalized 0-1 relative to image dimensions.

### Class Map
| Class ID | Defect Type |
|----------|-------------|
| 0 | color |
| 1 | cut |
| 2 | fold |
| 3 | glue |
| 4 | poke |

### Verified Counts
train: 245 images, 245 labels ✅
val:    19 images,  19 labels ✅
test:   73 images,  73 labels ✅
total: 337 images

---

## Key Decisions & Lessons Learned

### Why YOLOv8 over Autoencoder/Diffusion?
YOLOv8 produces bounding boxes with confidence scores — directly 
deployable as a REST API. Autoencoders produce anomaly heatmaps that 
require additional postprocessing. For production deployment and 
portfolio clarity, YOLO is the better choice.

### Why 640×640 resize?
YOLOv8's default and optimal input resolution. Chosen for speed/accuracy 
tradeoff. Preserves enough detail for defect detection while reducing 
memory and compute requirements.

### Why separate train/val/test splits?
- Train: model learns from this
- Val: monitor training, tune hyperparameters
- Test: single honest final evaluation (touched once, at the very end)
Mixing splits produces artificially inflated metrics — a common mistake.

### Technical Debt
Dataset download is manual (official MVTec site).
Future fix: store dataset in Azure Blob Storage for automated download.

---

## Project Phases

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 0 | ✅ Complete | Environment setup, project structure, Docker, Git |
| Phase 1 | ✅ Complete | Dataset download, exploration, preprocessing pipeline |
| Phase 2 | 🔄 In Progress | Model training with YOLOv8 |
| Phase 3 | ⬜ Pending | FastAPI application |
| Phase 4 | ⬜ Pending | Docker containerization |
| Phase 5 | ⬜ Pending | Azure deployment |
| Phase 6 | ⬜ Pending | MLflow experiment tracking |
| Phase 7 | ⬜ Pending | CI/CD with GitHub Actions |

---

## Common Commands Reference

```bash
# Activate environment
source .venv/bin/activate

# Run dataset validation
python src/data/download_dataset.py

# Run preprocessing pipeline
python src/data/preprocess.py

# Launch Jupyter for exploration
jupyter notebook

# Build Docker image
docker build -f docker/Dockerfile -t cv-defect-detection .

# Run Docker container
docker run -p 8000:8000 cv-defect-detection

# Launch Claude Code CLI
claude
```

---

## Repository Structure
cv-defect-detection/
├── data/                   # Not in Git
│   ├── raw/leather/        # Original MVTec dataset
│   └── processed/          # YOLO-formatted dataset
├── models/                 # Saved weights — not in Git
├── notebooks/              # Exploration notebooks only
│   └── 01_data_exploration.ipynb
├── src/
│   ├── data/
│   │   ├── download_dataset.py
│   │   └── preprocess.py
│   ├── training/           # Phase 2
│   ├── inference/          # Phase 3
│   └── utils/
├── app/                    # Phase 3 — FastAPI
├── tests/                  # Phase 6
├── docker/
│   ├── Dockerfile
│   └── .dockerignore
├── .github/workflows/      # Phase 7 — CI/CD
├── CLAUDE.md
├── SUMMARY.md
├── requirements.txt
└── README.md