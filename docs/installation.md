# Multi-Modal Breast Cancer Detection

## Installation

### Option 1: Install from requirements
```bash
git clone https://github.com/IH-Arik/Multi-Modal-Breast-Cancer-Detection.git
cd Multi-Modal-Breast-Cancer-Detection
pip install -r requirements.txt
```

### Option 2: Install as package
```bash
git clone https://github.com/IH-Arik/Multi-Modal-Breast-Cancer-Detection.git
cd Multi-Modal-Breast-Cancer-Detection
pip install -e .
```

## Quick Start

1. **Update dataset paths** in `configs/config.yaml`
2. **Train all modalities:**
   ```bash
   python scripts/train_all_modalities.py
   ```
3. **Train specific modality:**
   ```bash
   python scripts/train_all_modalities.py --modality ultrasound
   ```

## Project Structure

```
Multi-Modal-Breast-Cancer-Detection/
├── src/                          # Source code
│   ├── models/                   # Model architectures
│   ├── data/                     # Data loading utilities
│   ├── training/                 # Training functions
│   ├── evaluation/               # Evaluation metrics
│   └── utils/                    # Common utilities
├── configs/                      # Configuration files
├── scripts/                      # Training scripts
├── results/                      # Results and outputs
└── docs/                         # Documentation
```

## Datasets

- **BUSI Dataset**: Ultrasound images
- **CBIS-DDSM Dataset**: Mammography images  
- **BreakHis Dataset**: Histology slides

Update the paths in `configs/config.yaml` to point to your dataset locations.

## Citation

If you use this code, please cite:
```bibtex
@article{breast_cancer_detection_2024,
  title={Multi-Modal Breast Cancer Detection with Deep Learning},
  author={Research Team},
  journal={Medical Imaging with AI},
  year={2024}
}
```
