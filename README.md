# YOLOv8 with CBAM Integration

This project compares the performance of the standard YOLOv8 model against a YOLOv8 model enhanced with Convolutional Block Attention Module (CBAM) for acne detection in skin images.

## Project Structure

```
.
├── cbam.py                # CBAM module implementation
├── compare_models.py      # Script to compare model performance
├── data.yaml              # Dataset configuration
├── dataset/               # Dataset directory
├── run_pipeline.py        # Main script to run the entire pipeline
├── train_basic.py         # Script to train the standard YOLOv8 model
├── train_cbam.py          # Script to train the CBAM-enhanced YOLOv8 model
├── yolov8_cbam.py         # YOLOv8 model with CBAM integration
└── yolov8n.pt             # Pre-trained YOLOv8n model weights
```

## Requirements

- Python 3.8+
- PyTorch
- Ultralytics YOLOv8
- matplotlib
- Other dependencies (installed via pip)

## Installation

```bash
pip install ultralytics torch torchvision matplotlib
```

## Usage

### Run the Complete Pipeline

To run the entire pipeline (training both models and comparing them):

```bash
python run_pipeline.py
```

### Advanced Options

Skip training specific models:

```bash
# Skip training the basic model
python run_pipeline.py --skip-basic

# Skip training the CBAM model
python run_pipeline.py --skip-cbam

# Only run the comparison (skip all training)
python run_pipeline.py --compare-only
```

### Train Individual Models

Train standard YOLOv8:

```bash
python train_basic.py
```

Train YOLOv8 with CBAM:

```bash
python train_cbam.py
```

### Compare Models

Compare the trained models:

```bash
python compare_models.py
```

## Results

After running the comparison, results will be available in the `comparison_results` directory:

- `metrics_comparison.json`: JSON file containing performance metrics
- `metrics_comparison.png`: Visual comparison of model performance

The training results for each model can be found in:

- Basic YOLOv8: `runs/train_basic/`
- YOLOv8 with CBAM: `runs/train_cbam/`

## CBAM Architecture

Convolutional Block Attention Module (CBAM) consists of two sequential sub-modules:

1. **Channel Attention Module**: Focuses on 'what' is meaningful in the input features
2. **Spatial Attention Module**: Focuses on 'where' is meaningful in the input features

This attention mechanism helps the model focus on important features and suppress less useful ones.

## Reference

- YOLOv8: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- CBAM: [CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521)
