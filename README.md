# WoodYOLO

## Overview
WoodYOLO is a specialized object detection model designed for identifying vessel elements in microscopic wood images. It features:
- Single-class optimization with high recall focus
- Clean, minimalist codebase for easy adaptation
- Support for both PyTorch and ONNX inference
- Flexible architecture with modular components

While primarily optimized for wood microscopy, the model can be used for other datasets. This repository includes a car detection example to demonstrate its usage.

## Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)

## Installation

### Basic Installation
```bash
pip install torch torchvision numpy pandas tqdm Pillow
```

### ONNX Support (Optional)
```bash
pip install onnxruntime-gpu onnx
```

## Data Format

### Dataset Structure
Training and validation data should be provided as CSV files with the following format:

| Column | Description |
|--------|-------------|
| image_path | Path to the image file |
| x0, y0, x1, y1 | Bounding box coordinates (absolute pixels) |
| class_id | Class identifier (currently unused) |

### Example CSV Format
```csv
image_path,x0,y0,x1,y1,class_id
images/001.jpg,100,200,300,400,0
images/002.jpg,500,100,700,300,1
```

## Usage

### Training
Start training using the `train.py` script:

```bash
python train.py \
    --training_file data/train_boxes.csv \
    --validation_file data/val_boxes.csv \
    --backbone vgg11_bn \
    --neck yolov7 \
    --batch_size 8 \
    --epochs 150 \
    --device "cuda:0" \
    --mosaic_prob 0.5
```

For a complete list of training parameters:
```bash
python train.py --help
```

### Inference
Refer to `example_prediction.ipynb` for detailed inference examples and code.

## Quick Start: Car Detection Example

1. Download the [Car Object Detection](https://www.kaggle.com/datasets/sshikamaru/car-object-detection) dataset
2. Extract the dataset:
   ```bash
   unzip archive.zip
   ```

3. Create and run the following script to prepare train/validation splits:
   ```python
   import pandas as pd
   from sklearn.model_selection import GroupShuffleSplit
   
   df = pd.read_csv("data/train_solution_bounding_boxes (1).csv")
   df = df.rename(columns={
       "xmin": "x0", 
       "ymin": "y0", 
       "xmax": "x1", 
       "ymax": "y1"
   })
   df["image_path"] = "data/training_images/" + df["image"]
   df["class_id"] = 0
   
   train_idx, val_idx = next(GroupShuffleSplit().split(df, groups=df["image"]))
   df.iloc[train_idx].to_csv("data/train_boxes.csv", index=False)
   df.iloc[val_idx].to_csv("data/val_boxes.csv", index=False)
   ```

4. Train the model:
   ```bash
   python train.py \
       --training_file=data/train_boxes.csv \
       --validation_file=data/val_boxes.csv \
       --image_width=512 \
       --image_height=512 \
       --device="cuda:0" \
       --mosaic_prob=0.3
   ```

5. Run inference using `example_prediction.ipynb`

## Architecture

WoodYOLO employs a modular architecture with three main components:

### Backbone
- YOLOv7-tiny
- Any timm-supported model

### Neck
- Modified PANet (adapted from YOLOv7)
- YOLOX neck (alternative option)

### Head
- Simplified YOLOv7 detection head optimized for bounding box prediction

The architecture can be extended by adding custom modules in the `model` directory.

## Citation
If you use WoodYOLO in your research, please cite our paper:

```bibtex
@article{woodyolo,
  title = {WoodYOLO: A Novel Object Detector for Wood Species Detection in Microscopic Images},
  volume = {15},
  ISSN = {1999-4907},
  url = {http://dx.doi.org/10.3390/f15111910},
  DOI = {10.3390/f15111910},
  number = {11},
  journal = {Forests},
  publisher = {MDPI AG},
  author = {Nieradzik, Lars and Stephani, Henrike and Sieburg-Rockel, Jördis and Helmling, Stephanie and Olbrich, Andrea and Wrage, Stephanie and Keuper, Janis},
  year = {2024},
  month = oct,
  pages = {1910}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.