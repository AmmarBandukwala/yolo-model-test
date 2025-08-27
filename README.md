# Custom YOLO Object Detection

A simplified YOLO (You Only Look Once) implementation for object detection training and inference. This repository provides easy-to-use scripts for training custom object detection models and running inference on images, videos, and webcam feeds.

## 🚀 Features

- **Custom Training**: Train YOLO models on your own datasets
- **Flexible Inference**: Run detection on images, videos, or live webcam
- **Easy Configuration**: Command-line arguments for all parameters
- **GPU Support**: Automatic CUDA detection and utilization
- **Real-time Processing**: Live webcam detection with FPS monitoring
- **Export Results**: Save annotated images/videos and detection JSON

## 📋 Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Webcam (optional, for live detection)

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/AmmarBandukwala/yolo-model-test.git
cd yolo-model-test
```

2. **Create a virtual environment:**
```bash
python -m venv yolo_env
source yolo_env/bin/activate  # Linux/Mac
# or
yolo_env\Scripts\activate     # Windows
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Install PyTorch with CUDA support (recommended):**
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## 📁 Dataset Preparation

### Directory Structure
```
your_dataset/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── labels/
│       ├── image1.txt
│       ├── image2.txt
│       └── ...
├── val/
│   ├── images/
│   │   └── ...
│   └── labels/
│       └── ...
└── data.yaml (optional)
```

### Label Format
Each label file should contain one line per object:
```
class_id center_x center_y width height
```

Where all coordinates are normalized (0-1):
- `class_id`: Integer class identifier (0, 1, 2, ...)
- `center_x, center_y`: Object center coordinates
- `width, height`: Object dimensions

**Example label file (`image1.txt`):**
```
0 0.5 0.3 0.2 0.4
1 0.7 0.6 0.1 0.2
```

### Class Names File
Create a `class_names.txt` file listing your classes:
```
person
bicycle
car
motorcycle
airplane
```

## 🎯 Training

### Basic Training
```bash
python yolo_train.py --data-dir /path/to/dataset --num-classes 5 --epochs 100
```

### Advanced Training Options
```bash
python yolo_train.py \
    --data-dir /path/to/dataset \
    --num-classes 10 \
    --epochs 200 \
    --batch-size 16 \
    --learning-rate 0.001 \
    --img-size 640 \
    --weight-decay 0.0005 \
    --save-freq 10
```

### Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--data-dir` | Path to dataset directory | Required |
| `--num-classes` | Number of object classes | 80 |
| `--epochs` | Training epochs | 100 |
| `--batch-size` | Batch size | 16 |
| `--learning-rate` | Learning rate | 0.001 |
| `--img-size` | Input image size | 640 |
| `--weight-decay` | Weight decay | 0.0005 |
| `--workers` | Data loading workers | 4 |
| `--weights` | Pretrained weights path | None |
| `--save-freq` | Checkpoint save frequency | 10 |

### Training Outputs
- `best_model.pth`: Best model weights (lowest validation loss)
- `checkpoint_epoch_N.pth`: Periodic checkpoints
- Training logs with loss metrics

## 🔍 Inference

### Image Detection
```bash
python yolo_inference.py \
    --weights best_model.pth \
    --source image.jpg \
    --num-classes 5 \
    --class-names class_names.txt \
    --show \
    --output result.jpg
```

### Video Detection
```bash
python yolo_inference.py \
    --weights best_model.pth \
    --source video.mp4 \
    --num-classes 5 \
    --class-names class_names.txt \
    --output output_video.mp4
```

### Webcam Detection
```bash
python yolo_inference.py \
    --weights best_model.pth \
    --source 0 \
    --num-classes 5 \
    --class-names class_names.txt \
    --show
```

### Batch Processing
Process multiple images in a directory:
```bash
for img in /path/to/images/*.jpg; do
    python yolo_inference.py \
        --weights best_model.pth \
        --source "$img" \
        --num-classes 5 \
        --output "results/$(basename "$img")" \
        --save-results
done
```

### Inference Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--weights` | Path to trained model | Required |
| `--source` | Input source (image/video/webcam) | Required |
| `--num-classes` | Number of classes | 80 |
| `--class-names` | Path to class names file | None |
| `--output` | Output path | None |
| `--img-size` | Inference image size | 640 |
| `--conf-thresh` | Confidence threshold | 0.5 |
| `--nms-thresh` | NMS threshold | 0.4 |
| `--show` | Display results | False |
| `--save-results` | Save JSON results | False |

## 📊 Results and Outputs

### Detection Results
The inference script outputs:
- **Console**: Detection count, inference time, FPS
- **Visual**: Annotated images/videos with bounding boxes
- **JSON**: Detailed detection results (with `--save-results`)

### JSON Output Format
```json
[
  {
    "bbox": [x1, y1, x2, y2],
    "confidence": 0.85,
    "class_id": 0,
    "class_name": "person"
  }
]
```

## 🎛️ Configuration Examples

### High Accuracy Setup
```bash
# Training
python yolo_train.py \
    --data-dir dataset \
    --num-classes 10 \
    --epochs 300 \
    --batch-size 8 \
    --img-size 832 \
    --learning-rate 0.0005

# Inference
python yolo_inference.py \
    --weights best_model.pth \
    --source test_image.jpg \
    --img-size 832 \
    --conf-thresh 0.3 \
    --nms-thresh 0.3
```

### Speed Optimized Setup
```bash
# Training
python yolo_train.py \
    --data-dir dataset \
    --num-classes 10 \
    --epochs 100 \
    --batch-size 32 \
    --img-size 416

# Inference
python yolo_inference.py \
    --weights best_model.pth \
    --source webcam \
    --img-size 416 \
    --conf-thresh 0.7
```

## 📈 Performance Tips

### Training Optimization
- **GPU Memory**: Reduce batch size if getting CUDA out of memory
- **Data Loading**: Increase `--workers` for faster data loading
- **Image Size**: Use 416/640/832 for different speed/accuracy trade-offs
- **Learning Rate**: Start with 0.001, reduce if loss plateaus

### Inference Optimization
- **Batch Processing**: Process multiple images together
- **Image Size**: Smaller sizes = faster inference
- **Thresholds**: Higher confidence threshold = fewer false positives
- **GPU**: Ensure CUDA is available for GPU acceleration

## 🐛 Troubleshooting

### Common Issues

**CUDA Out of Memory:**
```bash
# Reduce batch size
python yolo_train.py --batch-size 8  # instead of 16

# Or reduce image size
python yolo_train.py --img-size 416  # instead of 640
```

**No Detections:**
- Lower confidence threshold: `--conf-thresh 0.3`
- Check class names file matches training
- Verify model was trained properly

**Slow Inference:**
- Use GPU: Check `torch.cuda.is_available()`
- Reduce image size: `--img-size 416`
- Close other applications using GPU

**Video Processing Issues:**
```bash
# Install additional video codecs
pip install imageio-ffmpeg
sudo apt install ffmpeg  # Linux
brew install ffmpeg      # macOS
```

### Debug Mode
Add verbose logging:
```bash
python yolo_train.py --data-dir dataset --num-classes 5 --epochs 10 -v
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by the original YOLO papers by Joseph Redmon
- PyTorch community for excellent documentation
- Open source computer vision community