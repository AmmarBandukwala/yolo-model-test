# Gas Leak Detection with OpenCV

An experimental computer vision system for detecting and analyzing gaseous leaks using multiple OpenCV techniques. This project demonstrates the capabilities and limitations of using standard cameras for gas detection.

## ⚠️ **IMPORTANT DISCLAIMER**

**THIS IS AN EXPERIMENTAL PROJECT FOR RESEARCH AND EDUCATIONAL PURPOSES ONLY.**

- **DO NOT USE** for safety-critical gas detection
- **NOT A REPLACEMENT** for certified gas detection equipment
- **MANY DANGEROUS GASES** (methane, CO, propane) are invisible to cameras
- **FALSE POSITIVES/NEGATIVES** are common due to environmental factors
- **FOR REAL GAS SAFETY**: Use professional gas detectors and sensors

## 🔬 **Detection Methods**

| Method | Best For | Limitations |
|--------|----------|-------------|
| **Thermal** | Hot gases, heat shimmer | Requires temperature difference |
| **Background** | Visible vapor, steam | Only works with visible gases |
| **Edge** | Refractive distortion | Needs stable baseline, sensitive to movement |
| **Histogram** | Opacity changes | Affected by lighting changes |

## 📋 **Requirements**

- Python 3.8+
- OpenCV 4.8+
- NumPy
- Matplotlib
- Camera/webcam (for live testing)

## 🛠️ **Installation**

1. **Clone the repository:**
```bash
git clone https://github.com/AmmarBandukwala/yolo-model-test.git
cd yolo-model-test/misc
```

2. **Install dependencies:**
```bash
pip install opencv-python numpy matplotlib pathlib argparse logging json time collections
```

Or create a `requirements.txt`:
```bash
pip install -r requirements.txt
```

## 🚀 **Quick Start**

### Live Webcam Testing
```bash
# Basic thermal detection
python gas_leak_detection.py --source 0 --method thermal

# High sensitivity background subtraction
python gas_leak_detection.py --source 0 --method background --sensitivity 0.8
```

### Video File Analysis
```bash
# Process video with thermal detection
python gas_leak_detection.py --source test_video.mp4 --method thermal --output results.mp4

# Background subtraction with results export
python gas_leak_detection.py --source input.mp4 --method background --save-results detection_log.json
```

### Single Image Analysis
```bash
# Edge distortion detection
python gas_leak_detection.py --source image.jpg --method edge --output result.jpg

# Histogram analysis with no display
python gas_leak_detection.py --source photo.png --method histogram --no-display --save-results results.json
```

## 📖 **Command Line Options**

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--source` | Input source | `0` | Camera index, video file, image file |
| `--method` | Detection method | `thermal` | `thermal`, `background`, `edge`, `histogram` |
| `--sensitivity` | Detection sensitivity | `0.5` | `0.0` - `1.0` |
| `--output` | Output file path | `None` | Video/image file path |
| `--save-results` | Save JSON results | `None` | JSON file path |
| `--no-display` | Disable video display | `False` | Flag |

## 🧪 **Testing Scenarios**

### ✅ **What You Can Test (Visible Effects)**

**Thermal Detection:**
```bash
# Test with hair dryer heat shimmer
python gas_leak_detection.py --source 0 --method thermal --sensitivity 0.6

# Hot water steam
python gas_leak_detection.py --source 0 --method thermal --sensitivity 0.4
```

**Background Subtraction:**
```bash
# Steam from kettle
python gas_leak_detection.py --source 0 --method background --sensitivity 0.7

# Dry ice vapor (safe handling!)
python gas_leak_detection.py --source 0 --method background --sensitivity 0.5
```

**Edge Distortion:**
```bash
# Heat waves from hot surface
python gas_leak_detection.py --source 0 --method edge --sensitivity 0.3
```

### ❌ **What Won't Work Reliably**

- Natural gas (methane) - invisible
- Propane - invisible  
- Carbon monoxide - invisible
- Most industrial gases - invisible
- Small leaks without visible effects

## 📊 **Understanding Results**

### Detection Output
```
Method: thermal
Confidence: 0.75
Affected Area: 12.5%
Est. Concentration: 8.3 AU
```

**Confidence Score (0-1):**
- `0.0-0.3`: Low/No detection
- `0.3-0.7`: Moderate detection
- `0.7-1.0`: Strong detection

**Concentration (AU = Arbitrary Units):**
- NOT real gas concentration
- Relative measure of visual effect intensity
- Requires calibration with actual gas sensors

### JSON Results Format
```json
{
  "method": "thermal",
  "total_detections": 45,
  "results": [
    {
      "timestamp": 1640995200,
      "detection_confidence": 0.75,
      "concentration_data": {
        "affected_area_percent": 12.5,
        "mean_intensity": 0.68,
        "estimated_concentration_au": 8.3,
        "confidence_score": 0.75
      }
    }
  ]
}
```

## 🎮 **Interactive Controls**

**During Live Detection:**
- `q`: Quit application
- `s`: Save current frame (webcam mode)
- `ESC`: Exit (image mode)

## 📈 **Performance Tips**

### Optimize for Speed
```bash
# Lower resolution processing
python gas_leak_detection.py --source 0 --method thermal --sensitivity 0.5

# Reduce sensitivity for faster processing
python gas_leak_detection.py --source 0 --method background --sensitivity 0.3
```

### Optimize for Accuracy
```bash
# Higher sensitivity, stable environment
python gas_leak_detection.py --source 0 --method edge --sensitivity 0.8

# Use multiple methods (run separately)
python gas_leak_detection.py --source test.mp4 --method thermal --save-results thermal_results.json
python gas_leak_detection.py --source test.mp4 --method background --save-results background_results.json
```

## 🔧 **Troubleshooting**

### Common Issues

**No Camera Access:**
```bash
# Try different camera indices
python gas_leak_detection.py --source 1 --method thermal
python gas_leak_detection.py --source 2 --method thermal
```

**Too Many False Positives:**
- Reduce sensitivity: `--sensitivity 0.2`
- Ensure stable environment (no moving objects)
- Use `edge` or `histogram` methods for stationary setups

**No Detections:**
- Increase sensitivity: `--sensitivity 0.8`
- Try different methods
- Ensure detectable gas effects (visible vapor, heat shimmer)
- Check lighting conditions

**Poor Performance:**
- Close other applications using camera
- Use lower resolution videos
- Reduce sensitivity for faster processing

### Method-Specific Tips

**Thermal Detection:**
- Works best with significant temperature differences
- Requires movement between frames
- Good for: Hot gas leaks, heat shimmer

**Background Subtraction:**
- Needs time to establish background model
- Sensitive to lighting changes
- Good for: Visible vapors, moving gas clouds

**Edge Detection:**
- Requires stable camera position
- Sensitive to any movement
- Good for: Refractive distortion, density changes

**Histogram Analysis:**
- Best for uniform lighting
- Detects overall opacity changes
- Good for: Large gas clouds, vapor presence

## 📚 **Example Test Procedures**

### Safe Steam Test
```bash
# 1. Start detection
python gas_leak_detection.py --source 0 --method background --sensitivity 0.6

# 2. Boil water safely
# 3. Direct steam toward camera area
# 4. Observe detection overlay
# 5. Save results if needed
```

### Heat Shimmer Test
```bash
# 1. Start thermal detection
python gas_leak_detection.py --source 0 --method thermal --sensitivity 0.5

# 2. Use hair dryer on hot setting
# 3. Create heat shimmer in camera view
# 4. Observe optical flow detection
# 5. Check confidence scores
```

### Controlled Environment Test
```bash
# 1. Set up stable camera
# 2. Establish baseline
python gas_leak_detection.py --source 0 --method edge --sensitivity 0.4

# 3. Introduce controlled disturbance
# 4. Monitor detection results
# 5. Export data for analysis
```

## 📄 **License**

This project is licensed under the MIT License - see LICENSE file for details.

## 🔗 **Related Projects**

- **Professional Gas Detection**: Look into FLIR thermal cameras
- **Specialized Gas Cameras**: Research infrared gas imaging
- **Sensor Integration**: Combine with actual gas sensors
- **Industrial Solutions**: Explore certified gas detection systems

---

**Remember: This is a computer vision experiment, not a safety device. For real gas detection, always use certified equipment and follow proper safety protocols.** 🛡️