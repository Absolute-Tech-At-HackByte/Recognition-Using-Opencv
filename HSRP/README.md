# License Plate Detection System

This system uses YOLOv8 to detect and classify license plates into two categories:
- **Ordinary** license plates
- **HSRP** (High Security Registration Plate) license plates

The detection system is optimized to run on NVIDIA GPUs for fast inference.

## Features

- Detect license plates in images and videos
- Classify plates as ordinary or HSRP
- Extract license plate regions for further processing
- GPU-accelerated for real-time performance 
- Support for batch processing of images
- Visualization and saving of detection results

## Requirements

- Python 3.8+
- CUDA-compatible NVIDIA GPU (recommended)
- PyTorch with CUDA support
- OpenCV
- Ultralytics YOLOv8

## Installation

Make sure PyTorch is installed with CUDA support:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics opencv-python
```

## Usage

### Quick Start

To test the license plate detector on the test images:

```bash
cd HSRP
python run_detection.py
```

This will:
1. Load the trained model
2. Process all test images
3. Display the detection results
4. Save annotated images to the `HSRP/results` directory

### Command Line Interface

For more control, use the `license_plate_detector.py` script directly:

```bash
python license_plate_detector.py --input path/to/image_or_directory --output path/to/save/results --device 0
```

Options:
- `--model`: Path to YOLOv8 model (.pt file). If not specified, will use the latest trained model.
- `--input`: Path to input image, video, or directory.
- `--output`: Path to save output files (optional).
- `--conf`: Confidence threshold (default: 0.5).
- `--device`: Device to use (0 for GPU, cpu for CPU).
- `--show`: Display results during processing.

### Examples

Process a single image:
```bash
python license_plate_detector.py --input data/test/images/test.jpg --output HSRP/results/test_output.jpg --show
```

Process a video:
```bash
python license_plate_detector.py --input path/to/video.mp4 --output HSRP/results/output_video.mp4
```

Process all images in a directory:
```bash
python license_plate_detector.py --input path/to/images/ --output HSRP/results/
```

## API Usage

You can also use the detector in your own code:

```python
from license_plate_detector import LicensePlateDetector

# Initialize detector
detector = LicensePlateDetector(
    model_path="runs/detect/train3/weights/best.pt",
    device="0",  # Use GPU
    conf_threshold=0.5
)

# Process an image
annotated_img, plate_images, plate_info = detector.process_image(
    "test_image.jpg",
    show_result=True,
    save_path="output.jpg"
)

# Process a video
detector.process_video("input.mp4", "output.mp4", display=True)

# Process all images in a directory
results = detector.process_directory(
    "input_dir",
    "output_dir",
    show_result=False
)
```

## Model Information

The detection model is a YOLOv8 model trained on license plate images with two classes:
- Class 0: Ordinary
- Class 1: HSRP

The model was trained with GPU acceleration using an NVIDIA GeForce RTX 3050 Laptop GPU.

## Performance

On an NVIDIA GeForce RTX 3050 GPU, the system can process:
- Images: 10-30 frames per second (depending on image size)
- Videos: Real-time processing for 720p video 