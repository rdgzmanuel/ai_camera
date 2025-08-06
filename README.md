<center>
<h1 align="center">Azzulei Technologies - AI</h1>
<p align="center">AI Camera and Virtual Commentary System</p>
<br>
</center>

This repository contains the source code of Azzulei Technologies' AI camera control system and virtual commentary generator.

The former is a pipeline for automating camera control in live sports events using fixed cameras. This allows viewers to enjoy a much more pleasant experience, with the video focused on the actual action on the field. It works by detecting football players (with potential to extend to other sports) and uses PID control to move the camera, aiming to include a high percentage of detected players. As a future improvement, ball detection could enhance the output, although this has proven difficult due to varying video conditions, the small size of the ball, and its high speed.

The latter is a virtual commentary generation system designed to play when a viewer joins a live stream. It is based on a JSON file that stores match-related information, such as stadium, weather conditions, and rival teams. This file is passed to an LLM (Large Language Model) to generate text, which is then converted to MP4 using TTS (Text-to-Speech).

---

### Repository Layout

```text
.
├── src/                                  # Source code
│   ├── quantization/                     # Model quantization files (via ONNX)
│   ├── unused/                           # Deprecated files (early development)
│   │   ├── divide_yolo_dataset.py        # Splits YOLO data for fine-tuning
│   │   ├── models.py                     # Initial fine-tuning approach
│   │   ├── object_detector.py            # Early object detection implementation
│   │   ├── prepare_ball_only.py          # Prepares YOLO dataset for ball-only detection
│   │   ├── prepare_no_ball.py            # Prepares YOLO dataset for player-only detection
│   │   ├── test_import.py                # Library test file
│   │   └── train_yolo.py                 # YOLO fine-tuning
│   │
│   ├── __init__.py                       # Enables odule formatting
│   ├── adaptive_resolution.py            # Adapts input resolution based on scene complexity
│   ├── commentary.py                     # Generates virtual commentary
│   ├── config.py                         # Contains secret keys (not pushed)
│   ├── controller.py                     # PID control system
│   ├── download_soccertrack.py           # Downloads SoccerTrack dataset
│   ├── football_field_detection.py       # Detects field coordinates for filtering
│   ├── optical_flow.py                   # Optical flow for performance optimization
│   ├── player_tracker.py                 # Performs player tracking with optimizations
│   └── utils.py                          # Auxiliary functions
│
├── videos/                               # Input videos for testing
├── football.yaml                         # YOLO fine-tuning configuration
├── requirements.txt                      # Required Python packages
└── README.md                             # You are here 🖖
```

---

## Folder Structure

The `/ai_camera` folder is the main directory of the repository. It contains all the required files and subdirectories:

- `/models`: Downloaded models for object detection.  
  ➤ [Model download link](https://github.com/ultralytics/ultralytics)

- `/videos`: Input videos for object detection and prepared SoccerTrack dataset for model training.

- `.gitignore`: Specifies files and folders to be ignored by Git (e.g., large local files).

- `.yaml`: Configuration for YOLO fine-tuning.

- `/src`: Source code, including `__init__.py`. Some files are deprecated or used for testing.

---

## How to Run

From the `ai_camera` directory, run:

```bash
python -m src.<file_name_without_extension>
```

For example:

```bash
python -m src.player_tracker
```

To prepare datasets for training, use the `prepare_*.py` files — each is designed for a specific type of data (ball only, no ball, etc.).

`player_tracker.py` performs player tracking and final camera control on a selected input video. Multiple hyperparameters can be adjusted depending on the input. It uses the YOLOv8n model and its quantized version, which offers a lightweight alternative with nearly equal performance. The system runs full YOLO detections every _n_ frames, tracking in between using optical flow. Adaptive resolution can be enabled to adjust input size dynamically based on scene complexity (using edge detection).

For wide-view settings, where the input resolution is highly horizontal, the system can divide frames into patches, process them independently, and merge detections. This improves detection but can significantly reduce performance due to multiple forward passes.

---

## Installation

Install required libraries using:

```bash
pip install -r requirements.txt
```

---

## Dependencies

To use this repository, clone the following dependencies **outside** the `ai_camera` folder (at the same directory level):

---

### [ByteTrack](https://github.com/ifzhang/ByteTrack)

```bash
git clone https://github.com/ifzhang/ByteTrack.git
cd ByteTrack
pip install -r requirements.txt
python setup.py develop
```

---

### [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) *(Only if not installed via `requirements.txt`)*

```bash
git clone https://github.com/Megvii-BaseDetection/YOLOX
cd YOLOX
pip install -v -e .
```

**Note:**  
You’ll need to manually fix two files:  
Replace `dtype=np.float` with `dtype=float` in:

- `YOLOX/yolox/tracker/matching.py`
- `YOLOX/yolox/tracker/byte_tracker.py`

---
