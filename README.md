# premium YOLO Object Detection Project 🚀 

A high-performance, real-time object detection system using the latest YOLO (Ultralytics) framework.

## ✨ Features
- **Real-time Detection:** High-FPS inference with webcam support.
- **Premium UI:** Polished overlays, anti-aliased boxes, and status dashboards.
- **Multi-Source Support:** Process webcams, images, or video files.
- **Modern Backend:** Built with YOLO11, optimized for speed and accuracy.

## 🛠️ Installation

1. **Clone or Download** this repository.
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### 1. Real-time Webcam Detection
```bash
python main.py --source 0
```

### 2. Process an Image/Video
```bash
python main.py --source path/to/video.mp4 --conf 0.5
```

### 3. Change Model Size
You can use different model sizes (`n` for nano, `s` for small, `m` for medium, etc.):
```bash
python main.py --model yolo11s.pt
```

## 📂 Project Structure
- `main.py`: Entry point and CLI.
- `src/detector.py`: Core YOLO logic and inference.
- `src/visualizer.py`: UI and visualization helpers.
- `requirements.txt`: Python package dependencies.

## 📝 License
MIT License. Feel free to use and modify!
