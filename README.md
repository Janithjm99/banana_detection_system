# 🍌 Banana Condition Detection System

A computer vision-based web application that detects and classifies banana ripeness stages and defects in real-time using color analysis and deep learning.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.3.2-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8.1-red.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-8.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Ripeness Stages](#-ripeness-stages)
- [Tech Stack](#-tech-stack)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Endpoints](#-api-endpoints)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

The Banana Condition Detection System is a Flask-based web application that uses computer vision to analyze banana images and video streams. It classifies bananas into **9 distinct ripeness stages** and detects **defects** such as bruises and mold.

**Key Capabilities:**
- 📸 **Image Upload** — Analyze uploaded banana photos
- 📹 **Live Camera** — Real-time detection via webcam
- 🎯 **Multi-banana Detection** — Process multiple bananas in a single image
- ⚠️ **Defect Detection** — Identify bruises, mold, and spoilage
- 📊 **Visual Analytics** — Color distribution breakdown per banana

---

## ✨ Features

### Core Detection
| Feature | Description |
|---------|-------------|
| **9 Ripeness Stages** | From Raw (Green) to Spoiled with detailed descriptions |
| **Color Analysis** | HSV-based segmentation for accurate color classification |
| **Defect Detection** | Bruise detection (dark spots) and mold detection (texture analysis) |
| **YOLOv8 Integration** | Optional deep learning detection for higher accuracy |
| **Simple Mode** | Lightweight color-based detection without heavy ML dependencies |

### Web Interface
- **Modern Responsive UI** — Works on desktop and mobile
- **Drag & Drop Upload** — Intuitive image upload with preview
- **Live Camera Feed** — Real-time webcam streaming with detection overlay
- **Stage Reference** — Interactive guide to all ripeness stages
- **Confidence Visualization** — Progress bars and color-coded indicators

### API
- **RESTful Endpoints** — JSON-based API for integration
- **Video Frame Processing** — Single frame analysis for external clients
- **Stage Reference API** — Fetch all ripeness definitions

---

## 🍌 Ripeness Stages

| Stage | Name | Color | Description | Best Use |
|-------|------|-------|-------------|----------|
| 1 | **Raw (Green)** | 🟢 | Unripe, starchy, not sweet | Cooking, frying |
| 2 | **Turning** | 🟢🟡 | Beginning to ripen, some starch | Frying, cooking |
| 3 | **More Green than Yellow** | 🟢🟡 | Mostly starchy, firm | Frying |
| 4 | **More Yellow than Green** | 🟡🟢 | Sweet, firm, creamy | Fresh eating |
| 5 | **Ripe (Yellow)** | 🟡 | Very sweet, soft texture | Snacking, smoothies |
| 6 | **Fully Ripe** | 🟡🟤 | Maximum sweetness, soft flesh | Cereal, smoothies |
| 7 | **Speckled (Sugar Spots)** | 🟡🟤 | Very sweet, aromatic, soft | Banana bread, baking |
| 8 | **Overripe (Brown)** | 🟤 | Mushy, extremely sweet | Pureeing, freezing |
| 9 | **Spoiled** | 🔴 | Moldy, rotten, not edible | **Compost only** |

---

## 🛠 Tech Stack

| Category | Technology |
|----------|------------|
| **Backend** | Flask 2.3.2, Python 3.8+ |
| **Computer Vision** | OpenCV 4.8.1, NumPy |
| **Deep Learning** | Ultralytics YOLOv8, TensorFlow/Keras |
| **Image Processing** | Pillow, imutils |
| **Frontend** | Vanilla JS, HTML5, CSS3 (CSS Grid/Flexbox) |
| **Deployment** | Gunicorn, Docker-ready |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Client (Browser)                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  Upload Tab │  │ Camera Tab  │  │   Stages Reference  │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
└─────────┼────────────────┼─────────────────────┼─────────────┘
          │                │                     │
          ▼                ▼                     ▼
┌─────────────────────────────────────────────────────────────┐
│                      Flask Backend                           │
│  ┌─────────────────┐    ┌─────────────────────────────────┐  │
│  │  REST API Routes│    │    BananaClassifierSimple       │  │
│  │  /upload        │───▶│    (or BananaClassifier)        │  │
│  │  /video_feed    │    │  ┌──────────────────────────┐  │  │
│  │  /process_...   │    │  │ Color Analysis (HSV)     │  │  │
│  │  /get_stages    │    │  │ Defect Detection         │  │  │
│  └─────────────────┘    │  │ Ripeness Classification  │  │  │
│                         │  └──────────────────────────┘  │  │
└─────────────────────────┼─────────────────────────────────┘  │
                          ▼
              ┌───────────────────────┐
              │   YOLOv8 Model (opt)  │
              │   yolov8n.pt (COCO)   │
              └───────────────────────┘
```

### Classifier Modes

| Classifier | Detection Method | Dependencies | Speed |
|------------|------------------|--------------|-------|
| `BananaClassifierSimple` | Color segmentation + contours | OpenCV only | ⚡ Fast |
| `BananaClassifier` | YOLOv8 + color analysis | Ultralytics, PyTorch | 🎯 Accurate |

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Webcam (for live camera feature)

### Quick Start

```bash
# 1. Clone the repository
git clone <repository-url>
cd banana_detection_system

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the application
python app.py
```

### Development Installation

```bash
# Install with development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
python -m pytest test_banana.py -v
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

```bash
# Build and run
docker build -t banana-detection .
docker run -p 5000:5000 banana-detection
```

---

## 💻 Usage

### Starting the Server

```bash
python app.py
```

The application will be available at **http://localhost:5000**

### Web Interface

1. **Upload Image Tab** — Click or drag & drop a banana image
2. **Live Camera Tab** — Start webcam for real-time detection
3. **Stages Tab** — View all ripeness stages with descriptions

### Programmatic Usage

```python
from banana_classifier_simple import BananaClassifierSimple

# Initialize classifier
classifier = BananaClassifierSimple()

# Process an image
result = classifier.process_image('path/to/banana.jpg')

if result['detected']:
    for banana in result['results']:
        print(f"Stage: {banana['ripeness']['name']}")
        print(f"Confidence: {banana['confidence']:.2f}")
        print(f"Usage: {banana['ripeness']['usage']}")
        if banana['defects']['has_defects']:
            print(f"⚠️ Defects: {banana['defects']['defect_type']}")
```

### Using Full YOLOv8 Classifier

```python
from banana_classifier import BananaClassifier

# Requires: pip install ultralytics torch
classifier = BananaClassifier()  # Downloads yolov8n.pt on first run
result = classifier.process_image('path/to/banana.jpg')
```

---

## 🔌 API Endpoints

### Upload Image
```http
POST /upload
Content-Type: multipart/form-data

file: <image_file>
```

**Response:**
```json
{
  "detected": true,
  "number_of_bananas": 1,
  "image_path": "20260815_182711_banana.jpg",
  "results": [
    {
      "bbox": [100, 150, 300, 400],
      "confidence": 0.92,
      "ripeness": {
        "stage": 5,
        "name": "Ripe (Yellow)",
        "description": "Very sweet, soft texture",
        "color": [0, 255, 255],
        "usage": "Snacking, smoothies"
      },
      "color_distribution": {
        "green": 0.05,
        "yellow": 0.85,
        "brown": 0.10
      },
      "defects": {
        "has_defects": false,
        "defect_type": null
      },
      "timestamp": "2026-08-15T18:27:11.123456"
    }
  ]
}
```

### Get All Stages
```http
GET /get_stages
```

**Response:**
```json
{
  "raw": {
    "stage": 1,
    "name": "Raw (Green)",
    "description": "Unripe, starchy, not sweet",
    "color": [0, 255, 0],
    "usage": "Cooking, frying"
  },
  ...
}
```

### Video Stream
```http
GET /video_feed
```
Returns MJPEG stream for real-time camera feed.

### Process Video Frame
```http
POST /process_video_frame
Content-Type: application/json

{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ..."
}
```

**Response:**
```json
{
  "success": true,
  "results": [...]
}
```

### Serve Uploaded Files
```http
GET /uploads/<filename>
```

---

## 📁 Project Structure

```
banana_detection_system/
├── app.py                      # Flask application entry point
├── banana_classifier.py        # Full YOLOv8-based classifier
├── banana_classifier_simple.py # Lightweight color-based classifier
├── test_banana.py              # Unit tests
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
├── templates/
│   └── index.html              # Main web interface (SPA)
├── uploads/                    # Uploaded images (auto-created)
│   └── *.jpg
├── venv/                       # Virtual environment (ignored)
└── README.md                   # This file
```

### Key Modules

| File | Purpose |
|------|---------|
| `app.py` | Flask routes, request handling, video streaming |
| `banana_classifier.py` | YOLOv8 + HSV analysis, 9-stage classification |
| `banana_classifier_simple.py` | Contour-based detection, no ML dependencies |
| `templates/index.html` | Single-page app with 3 tabs (upload, camera, stages) |

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Flask Configuration
SECRET_KEY=your-secure-random-key
FLASK_ENV=development
FLASK_DEBUG=1

# Server Configuration
HOST=0.0.0.0
PORT=5000

# Upload Configuration
MAX_CONTENT_LENGTH=16777216  # 16MB
UPLOAD_FOLDER=uploads

# Model Configuration (for full classifier)
YOLO_MODEL_PATH=yolov8n.pt
CONFIDENCE_THRESHOLD=0.5
```

### Classifier Thresholds

Adjust in `banana_classifier_simple.py` or `banana_classifier.py`:

```python
# Color HSV ranges
self.green_lower = np.array([35, 50, 50])
self.green_upper = np.array([85, 255, 255])
self.yellow_lower = np.array([20, 50, 50])
self.yellow_upper = np.array([35, 255, 255])
self.brown_lower = np.array([0, 50, 50])
self.brown_upper = np.array([20, 255, 255])

# Defect thresholds
self.bruise_threshold = 0.10    # 10% dark pixels
self.mold_threshold = 1000      # Laplacian variance
```

---

## 🧪 Testing

```bash
# Run unit tests
python test_banana.py

# Run with coverage
pip install pytest-cov
pytest --cov=banana_classifier_simple test_banana.py

# Lint code
flake8 *.py
black *.py
```

### Test Coverage Areas
- Color distribution analysis
- Ripeness classification logic
- Defect detection (bruises, mold)
- Image processing pipeline
- API endpoint responses

---

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Add type hints to new functions
- Write tests for new features
- Update documentation for API changes
- Keep commits atomic and descriptive

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Banana Detection System

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 🙏 Acknowledgments

- **Ultralytics** — YOLOv8 object detection framework
- **OpenCV** — Computer vision library
- **Flask** — Lightweight web framework
- **COCO Dataset** — Pre-trained model classes (banana = class 46)

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Email**: support@banana-detection.example.com

---

<div align="center">

**Made with 🍌 for banana enthusiasts everywhere**

⭐ Star this repo if you found it useful!

</div>
