
# 🧍‍♂️ Pose Fitness Evaluator

### Video-based Human Pose Detection & Posture Evaluation System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Pose-green)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Overview

Incorrect posture during workouts and physical activities can reduce effectiveness and cause injuries.
This project provides an **automated posture evaluation system** that analyzes human body pose from video input and gives **visual feedback** using deep learning–based pose estimation.

---

## 🎯 Problem Statement

Manual posture correction requires expert supervision, which is not always available.
The goal of this project is to build a system that:

* Detects human body keypoints from video
* Computes joint angles and body alignment
* Evaluates posture quality
* Overlays feedback directly onto the video

---

## 💡 Solution Summary

The system uses a **pretrained YOLOv8 Pose model** to extract human keypoints from each video frame.
Rule-based geometric analysis is applied to evaluate posture and generate interpretable feedback.

✔ Fully offline
✔ No dataset training required
✔ Modular and extensible design

---

## 🧠 System Architecture

```
Input Video
   │
   ▼
Frame Capture (OpenCV)
   │
   ▼
Pose Detection (YOLOv8)
   │
   ▼
Keypoint Extraction
   │
   ▼
Angle & Alignment Analysis
   │
   ▼
Posture Evaluation
   │
   ▼
Skeleton + Feedback Overlay
   │
   ▼
Output Video
```

---

## 🛠️ Tech Stack

| Category   | Tools                     |
| ---------- | ------------------------- |
| Language   | Python                    |
| Model      | YOLOv8 Pose (Ultralytics) |
| Vision     | OpenCV                    |
| Math       | NumPy                     |
| DL Backend | PyTorch                   |

---

## 📂 Project Structure

```
pose-fitness-evaluator/
│
├── src/
│   ├── detector.py        # Pose detection module
│   ├── evaluator.py       # Posture evaluation logic
│   ├── utils.py           # Angle & geometry utilities
│   ├── overlay.py         # Visualization utilities
│   └── main.py            # End-to-end pipeline
│
├── data/
│   └── samples/
│       └── test.mp4
│
├── outputs/
│   └── demo_output.mp4
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ How It Works

1. Video is read frame-by-frame using OpenCV
2. Each frame is passed to the YOLOv8 pose model
3. Human keypoints (17 per person) are extracted
4. Joint angles are calculated using vector math
5. Rule-based logic evaluates posture quality
6. Skeleton and feedback text are drawn on frames
7. Output video is saved to disk

---

## ▶️ How to Run

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Mercer011/pose-fitness-evaluator.git
cd pose-fitness-evaluator
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
# Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Application

```bash
python src/main.py
```

📁 Output video will be generated in the `outputs/` directory.

---

## 📊 Output Features

* Human skeleton overlay
* Real-time joint angle visualization
* Posture quality feedback
* Processed `.mp4` output video

---

## ⚠️ Limitations

* Rule-based evaluation (not ML classification)
* Single-person focus
* Sensitive to camera angle and lighting
* No custom dataset fine-tuning

---

## 🚀 Future Enhancements

* Train a posture classification model
* Multi-person posture analysis
* Real-time webcam inference
* Mobile / web deployment
* Exercise-specific posture scoring

---

## 👨‍💻 Author

**Abhishek**
Machine Learning & Computer Vision Enthusiast

* GitHub: [https://github.com/Mercer011](https://github.com/Mercer011)

---

## 📝 Notes for Reviewers

* Focused on **applied ML engineering**
* Clean modular design
* Interpretable evaluation logic
* Interview-defensible architecture

---

