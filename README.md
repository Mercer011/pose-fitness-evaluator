
Pose Fitness Evaluator

Video-based Human Pose Detection and Posture Evaluation System

1. Problem Statement

Incorrect body posture during physical activities such as exercise, yoga, or fitness training can lead to poor performance and long-term injuries. Manual posture correction requires expert supervision, which is not always accessible.

The objective of this project is to build an automated system that:

Detects human body keypoints from a video

Analyzes joint angles and body alignment

Provides visual and logical posture feedback

2. Solution Overview

This project implements a video-based pose evaluation pipeline using a pretrained deep learning pose estimation model.
The system processes each video frame, extracts human keypoints, evaluates posture quality using geometric rules, and overlays feedback directly onto the video.

The solution works fully offline and can be extended to real-time camera input.

3. System Architecture / Pipeline
Input Video
   ↓
Frame Extraction (OpenCV)
   ↓
Pose Detection (YOLOv8 Pose Model)
   ↓
Keypoint Extraction
   ↓
Posture Evaluation (Angle + Alignment Logic)
   ↓
Visual Overlay (Skeleton + Feedback)
   ↓
Output Video

4. Tech Stack

Python

YOLOv8 Pose (Ultralytics)

OpenCV

NumPy

PyTorch

5. How It Works (Step-by-Step)

A video file is read frame-by-frame using OpenCV.

Each frame is passed to a YOLOv8 pose detection model.

The model returns 17 body keypoints per detected person.

Joint angles (elbow, knee, back alignment) are calculated using vector geometry.

Rule-based logic evaluates posture quality.

Skeleton, angles, and feedback text are overlaid on the frame.

The processed frames are saved as a new output video.

6. Project Structure
pose-fitness-evaluator/
│
├── src/
│   ├── detector.py        # Pose detection using YOLOv8
│   ├── evaluator.py       # Posture evaluation logic
│   ├── utils.py           # Angle and geometry utilities
│   ├── overlay.py         # Visualization and drawing
│   └── main.py            # End-to-end execution
│
├── data/
│   └── samples/
│       └── test.mp4       # Input video
│
├── outputs/
│   └── demo_output.mp4    # Output video with overlay
│
├── requirements.txt
├── README.md
└── .gitignore

7. How to Run the Project
Step 1: Clone the Repository
git clone https://github.com/Mercer011/pose-fitness-evaluator.git
cd pose-fitness-evaluator

Step 2: Create Virtual Environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

Step 3: Install Dependencies
pip install -r requirements.txt

Step 4: Run the System
python src/main.py


The output video will be saved in the outputs/ directory.

8. Output

Detected skeleton overlay

Joint angle visualization

Posture feedback (e.g., Good posture, Adjust back alignment)

Processed output video (.mp4)

9. Limitations

Rule-based posture evaluation (not learned)

Single-person evaluation focus

Accuracy depends on camera angle and lighting

No dataset-specific fine-tuning performed

10. Future Improvements

Train a custom posture classification model

Support real-time webcam inference

Multi-person posture evaluation

Integration with mobile or web applications

Fitness-specific exercise recognition

11. Author

Abhishek
Machine Learning / Computer Vision Enthusiast
GitHub: https://github.com/Mercer011

