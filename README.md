# 👁️ Vision Hire: AI-Based Real-Time Interview Confidence Analyzer

A real-time computer vision system that analyzes interview confidence by tracking eye movements and head pose using your webcam — built with Python, OpenCV, and MediaPipe.

---

## 📌 Problem Statement

Interviews are stressful, and candidates often struggle with maintaining eye contact and stable head posture. Vision Hire helps candidates and interviewers evaluate **confidence levels in real-time** by analyzing facial cues through a webcam, providing instant feedback and a session summary.

---

## 🚀 Features

- **Real-time confidence scoring** — live 0–100 confidence score displayed on screen
- **Iris tracking** — detects eye contact by measuring iris position relative to eye corners
- **Head pose estimation** — computes yaw, pitch, and roll angles using solvePnP
- **Adaptive calibration** — calibrates to the user's natural position in the first 30 frames
- **Exponential smoothing** — ensures stable, noise-free score updates
- **Live graph** — real-time matplotlib graph showing confidence over time
- **Session summary** — professional Tkinter popup with average, max, min scores and overall rating (Excellent / Good / Needs Improvement)

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| Computer Vision | OpenCV |
| Face Landmark Detection | MediaPipe Face Mesh |
| Head Pose Estimation | OpenCV solvePnP + RQDecomp3x3 |
| Numerical Processing | NumPy |
| Live Graph | Matplotlib |
| Summary UI | Tkinter |

---

## 📁 Project Structure

```
Vision-Hire-AI-Interview-Confidence-Analyzer/
├── main.py                 # Main application — webcam loop, calibration, graph, popup
├── interview_analyzer.py   # Core analyzer class — iris tracking, head pose, confidence score
├── utils.py                # Utility functions — distance, eye aspect ratio, posture score
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/fathima-s-13/Vision-Hire-AI-Interview-Confidence-Analyzer.git
cd Vision-Hire-AI-Interview-Confidence-Analyzer
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install opencv-python mediapipe numpy matplotlib
```

### 4. Run the application
```bash
python main.py
```

Make sure your **webcam is connected**. Press **Q** to stop the session and see your summary.

---

## 🧠 How It Works

```
Webcam Frame
    │
    ▼
MediaPipe Face Mesh
(468 facial landmarks including iris refinement)
    │
    ├──► Iris Tracking
    │       └──► Measures iris X position vs eye corners
    │            → Eye contact score (0–100)
    │
    ├──► Head Pose Estimation (solvePnP)
    │       └──► Computes Yaw, Pitch, Roll angles
    │            → Head stability score
    │
    ├──► Face Centering
    │       └──► Nose X offset from frame center
    │            → Centering score
    │
    └──► Weighted Confidence Score
             (Eye 50% + Head 25% + Center 25%)
             → Exponential smoothing (α = 0.85)
             → Live display + Graph update
```

---

## 📊 Confidence Score Breakdown

| Component | Weight | What it measures |
|---|---|---|
| Eye Contact | 50% | Iris position relative to eye corners |
| Head Stability | 25% | Yaw and pitch angle deviation |
| Face Centering | 25% | Nose offset from frame center |

---

## 📋 Session Summary

At the end of each session, a summary popup shows:

- **Average Confidence** — overall session score
- **Maximum Confidence** — best moment
- **Minimum Confidence** — weakest moment
- **Attention Drops** — number of times score fell below 40%
- **Overall Rating** — Excellent (≥85%) / Good (≥65%) / Needs Improvement (<65%)

---

## 📌 Future Improvements

- [ ] Add audio confidence analysis (voice tone, pace)
- [ ] Export session report as PDF
- [ ] Add posture analysis using shoulder detection
- [ ] Support for recording and playback of sessions
- [ ] Web-based version using Flask + WebRTC

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

*Built with ❤️ using Python, OpenCV, MediaPipe, and NumPy*
