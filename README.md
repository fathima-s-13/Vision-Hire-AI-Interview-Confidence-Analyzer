# 🧠 AI-Based Real-Time Interview Confidence Analyzer

**Tech Stack:** Python, OpenCV, MediaPipe Face Mesh, NumPy, Matplotlib, Tkinter

## Project Overview
This project evaluates interview confidence in real-time using a webcam. It tracks **iris movement** and **head pose** to compute a **0–100 confidence score**. Scores are visualized live with dynamic graphs and a professional session summary UI.

## Features
- Real-time **face mesh** and **iris tracking**
- **Head pose estimation** for accurate attention detection
- Dynamic **0–100 confidence scoring** with adaptive calibration
- **Exponential smoothing** for stable analytics
- **Live graph** of confidence score over time
- **Professional UI popup** showing session summary

## Usage
1. Install required packages:  
   ```bash
   pip install opencv-python mediapipe numpy matplotlib
