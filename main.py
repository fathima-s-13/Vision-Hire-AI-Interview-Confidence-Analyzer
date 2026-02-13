import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt

# =========================
# INITIALIZATION
# =========================

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)

confidence_scores = []
attention_drops = 0

# Exponential smoothing parameters
previous_score = 100
alpha = 0.6   # responsiveness (0.5–0.7 recommended)

# Baseline calibration
baseline_eye = None
baseline_head = None
calibration_frames = 30
frame_count = 0

# =========================
# LIVE WHITE GRAPH
# =========================

plt.ion()
fig, ax = plt.subplots()
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
line, = ax.plot([], [], linewidth=2)
ax.set_ylim(0, 100)
ax.set_xlim(0, 100)
ax.set_title("Confidence Score Over Time")
ax.set_xlabel("Frame")
ax.set_ylabel("Confidence")
plt.show()

# =========================
# CONFIDENCE FUNCTION
# =========================

def calculate_confidence(eye_dev, head_dev):

    eye_penalty = min(eye_dev * 3000, 100)
    head_penalty = min(head_dev * 2000, 100)

    score = 100 - (eye_penalty * 0.6 + head_penalty * 0.4)

    return max(0, min(100, score))


# =========================
# POPUP FUNCTION
# =========================

def show_summary_popup():

    avg_conf = int(np.mean(confidence_scores))
    max_conf = int(np.max(confidence_scores))
    min_conf = int(np.min(confidence_scores))

    if avg_conf >= 85:
        rating = "EXCELLENT"
        color = "green"
    elif avg_conf >= 65:
        rating = "GOOD"
        color = "orange"
    else:
        rating = "NEEDS IMPROVEMENT"
        color = "red"

    pastel_pink = "#FADADD"

    root = tk.Tk()
    root.title("Interview Performance Summary")
    root.configure(bg=pastel_pink)
    root.geometry("420x380")
    root.resizable(False, False)

    tk.Label(root,
             text="Interview Performance Summary",
             font=("Helvetica", 16, "bold"),
             bg=pastel_pink).pack(pady=15)

    stats = f"""
Average Confidence: {avg_conf}%
Maximum Confidence: {max_conf}%
Minimum Confidence: {min_conf}%

Attention Drops: {attention_drops}
"""

    tk.Label(root,
             text=stats,
             font=("Helvetica", 12),
             bg=pastel_pink,
             justify="left").pack(pady=10)

    tk.Label(root,
             text=f"Overall Rating: {rating}",
             font=("Helvetica", 14, "bold"),
             fg=color,
             bg=pastel_pink).pack(pady=20)

    ttk.Button(root, text="Close", command=root.destroy).pack(pady=10)

    root.mainloop()


# =========================
# MAIN LOOP
# =========================

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]

        left_iris = landmarks.landmark[468]
        right_iris = landmarks.landmark[473]
        nose = landmarks.landmark[1]

        eye_center = (left_iris.x + right_iris.x) / 2
        head_position = nose.x

        # Calibration phase
        if frame_count < calibration_frames:
            baseline_eye = eye_center
            baseline_head = head_position

            cv2.putText(frame,
                        "Calibrating... Look Straight",
                        (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2)
        else:
            eye_dev = abs(eye_center - baseline_eye)
            head_dev = abs(head_position - baseline_head)

            confidence = calculate_confidence(eye_dev, head_dev)

            # Exponential smoothing
            smooth_conf = int(alpha * confidence + (1 - alpha) * previous_score)
            previous_score = smooth_conf

            confidence_scores.append(smooth_conf)

            if smooth_conf < 40:
                attention_drops += 1

            cv2.putText(frame,
                        f"Confidence: {smooth_conf}%",
                        (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 200, 0),
                        2)

            # Update Graph
            line.set_xdata(np.arange(len(confidence_scores)))
            line.set_ydata(confidence_scores)
            ax.set_xlim(0, max(100, len(confidence_scores)))
            fig.canvas.draw()
            fig.canvas.flush_events()

        frame_count += 1

    cv2.imshow("Confidence Analyzer", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# =========================
# CLEANUP
# =========================

cap.release()
cv2.destroyAllWindows()

if confidence_scores:
    show_summary_popup()
