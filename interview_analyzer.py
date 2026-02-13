import cv2
import mediapipe as mp
import numpy as np

class InterviewAnalyzer:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.mp_drawing = mp.solutions.drawing_utils

        # Iris indices
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        self.LEFT_EYE_CORNER = [33, 133]
        self.RIGHT_EYE_CORNER = [362, 263]

        # Head pose landmark indices
        self.HEAD_POSE_POINTS = [1, 152, 33, 263, 61, 291]

        self.confidence_score = 50

    # ---------------- IRIS POSITION ----------------
    def get_iris_position(self, landmarks, iris_indices, w, h):
        iris_points = []
        for idx in iris_indices:
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            iris_points.append((x, y))
        return np.mean(iris_points, axis=0)

    # ---------------- EYE CONTACT SCORE ----------------
    def eye_contact_score(self, landmarks, w, h):
        left_iris = self.get_iris_position(landmarks, self.LEFT_IRIS, w, h)
        right_iris = self.get_iris_position(landmarks, self.RIGHT_IRIS, w, h)

        left_corner1 = int(landmarks[self.LEFT_EYE_CORNER[0]].x * w)
        left_corner2 = int(landmarks[self.LEFT_EYE_CORNER[1]].x * w)

        right_corner1 = int(landmarks[self.RIGHT_EYE_CORNER[0]].x * w)
        right_corner2 = int(landmarks[self.RIGHT_EYE_CORNER[1]].x * w)

        left_ratio = (left_iris[0] - left_corner1) / (left_corner2 - left_corner1 + 1e-6)
        right_ratio = (right_iris[0] - right_corner1) / (right_corner2 - right_corner1 + 1e-6)

        avg_ratio = (left_ratio + right_ratio) / 2

        if 0.4 <= avg_ratio <= 0.6:
            return 100
        else:
            deviation = abs(avg_ratio - 0.5)
            return max(0, 100 - deviation * 250)

    # ---------------- HEAD POSE ----------------
    def estimate_head_pose(self, landmarks, w, h):
        image_points = []
        for idx in self.HEAD_POSE_POINTS:
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            image_points.append((x, y))

        image_points = np.array(image_points, dtype="double")

        model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0)
        ])

        focal_length = w
        center = (w / 2, h / 2)

        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs
        )

        rmat, _ = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        pitch = angles[0]
        yaw = angles[1]
        roll = angles[2]

        return pitch, yaw, roll

    # ---------------- CONFIDENCE SCORE ----------------
    def compute_confidence(self, eye_score, yaw, pitch, w, h, landmarks):

        # Head Stability (Normalized)
        yaw_penalty = min(abs(yaw), 30) / 30
        pitch_penalty = min(abs(pitch), 30) / 30

        head_score = 100 - ((yaw_penalty + pitch_penalty) / 2) * 40

        # Face Centering
        nose_x = landmarks[1].x * w
        center_offset = abs(nose_x - w / 2)
        center_ratio = center_offset / (w / 2)

        center_score = 100 - (center_ratio * 30)

        # Weighted combination
        raw_score = (
            eye_score * 0.5 +
            head_score * 0.25 +
            center_score * 0.25
        )

        # Smooth (exponential averaging)
        self.confidence_score = 0.85 * self.confidence_score + 0.15 * raw_score

        return int(self.confidence_score)

    # ---------------- PROCESS FRAME ----------------
    def process_frame(self, frame):
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:

                self.mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    self.mp_face_mesh.FACEMESH_TESSELATION
                )

                eye_score = self.eye_contact_score(
                    face_landmarks.landmark, w, h
                )

                pitch, yaw, roll = self.estimate_head_pose(
                    face_landmarks.landmark, w, h
                )

                confidence = self.compute_confidence(
                    eye_score, yaw, pitch, w, h,
                    face_landmarks.landmark
                )

                # Display
                cv2.putText(frame, f"Confidence: {confidence}%",
                            (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)

                cv2.putText(frame, f"Yaw: {int(yaw)}",
                            (30, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 0), 2)

                cv2.putText(frame, f"Pitch: {int(pitch)}",
                            (30, 110), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 0), 2)

                cv2.putText(frame, f"Roll: {int(roll)}",
                            (30, 140), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 0), 2)

        return frame
