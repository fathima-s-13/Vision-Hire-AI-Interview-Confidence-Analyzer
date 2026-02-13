import numpy as np

def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def eye_aspect_ratio(eye_points):
    vertical_1 = calculate_distance(eye_points[1], eye_points[5])
    vertical_2 = calculate_distance(eye_points[2], eye_points[4])
    horizontal = calculate_distance(eye_points[0], eye_points[3])
    return (vertical_1 + vertical_2) / (2.0 * horizontal)

def posture_score(left_shoulder, right_shoulder):
    return abs(left_shoulder[1] - right_shoulder[1])
