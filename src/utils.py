import math
import numpy as np

def calculate_angle(a, b, c):
    """
    Calculates the angle ABC (in degrees)
    where:
        a, b, c are (x, y) coordinates.
        b is the vertex.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # Vectors BA and BC
    ba = a - b
    bc = c - b

    # Dot product and magnitudes
    dot = np.dot(ba, bc)
    mag_ba = np.linalg.norm(ba)
    mag_bc = np.linalg.norm(bc)

    # Protect against division-by-zero
    if mag_ba == 0 or mag_bc == 0:
        return None

    # Compute angle
    cosine_angle = dot / (mag_ba * mag_bc)

    # Numerical stability (cosine value must be between -1 & 1)
    cosine_angle = max(min(cosine_angle, 1.0), -1.0)

    angle = math.degrees(math.acos(cosine_angle))
    return angle


def euclidean_distance(p1, p2):
    """
    Returns distance between 2 keypoints.
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    return np.linalg.norm(p1 - p2)


def is_valid_keypoint(point, threshold=1):
    """
    Check if a keypoint is valid.
    YOLO keypoints sometimes are:
        [0, 0]   -> not detected
        very low values
    """
    x, y = point
    return not (x < threshold and y < threshold)


def smooth_landmarks(prev_points, current_points, alpha=0.7):
    """
    Exponential smoothing:
        smoothed = alpha * previous + (1 - alpha) * current
    Helps reduce jitter.
    """
    if prev_points is None:
        return current_points

    prev = np.array(prev_points)
    curr = np.array(current_points)

    smoothed = alpha * prev + (1 - alpha) * curr
    return smoothed.tolist()
