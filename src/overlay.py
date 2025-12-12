import cv2
import numpy as np

# YOLOv8 Pose keypoint order (17 points)
# 0: nose, 1: left eye, 2: right eye, 3: left ear, 4: right ear
# 5: left shoulder, 6: right shoulder, 7: left elbow, 8: right elbow
# 9: left wrist, 10: right wrist, 11: left hip, 12: right hip
# 13: left knee, 14: right knee, 15: left ankle, 16: right ankle

SKELETON_CONNECTIONS = [
    (5, 7), (7, 9),          # Left arm
    (6, 8), (8, 10),         # Right arm
    (5, 6),                  # Shoulders
    (11, 12),                # Hips
    (5, 11), (6, 12),        # Torso
    (11, 13), (13, 15),      # Left leg
    (12, 14), (14, 16)       # Right leg
]

class OverlayVisualizer:
    def __init__(self):
        pass

    def draw_keypoints(self, frame, keypoints):
        """Draw circles on each detected keypoint."""
        for idx, (x, y) in enumerate(keypoints):
            if x > 0 and y > 0:
                cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
        return frame

    def draw_skeleton(self, frame, keypoints):
        """Draw lines connecting keypoints based on skeleton structure."""
        for p1, p2 in SKELETON_CONNECTIONS:
            x1, y1 = keypoints[p1]
            x2, y2 = keypoints[p2]
            if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        return frame

    def draw_feedback(self, frame, eval_data):
        """
        Draw text feedback from evaluator on video.
        eval_data is a dictionary:
        {
            "left_elbow_angle": 145,
            "back_alignment": "straight",
            "overall": "Good posture"
        }
        """
        y = 30
        for key, value in eval_data.items():
            text = f"{key}: {value}"
            cv2.putText(frame, text, (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y += 25

        return frame

    def visualize(self, frame, keypoints, eval_data):
        """Master function to draw everything on the frame."""
        frame = self.draw_keypoints(frame, keypoints)
        frame = self.draw_skeleton(frame, keypoints)
        frame = self.draw_feedback(frame, eval_data)
        return frame
