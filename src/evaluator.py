from utils import calculate_angle, euclidean_distance, is_valid_keypoint

class PoseEvaluator:
    """
    Evaluates human posture using YOLO pose keypoints.
    Keypoints follow COCO order (17 points).
    """

    # YOLO keypoint indexing
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

    def evaluate(self, keypoints):
        """
        Main evaluation method.
        keypoints: list of 17 (x, y) points for a single person.
        Returns a dictionary with posture results.
        """

        if keypoints is None or len(keypoints) != 17:
            return {"error": "Invalid keypoints"}

        # Extract commonly used joints
        ls = keypoints[self.LEFT_SHOULDER]
        rs = keypoints[self.RIGHT_SHOULDER]
        lh = keypoints[self.LEFT_HIP]
        rh = keypoints[self.RIGHT_HIP]
        lk = keypoints[self.LEFT_KNEE]
        rk = keypoints[self.RIGHT_KNEE]
        la = keypoints[self.LEFT_ANKLE]
        ra = keypoints[self.RIGHT_ANKLE]
        le = keypoints[self.LEFT_ELBOW]
        re = keypoints[self.RIGHT_ELBOW]
        lw = keypoints[self.LEFT_WRIST]
        rw = keypoints[self.RIGHT_WRIST]

        # Validate important keypoints
        important_points = [ls, rs, lh, rh, lk, rk, la, ra]
        if any(not is_valid_keypoint(p) for p in important_points):
            return {"error": "Missing keypoints"}

        # -----------------------------
        # 1. ELBOW ANGLES
        # -----------------------------
        left_elbow_angle = calculate_angle(ls, le, lw)
        right_elbow_angle = calculate_angle(rs, re, rw)

        # -----------------------------
        # 2. KNEE ANGLES
        # -----------------------------
        left_knee_angle = calculate_angle(lh, lk, la)
        right_knee_angle = calculate_angle(rh, rk, ra)

        # -----------------------------
        # 3. HIP ANGLE (back posture)
        # Shoulders → hips → ankles
        # -----------------------------
        left_hip_angle = calculate_angle(ls, lh, lk)
        right_hip_angle = calculate_angle(rs, rh, rk)

        # -----------------------------
        # 4. Back alignment check
        # Shoulder–Hip and Hip–Ankle vertical alignment
        # -----------------------------
        back_score = self.check_back_alignment(ls, lh, la)

        # -----------------------------
        # FINAL ASSESSMENT
        # -----------------------------
        assessment = {
            "left_elbow_angle": left_elbow_angle,
            "right_elbow_angle": right_elbow_angle,
            "left_knee_angle": left_knee_angle,
            "right_knee_angle": right_knee_angle,
            "left_hip_angle": left_hip_angle,
            "right_hip_angle": right_hip_angle,
            "back_alignment": back_score,
            "overall": self.generate_feedback(left_knee_angle, right_knee_angle, back_score)
        }

        return assessment

    # --------------------------------------------------------------------
    # CHECK BACK ALIGNMENT
    # --------------------------------------------------------------------
    def check_back_alignment(self, shoulder, hip, ankle):
        """
        Simple logic: checks if shoulder, hip, ankle are aligned
        by comparing vertical slope.
        """

        sx, sy = shoulder
        hx, hy = hip
        ax, ay = ankle

        # If x coordinates almost form a vertical line
        slope1 = abs(sx - hx)
        slope2 = abs(hx - ax)

        if slope1 < 15 and slope2 < 15:
            return "straight"
        elif slope1 < 40 and slope2 < 40:
            return "slightly bent"
        else:
            return "bent"

    # --------------------------------------------------------------------
    # OVERALL FEEDBACK LOGIC
    # --------------------------------------------------------------------
    def generate_feedback(self, left_knee, right_knee, back_status):
        """
        Combines multiple signals to generate simple feedback.
        """

        feedback = []

        # Knee depth example rules
        if left_knee and left_knee < 120:
            feedback.append("Left knee deep bend")
        if right_knee and right_knee < 120:
            feedback.append("Right knee deep bend")

        # Back posture
        if back_status == "bent":
            feedback.append("Straighten your back")
        elif back_status == "slightly bent":
            feedback.append("Back posture slightly off")

        # If no warnings → good posture
        if len(feedback) == 0:
            return "Good posture"

        return ", ".join(feedback)
