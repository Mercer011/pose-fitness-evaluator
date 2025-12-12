import cv2
from ultralytics import YOLO

class PoseDetector:
    def __init__(self, model_path="yolov8n-pose.pt"):
        """
        Constructor: loads the YOLO pose model.
        
        model_path: name of the YOLO pose weights file.
                    'yolov8n-pose.pt' = nano model (fastest, lightest)
        """
        self.model = YOLO(model_path)   # Load YOLO pose model

    def detect(self, frame):
        """
        Runs YOLO pose detection on a single frame.

        Input:
            frame -> a BGR image from OpenCV

        Output:
            persons_keypoints -> list of keypoints for each detected person
                                 Each person = array of 17 (x,y) keypoints
            results[0]        -> the raw YOLO output (optional for drawing)
        """

        # YOLO works directly with BGR frames, no need to convert to RGB.
        results = self.model(frame, verbose=False)

        # Extract all detected persons from the first result
        persons_keypoints = []

        # YOLO returns a results list even for a single image.
        # results[0].keypoints.xy is a tensor of shape:
        #   number_of_persons × 17 × 2
        # Each (x, y) already in pixel coordinates.
        if results and results[0].keypoints is not None:
            for person in results[0].keypoints.xy:
                # 'person' is a tensor with 17 rows → one per keypoint
                # convert tensor to plain python list
                kp_list = person.cpu().numpy().tolist()
                persons_keypoints.append(kp_list)

        return persons_keypoints, results[0]
