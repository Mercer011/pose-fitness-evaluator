import cv2
from detector import PoseDetector
from evaluator import PoseEvaluator
from overlay import OverlayVisualizer

video_path = r"C:\Users\LENOVO\OneDrive\Desktop\DS Projects\SMARTAN FITTECH PVT\PoseDetector\data\samples\test.mp4"
output_path = r"C:\Users\LENOVO\OneDrive\Desktop\DS Projects\SMARTAN FITTECH PVT\PoseDetector\outputs\demo_output.mp4"

detector = PoseDetector()
evaluator = PoseEvaluator()
visualizer = OverlayVisualizer()

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Cannot open video!")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30

out = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    persons, raw = detector.detect(frame)

    if len(persons) > 0:
        keypoints = persons[0]
        eval_data = evaluator.evaluate(keypoints)
        frame = visualizer.visualize(frame, keypoints, eval_data)

    out.write(frame)
    cv2.imshow("Pose Evaluation System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
