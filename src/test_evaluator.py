import cv2
from detector import PoseDetector
from evaluator import PoseEvaluator

video_path = r"C:\Users\LENOVO\OneDrive\Desktop\DS Projects\SMARTAN FITTECH PVT\PoseDetector\data\samples\test.mp4"

detector = PoseDetector()
evaluator = PoseEvaluator()

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Cannot open video!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    persons, raw = detector.detect(frame)

    if len(persons) > 0:
        keypoints = persons[0]      # first person

        result = evaluator.evaluate(keypoints)
        print("Evaluation:", result)     # <-- YOU WILL GET DICTIONARY OUTPUT HERE

    cv2.imshow("Evaluator Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
