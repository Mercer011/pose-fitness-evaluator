import cv2
from detector import PoseDetector

detector = PoseDetector()

# USE YOUR WORKING ABSOLUTE PATH
cap = cv2.VideoCapture("C:/Users/LENOVO/OneDrive/Desktop/DS Projects/SMARTAN FITTECH PVT/PoseDetector/data/samples/test.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    keypoints_list, raw_result = detector.detect(frame)

    print("Frame processed — persons detected:", len(keypoints_list))

    if len(keypoints_list) > 0:
        print("First person's first 3 keypoints:", keypoints_list[0][:3])

    cv2.imshow("Pose Detection Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
