import cv2

path = "C:/Users/LENOVO/OneDrive/Desktop/DS Projects/SMARTAN FITTECH PVT/PoseDetector/data/samples/test.mp4"

cap = cv2.VideoCapture(path)

print("Video path:", path)

if not cap.isOpened():
    print("ERROR: Cannot open video!")
else:
    print("SUCCESS: Video opened!")

ret, frame = cap.read()
print("First frame read:", ret)

cap.release()
