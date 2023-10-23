# Importing the dependencies.
from ultralytics import YOLO
import cv2
import cvzone
import math

# 0 for a single webcam and 1 for multiple webcams.
cap = cv2.VideoCapture(0) # For Webcam
#cap = cv2.VideoCapture('../Videos/people.mp4')
# Setting the width (propID-3)
cap.set(3, 1280)
# Setting the height (propID-4)
cap.set(4, 720)

model = YOLO("ppe.pt")

classNames = ['Gloves', 'Helmet', 'Non-Helmet', 'Person', 'Shoes', 'Vest', 'bare-arms']

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    # Each result is processed as a single detection
    for r in results:
        # Extract bounding box coordinates for a single detected object
        boxes = r.boxes
        for box in boxes:
            # Obtaining the bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # print(x1, y1, x2, y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            w, h = x2 - x1, y2 - y1
            x1, y1, w, h = int(x1), int(y1), int(w), int(h)
            cvzone.cornerRect(img, (x1, y1, w, h))

            # Confidence
            conf = (math.ceil(box.conf[0] * 100)) / 100
            print(conf)

            # Class Name
            cls = int(box.cls[0])

            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
