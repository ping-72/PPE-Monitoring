from ultralytics import YOLO
import cv2 as cv
import cvzone
import math
from sort import *
import numpy as np
from collections import deque
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# cap = cv.VideoCapture(0)  # for web cam
cap = cv.VideoCapture("../videos/cars.mp4")  # for videos

model = YOLO("../YOLO weights/yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush", "middle finger"]

mask = cv.imread("mask-950x480.png")

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# creating limits for line to count after that
limits = [150, 450, 640, 450]
totalCount = []
carCountPerMinute = deque()
currentMinute = datetime.now().replace(second=0, microsecond=0)

# Traffic Congestion Threshold
congestion_threshold = 20  # Define a threshold for congestion

# making video size and image size same
w1 = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
h1 = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

h2, w2 = mask.shape[:2]
if h1 * w1 > h2 * w2:
    mask = cv.resize(mask, (w1, h1))
else:
    cap = cv.resize(cap, (w2, h2))


# Helper function to update the car count per minute
def update_car_count():
    global currentMinute
    now = datetime.now().replace(second=0, microsecond=0)
    if now > currentMinute:
        carCountPerMinute.append((currentMinute, len(totalCount)))
        totalCount.clear()
        currentMinute = now
        if len(carCountPerMinute) > 60:
            carCountPerMinute.popleft()


# Helper function to detect peak minutes and congestion
def detect_peak_minutes_and_congestion():
    if not carCountPerMinute:
        return "", False
    max_count = max(carCountPerMinute, key=lambda x: x[1])[1]
    peak_minute = [time for time, count in carCountPerMinute if count == max_count]
    congestion = max_count > congestion_threshold
    return peak_minute, congestion


# Timer start
start_time = datetime.now()

# main code
while True:
    success, vid = cap.read()
    if not success:
        break

    maskVid = cv.bitwise_and(vid, mask)

    vidGraphics = cv.imread("graphics.png", cv.IMREAD_UNCHANGED)  # masked img
    vid = cvzone.overlayPNG(vid, vidGraphics, (0, 0))

    results = model(maskVid, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2 - x1, y2 - y1
            bbox = x1, y1, w, h

            # confidence
            confidence = math.ceil(box.conf[0] * 100) / 100

            # class name
            cls = int(box.cls[0])
            currClass = classNames[cls]

            if currClass in ["car", "truck", "bus", "motorbike"] and confidence > 0.3:
                currentArray = np.array([x1, y1, x2, y2, confidence])
                detections = np.vstack((detections, currentArray))

    # tracker
    resultsTracker = tracker.update(detections)

    # making a line using the limits
    cv.line(vid, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        bbox = x1, y1, w, h
        cvzone.cornerRect(vid, bbox, l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(vid, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)

        # making center over object
        cx, cy = x1 + w // 2, y1 + h // 2
        cv.circle(vid, (cx, cy), 5, (255, 0, 255), cv.FILLED)

        # counting the centers after it crossed the limit
        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv.line(vid, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    # Update car count and check for peak minutes and congestion
    update_car_count()
    peak_minute, congestion = detect_peak_minutes_and_congestion()

    # Calculate elapsed time
    elapsed_time = datetime.now() - start_time
    elapsed_seconds = int(elapsed_time.total_seconds())

    # Display traffic flow information at the bottom right
    cv.putText(vid, str(len(totalCount)), (255, 100), cv.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)
    cv.putText(vid, f'Current Date-Time: {currentMinute.strftime("%Y-%m-%d %H:%M")}',
               (600, 500), cv.FONT_HERSHEY_PLAIN, 2, (50, 50, 255), 2)
    cv.putText(vid, f'Timer: {str(timedelta(seconds=elapsed_seconds))}', (600, 540),
               cv.FONT_HERSHEY_PLAIN, 2, (50, 50, 255), 2)
    if peak_minute:
        cv.putText(vid, f'Peak Minute: {peak_minute[0].strftime("%Y-%m-%d %H:%M")}',
                   (600, 580), cv.FONT_HERSHEY_PLAIN, 2, (50, 50, 255), 2)
    if congestion:
        cv.putText(vid, 'Congestion Detected!', (600, 620), cv.FONT_HERSHEY_PLAIN, 2,
                   (0, 0, 255), 2)

    cv.imshow("Video", vid)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()


# Plotting the traffic flow data
def plot_traffic_flow(data):
    times, counts = zip(*data)
    plt.figure(figsize=(12, 6))
    plt.plot(times, counts, marker='o')
    plt.xlabel('Time')
    plt.ylabel('Car Count')
    plt.title('Traffic Flow Per Minute')
    plt.xticks(rotation=45)
    plt.grid(True)

