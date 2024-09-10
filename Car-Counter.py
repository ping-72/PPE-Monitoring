from ultralytics import YOLO
import cv2 as cv
import cvzone
import math
from sort import*

# cap = cv.VideoCapture(0)  # for web cam
cap = cv.VideoCapture("../videos/cars.mp4")  # for videos
# cap.set(3, 1080)
# cap.set(4, 720)

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
limits = [180, 450, 640, 450]
totalCount = []

# making video size and image size same
w1 = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
h1 = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

h2, w2 = mask.shape[:2]
if h1 * w1 > h2 * w2:
    mask = cv.resize(mask, (w1, h1))
else:
    cap = cv.resize(cap, (w2, h2))

# main code
while True:
    success, vid = cap.read()
    maskVid = cv.bitwise_and(vid, mask)

    vidGraphics = cv.imread("graphics.png", cv.IMREAD_UNCHANGED)  # car banner
    vid = cvzone.overlayPNG(vid, vidGraphics, (0, 0))

    results = model(maskVid, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # print(x1, y1, x2, y2)
            # cv.rectangle(vid, (x1, y1), (x2, y2), (255, 0, 255), 3)

            w, h = x2-x1, y2-y1
            bbox = x1, y1, w, h
            # cvzone.cornerRect(vid, bbox, l=9)
            # to make box fancy, we can use other function of it, just press ctrl and click cornerRect

            # confidence
            confidence = math.ceil(box.conf[0]*100)/100  # to display it to hundredth place (rounding off)
            # print(confidence)

            # class name
            cls = int(box.cls[0])
            currClass = classNames[cls]

            if currClass == "car" or currClass == "truck" or currClass == "bus"\
                    or currClass == "motorbike" and confidence > 0.3:
                # cvzone.putTextRect(vid, f'{currClass} {confidence}', (max(0, x1), max(30, y1)), scale=1, thickness=1,/
                # offset=3)
                # cvzone.cornerRect(vid, bbox, l=9, rt=2)  # (placing it here only to make box around in "if" cases)
                currentArray = np.array([x1, y1, x2, y2, confidence])
                detections = np.vstack((detections, currentArray))

    # tracker
    resultsTracker = tracker.update(detections)

    # making a line using the limits
    cv.line(vid, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
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

    # cvzone.putTextRect(img, f' Count: {len(totalCount)}', (50, 50))
    cv.putText(vid, str(len(totalCount)), (255, 100), cv.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)

    cv.imshow("Video", vid)
    # cv.imshow("Masked Video", maskVid)  # masked video
    cv.waitKey(1)