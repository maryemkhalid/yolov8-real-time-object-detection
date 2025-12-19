"""
YOLOv8-based real-time object detection on video streams.
Author: Maryam Khalid
"""

import cv2
import random
from ultralytics import YOLO
import opencv_jupyter_ui as jcv2

#Load Yolov8 model
yolo = YOLO("yolov8s.pt")

def getColours(cls_num):
    """Generate unique colors for each class ID"""
    random.seed(cls_num)
    return tuple(random.randint(0, 255) for _ in range(3))
  
#Input Video Path
video_path = "sample.mp4"
videoCap = cv2.VideoCapture(video_path)

frame_count = 0

while True:
    ret, frame = videoCap.read()   #videoCap.read(): Reads one frame from video.
    if not ret:
        break
    #Run YOLOv8 tracking  
    results = yolo.track(frame, stream=True)     

    for result in results:
        class_names = result.names
        for box in result.boxes:           #result.boxes: Contains bounding boxes for detected objects.
            if box.conf[0] > 0.4:                #box.conf[0]:Confidence score of detection.
                x1, y1, x2, y2 = map(int, box.xyxy[0])  #box.xyxy[0]: Coordinates of bounding box (x1,y1,x2,y2).

                cls = int(box.cls[0])
                class_name = class_names[cls]

                conf = float(box.conf[0])           

                colour = getColours(cls)

                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)     #cv2.rectangle(): Draws bounding box.

                cv2.putText(frame, f"{class_name} {conf:.2f}",           #cv2.putText(): Adds class name + confidence on frame.
                            (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, colour, 2)

    if frame_count < 20:
        jcv2.imshow(ret, frame)
    else:
        break

    frame_count += 1

videoCap.release()
