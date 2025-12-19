# YOLOv8 Real-Time Object Detection

## Overview
This project implements a real-time object detection pipeline using YOLOv8.  
It can detect multiple objects in video streams with bounding boxes and confidence scores.

## Objectives
- Implement a robust real-time object detection pipeline.
- Demonstrate object detection using video data.
- Explore applications in safety monitoring and other domains.

## Technologies Used
- Python
- YOLOv8 (Ultralytics)
- OpenCV
- opencv-jupyter-ui

## Methodology
1. Load YOLOv8 pre-trained model (`yolov8s.pt`).
2. Read video frames from input video.
3. Perform real-time detection using `yolo.track()`.
4. Draw bounding boxes and class labels on detected objects.
5. Display output frames.

## Demo / Dataset
Due to data availability, a public road traffic video is used for demonstration purposes.  
The same detection pipeline is directly applicable to safety-critical monitoring scenarios, including industrial safety or other environments.

## How to Run
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Place a video file in the data/ directory.
3. Run the detection script:
```bash
python src/yolov8_video_detection.py
```

## Project Status
Initial implementation completed; repository documentation finalized.

## Author
Maryam Khalid
