import cv2
import numpy as np
import json
import argparse

from joblib import load, dump
from workflowUtils.objectDetection import *

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--video", default=1,
                    help="Index of the video")
args = parser.parse_args()

VIDEOPATH = f"source_material/streetVideos/seattle{args.video}.mp4"


cap = cv2.VideoCapture(VIDEOPATH)
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
YOLOSIZE = (512,512)
FRAMESIZE = (height, width)

yolo = objectDetectorV4(YOLOSIZE,
                        FRAMESIZE,
                        weightPath="workflowUtils/yolov4/data/checkpoints/yolov4")


detection_container = []

while cap.isOpened():
    
  ret, frame = cap.read()
  
  if not ret:
    print("Can't receive frame (stream end?). Exiting ...")
    break

  img_in = cv2.resize(frame, YOLOSIZE)
  img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)

  detections = yolo.detect(img_in)
  
  bboxes, _, _, _ = detections
  if len(bboxes) > 0:
    detection_container.append(detections)

  image = yolo.draw_bbox(frame, detections)
  
  cv2.imshow('output', image)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()

dump(detection_container, f"results/detections/video{args.video}")