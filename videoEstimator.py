import cv2
import numpy as np
import json

from workflowUtils.objectDetection import *
from workflowUtils.tracking import *
import workflowUtils.yolov4.core.utils as utils
VIDEOPATH = "documents/streetVideos/seattle11.mp4"

PARAMFILE = "manual_calib/data/RANSAC/calibration11.txt"
TYPE = 0

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

tracker = Tracker(height, width,20, params_file=PARAMFILE)

i = 60*2*fps

while i:
    
  ret, frame = cap.read()
  img_in = cv2.resize(frame, YOLOSIZE)
  img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)

  detections = yolo.detect(img_in)
  out_boxes, _, _, num_boxes = detections
  bboxes = out_boxes[0][out_boxes[0][:,0] > 0.]
  print(bboxes)
  image = yolo.draw_bbox(frame, detections)
  
  if num_boxes[0] > 0 and out_boxes[0] is not None:

    objects, velocities = tracker.update(bboxes, fps)
    image = tracker.write_velocities(image)
   
  
  cv2.imshow('output', image)

  i -= 1
  print(i)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()

with open('velocities7.json', 'w') as fp:
  json.dump(tracker.objVelHist , fp)

