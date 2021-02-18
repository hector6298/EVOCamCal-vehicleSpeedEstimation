import cv2
import numpy as np
import torchvision.transforms as T
import matplotlib.pyplot as plt

from workflowUtils.objectDetection import *
from workflowUtils.tracking import *
from workflowUtils.carMasking import *

VIDEOPATH = "/home/hector/seattle2_corrected.mp4"
FRAMESIZE = (512,512)
PARAMFILE = "manual_calib/data/calibration.txt"
TYPE = 0


yolo = objectDetector(FRAMESIZE,
                       weightPath="workflowUtils/yolov3/weights/yolov3.weights",
                       modelDef="workflowUtils/yolov3/config/yolov3.cfg")
viz = objectVisualizer(classPath="workflowUtils/yolov3/data/coco.names")
tools = objectFocusTools(FRAMESIZE,yolo)

cap = cv2.VideoCapture(VIDEOPATH)
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

tracker = Tracker(height, width,20, params_file=PARAMFILE)

while(cap.isOpened()):
    
  ret, frame = cap.read()
  frame = cv2.resize(frame, (512,512))
  img_in = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  grayFrame = convertImg2Grayscale(frame)

  detections = yolo.detect(img_in)
  #if detections is not None:
  bboxes = detections[:,0:4]
  objects, velocities = tracker.update(bboxes, fps)

  for objectID, centroid in objects.items():
    #qif velocities[objectID] > 0:
    text = "{:.2f} km/h".format(velocities[objectID])
    cv2.putText(frame, text, (centroid[0]-10, centroid[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    cv2.circle(frame, (centroid[0], centroid[1]), 4, (0,255,0), -1)

    cv2.imshow('output', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()