import cv2
import numpy as np
import json

from workflowUtils.objectDetection import *
from workflowUtils.tracking import *

VIDEOPATH = "documents/streetVideos/seattle11.mp4"

PARAMFILE = "manual_calib/data/RANSAC/calibration11.txt"
TYPE = 0

cap = cv2.VideoCapture(VIDEOPATH)
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

YOLOSIZE = (256,256)
FRAMESIZE = (height, width)

yolo = objectDetector(YOLOSIZE,
                      FRAMESIZE,
                       weightPath="workflowUtils/yolov3/weights/yolov3.weights",
                       modelDef="workflowUtils/yolov3/config/yolov3.cfg")
viz = objectVisualizer(classPath="workflowUtils/yolov3/data/coco.names")

tracker = Tracker(height, width,20, params_file=PARAMFILE)

i = 60*2*fps

while i:
    
  ret, frame = cap.read()
  img_in = cv2.resize(frame, YOLOSIZE)
  img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
  grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  detections = yolo.detect(img_in)

  if len(detections) > 0 and detections[0] is not None:
    print(detections)
    bboxes = np.array(detections)[:,0:4]
    objects, velocities = tracker.update(bboxes, fps)

    for objectID, centroid in objects.items():
      #qif velocities[objectID] > 0:
      text = "{}: {:.2f} km/h".format(objectID, velocities[objectID])
      cv2.putText(frame, text, (centroid[0]-10, centroid[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
      cv2.circle(frame, (centroid[0], centroid[1]), 4, (255,0,0), -1)

  cv2.imshow('output', frame)

  i -= 1
  print(i)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()

with open('velocities7.json', 'w') as fp:
  json.dump(tracker.objVelHist , fp)

