import cv2
import numpy as np
import json
import argparse

from workflowUtils.objectDetection import *
from workflowUtils.tracking import *

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--video", default=1,
                    help="Index of the video")
parser.add_argument("-m", "--metric", default='euclidean',
                    help="metric for the tracking backend")
args = parser.parse_args()

VIDEOPATH = f"source_material/streetVideos/seattle{args.video}.mp4"
PARAMFILE = f"manual_calib/data/RANSAC/calibration{args.video}.txt"


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

tracker = Tracker(maxlost=20, 
                  distance_sigma=80,
                  metric=args.metric, reducer=False,
                  video=args.video, params_file=PARAMFILE)

while cap.isOpened():
    
  ret, frame = cap.read()

  if not ret:
    print("Can't receive frame (stream end?). Exiting ...")
    break

  img_in = cv2.resize(frame, YOLOSIZE)
  img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)

  detections = yolo.detect(img_in)
  bboxes, _, _, _ = detections
  
  #print(bboxes)
  image = yolo.draw_bbox(frame, detections)
  
 
  if len(bboxes) > 0:

    objects, velocities = tracker.update( bboxes, fps)
    image = tracker.write_velocities(image)
   
  
  cv2.imshow('output', image)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()



# Save all the records of velocities of all vehicles
print(tracker.objsVelHist.keys())
dump(tracker.objsVelHist, 
    f'results/velocities/velocities/{args.metric}/{args.video}')

# save the records of all diagonal bounding box lengths for all vehicles
#print(type(tracker.objsBBdist), type(tracker.objsVelHist))
print(tracker.objsBBdist.keys())
dump(str(tracker.objsBBdist),
    f'results/bbox_distances/bboxes/{args.metric}/{args.video}')

#print(tracker.velocitiesInLine)
# Save all the velocities captured in the virtual line
dump(tracker.velocitiesInLine, 
     f'results/spot_velocities/spot_velocities{args.metric}/{args.video}')

print("Success!!")