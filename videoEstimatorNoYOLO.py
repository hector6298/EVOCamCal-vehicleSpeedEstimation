import cv2
import numpy as np
import json
import argparse
from joblib import load, dump

from workflowUtils.tracking import *
from workflowUtils.imgFeatureUtils import draw_bbox
from workflowUtils.yolov4.core.config import cfg

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--video", default=11,
                    help="Index of the video")
parser.add_argument("-m", "--metric", default='euclidean',
                    help="metric for the tracking backend")
args = parser.parse_args()

VIDEOPATH = f"source_material/streetVideos/seattle{args.video}.mp4"
PARAMFILE = f"manual_calib/data/RANSAC/calibration{args.video}.txt"

# Load the capture line endpoints to measure speed at that virtual line
with open('source_material/speedDetectionSpots/detectionSpots.json') as json_file:
  pt1, pt2 = json.load(json_file)[args.video]
xs, ys = (pt1[0], pt2[0]), (pt1[1], pt2[1])

def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

classes = read_class_names(cfg.YOLO.CLASSES)

cap = cv2.VideoCapture(VIDEOPATH)
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
YOLOSIZE = (512,512)
FRAMESIZE = (height, width)


tracker = Tracker(xs=xs,ys=ys,sigma_h=0.9,sigma_iou=0.7, metric=args.metric,
                      t_min=10, params_file=PARAMFILE, frameRate=fps)

all_detections = load(f"results/detections/video{args.video}")

i = 0

for frame_detections in all_detections:
    
  ret, frame = cap.read()
  img_in = cv2.resize(frame, YOLOSIZE)
  img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
  bboxes, scores, classes, num_dets = frame_detections
  #print(bboxes)
  image = draw_bbox(frame, frame_detections)
  if len(frame_detections[0]) > 0:
    tracker.update(bboxes, i)
    image = tracker.write_velocities(image)
   
  cv2.imshow('output', image)

  i += 1
  print(i)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()


# Save all the records of velocities of all vehicles
dump(tracker.getVelocityHistory(), 
    f'results/velocities/{args.metric}/velocities{args.video}')

# save the records of all diagonal bounding box lengths for all vehicles
#print(type(tracker.objsBBdist), type(tracker.objsVelHist))
dump(str(tracker.getBBDists()),
    f'results/bbox_distances/{args.metric}/bbox_distances{args.video}')

#print(tracker.velocitiesInLine)
# Save all the velocities captured in the virtual line
dump(tracker.getVelocitiesInLine(), 
     f'results/spot_velocities/{args.metric}/spot_velocities{args.video}')

print("Success!!")