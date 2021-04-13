import numpy as np
import sklearn
import cv2
import umap
import umap
from joblib import dump, load
import argparse

from workflowUtils.objectDetection import *

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--video", default=1,
                    help="Index of the video")
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

i = 60*1*fps
dets = 0
histograms = []

while i and cap.isOpened():

  if i % 11 != 0: 
    i -= 1
    continue

  ret, frame = cap.read()
  img_in = cv2.resize(frame, YOLOSIZE)
  img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)

  detections = yolo.detect(img_in)
  image = yolo.draw_bbox(frame, detections)
  out_boxes, _, _, num_boxes = detections
  bboxes = out_boxes[0][out_boxes[0][:,0] > 0.]
  
  #Extract pieces of the image for each detection here
  for bbox in bboxes:
    x1,y1,x2,y2 = int(bbox[1]), int(bbox[0]), int(bbox[3]), int(bbox[2])
    frame_piece = frame[y1:y2+1, x1:x2+1]
    det_hist_channels = []
    for channel in range(3):
        histr = np.squeeze(cv2.calcHist([frame_piece],[channel],None,[256],[0,256]))
  
        det_hist_channels.extend(histr)
    dets += 1
    histograms.append(det_hist_channels)
  print(i, ", detected: ", dets)
  i -= 1

histograms = np.array(histograms)
print(histograms.shape)
reducer = umap.UMAP(n_components=15)
reduced_histograms = reducer.fit_transform(histograms)

dump(reducer, f'models/reducer{args.video}.joblib') 