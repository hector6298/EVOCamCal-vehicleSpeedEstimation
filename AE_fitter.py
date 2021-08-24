
import numpy as np
import sklearn
import cv2
import argparse
import json

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from workflowUtils.objectDetection import objectDetectorV4
from workflowUtils.autoencoderSpec import Autoencoder

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--video", default=1,
                    help="Index of the video")
parser.add_argument("-e", "--epochs", default=20,
                    help="Autoencoder training epochs")
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

i = 60*2*fps
dets = 0
histograms = []

while i and cap.isOpened():

  ret, frame = cap.read()

  if i % 11 != 0: 
    i -= 1
    continue

  frame_o = frame.copy()
  img_in = cv2.resize(frame, YOLOSIZE)
  img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
  
  detections = yolo.detect(img_in)
  image = yolo.draw_bbox(frame, detections)
  bboxes, _, _, _ = detections

  #Extract pieces of the image for each detection here
  for bbox in bboxes:
    x1,y1,x2,y2 = int(bbox[1]), int(bbox[0]), int(bbox[3]), int(bbox[2])
    frame_piece = frame_o[y1:y2+1, x1:x2+1]

    det_hist_channels = []
    for channel in range(3):
        histr = np.squeeze(cv2.calcHist([frame_piece],[channel],None,[256],[0,256]))
  
        det_hist_channels.extend(histr)
    dets += 1
    histograms.append(det_hist_channels)
  print(i, ", detected: ", dets)
  i -= 1


histograms = np.array(histograms)
print("final shape of all detections: ", histograms.shape)
x_train, x_test = train_test_split(
          histograms, test_size=0.25, random_state=42)

print("fitting Autoencoder reductor")
reducer = Autoencoder(latent_dim=20)
reducer.compile(optimizer='adam', loss=losses.MeanSquaredError())
metrics_history = reducer.fit(x_train, x_train,
                                epochs=int(args.epochs),
                                shuffle=True,
                                validation_data=(x_test, x_test))

print("Save model and metrics")
reducer.save_weights(f"AE_models/autoencoder{args.video}.h5")

reducer = Autoencoder(latent_dim=20)
reducer.build((None,256*3))
reducer.load_weights(f"AE_models/autoencoder{args.video}.h5")

with open(f'results/AE_fits/fit{args.video}.json', 'w') as fp:
    json.dump(metrics_history.history, fp)

with open(f'results/detections_in_fits.txt', 'w') as fp:
  fp.write(f"{args.video},{dets}\n")

print("done!")