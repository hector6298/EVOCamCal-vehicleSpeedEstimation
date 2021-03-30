import numpy as np
import cv2
import json
import argparse

#Macro for points container
refPt = []

#Callback for click events
def click_and_crop(event, x, y, flags, param):
	global refPt, cropping

	if event == cv2.EVENT_LBUTTONDOWN:
		refPt.append((x, y))
        #cv2.circle(image, (x,y), 1, (255,255,0))
	elif event == cv2.EVENT_LBUTTONUP:
		cv2.imshow("image", image)

#argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True, help="Path to the video")
ap.add_argument("-f", "--jfile", required=True, help="Path to the output data")
ap.add_argument("-n", "--number", required=True, help="Number of the video instance")
ap.add_argument("-o", "--image", required=True, help="Output path for image")


args = vars(ap.parse_args())

#Capture the first video frame
cap = cv2.VideoCapture(args["video"])
if (cap.isOpened()== False): 
  raise Exception("Error opening video stream or file")
ret, image = cap.read()

clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

while True:
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF
    

    if key == ord("r"):
        imga = clone.copy()
    elif key == ord("c"):
        break

#save to JSON file
cv2.destroyAllWindows()
print(args["jfile"])
with open(args['jfile']) as f:
    data = json.load(f)
with open(args['jfile'], 'w') as f:
    data["points"][args["number"]]["2dpts"] = refPt
    json.dump(data, f, indent=2)

cv2.imwrite(args["image"], image)