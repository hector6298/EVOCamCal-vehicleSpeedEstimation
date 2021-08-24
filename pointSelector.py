import numpy as np
import cv2
import json
import argparse

#Macro for points container
refPt = []

#Callback for click events
def pointSelector(event, x, y, flags, param):
	global refPt, cropping

	if event == cv2.EVENT_LBUTTONDOWN:
		refPt.append((x, y))
        #cv2.circle(image, (x,y), 1, (255,255,0))
	elif event == cv2.EVENT_LBUTTONUP:
		cv2.imshow("image", image)

def lineSelector(event, x, y, flags, param):
    global refPt, cropping
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
        cv2.line(image, refPt[0], (x,y), (0, 255, 0), 2) 
    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        cropping = False
        cv2.line(image, refPt[0], refPt[1], (0, 255, 0), 2)        
        cv2.imshow("image", image)

#argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--number", required=True, help="Number of the video instance")
ap.add_argument("-m", "--mode", required=True, help="point or line")

args = vars(ap.parse_args())
print(args["number"])

#Capture the first video frame
cap = cv2.VideoCapture(f"source_material/streetVideos/seattle{args['number']}.mp4")
if (cap.isOpened()== False): 
  raise Exception("Error opening video stream or file")
ret, image = cap.read()

clone = image.copy()
cv2.namedWindow("image")

if args["mode"] == "point":
    cv2.setMouseCallback("image", pointSelector)

elif args["mode"] == "line":
    cv2.setMouseCallback("image", lineSelector)

else:
    raise Exception(f"Mode {args['mode']} not supported! Abort...")

while True:
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF
    

    if key == ord("r"):
        image = clone.copy()

    elif key == ord("c"):
        break

#save to JSON file
cv2.destroyAllWindows()

if args["mode"] == "point":
    with open("source_material/reference_pts/points.json") as f:
        data = json.load(f)
    with open("source_material/reference_pts/points.json", 'w') as f:
        data["points"][args["number"]]["2dpts"] = refPt
        json.dump(data, f, indent=2)

    cv2.imwrite(f"source_material/imgs/seattle{args['number']}.jpg", image)

elif args["mode"] == "line":
    with open("source_material/speedDetectionSpots/detectionSpots.json") as f:
        data = json.load(f)

    with open("source_material/speedDetectionSpots/detectionSpots.json", 'w') as f:
        data[args['number']]= refPt
        json.dump(data, f, indent=2)
    
    cv2.imwrite(f"source_material/spotImages/line{args['number']}.jpg", image)

