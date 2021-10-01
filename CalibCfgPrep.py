import numpy as np
import cv2
import json
import argparse


#argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True, help="Number of the video instance")
ap.add_argument("-m", "--mode", required=True, help="Calibration Model 0 Base, 1 RANSAC, 2 EDA")


args = vars(ap.parse_args())
print(f"Setting Configuration for video {args['video']} using mode {args['mode']}")

# Ensure mode is supported
if args['mode'] not in ["0", "1", "2"]:
    raise Exception(f"Mode {args['mode']} not supported. Abort")

# Load points dataset
with open("/home/hector/Documents/tesis/source_material/reference_pts/points.json") as f:
    data = json.load(f)

# Load Config Json
with open("/home/hector/Documents/tesis/manual_calib/data/cfg.json", "r") as f:
    dataCfg = json.load(f)

# Backup Config Json
with open("/home/hector/Documents/tesis/manual_calib/data/cfg_backup.json", 'w') as f:
    json.dump(dataCfg, f, indent=2)

# Mutate Config Json
with open("/home/hector/Documents/tesis/manual_calib/data/cfg.json", 'w') as f:

    dataCfg["camCal"]["cal2dPtLs"] = data["points"][args["video"]]["2dpts"]
    dataCfg["camCal"]["cal3dPtLs"] = data["points"][args["video"]]["pts"]

    dataCfg["genInfo"]['inFrmPth'] = f"/home/hector/Documents/tesis/source_material/imgs/seattle{args['video']}.jpg"

    if args['mode'] == "0":
        dataCfg["camCal"]["calTyp"] = 0
        dataCfg["camCal"]["edaOptFlg"] = 0
        dataCfg["genInfo"]["outCamMatPth"] = f"./data/base/calibration{args['video']}.txt"
        dataCfg["genInfo"]["outCalDispPth"] = f"./data/base/calibration{args['video']}.jpg"

    elif args['mode'] == "1":
        dataCfg["camCal"]["calTyp"] = 8
        dataCfg["camCal"]["edaOptFlg"] = 0
        dataCfg["genInfo"]["outCamMatPth"] = f"./data/RANSAC/calibration{args['video']}.txt"
        dataCfg["genInfo"]["outCalDispPth"] = f"./data/RANSAC/calibration{args['video']}.jpg"

    elif args['mode'] == "2":
        dataCfg["camCal"]["calTyp"] = 0
        dataCfg["camCal"]["edaOptFlg"] = 1
        dataCfg["genInfo"]["outCamMatPth"] = f"./data/EDA/calibration{args['video']}.txt"
        dataCfg["genInfo"]["outCalDispPth"] = f"./data/EDA/calibration{args['video']}.jpg"

    json.dump(dataCfg, f, indent=2)

print("Success")
