import os
import datetime
import torch
import cv2
import numpy as np

from PIL import Image
from torch.autograd import Variable

from .yolov3.models import *
from .yolov3.utils import *
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class objectDetector(object):
    """
    Object Detector high level object using YOLOv3 from https://github.com/eriklindernoren/PyTorch-YOLOv3.git
    """
    def __init__(self, imgSize, orgSize, weightPath, modelDef, confidenceThres=0.5, IoUThres=0.2):
        self.imgSize = imgSize
        self.confidenceThres = confidenceThres
        self.IoUThres = IoUThres
        self.orgSize = orgSize
        self.model = Darknet(modelDef, img_size=imgSize).to(device)
        self.tensorType = torch.cuda.FloatTensor if torch.cuda.is_available()\
                                                 else torch.FloatTensor
        self.trf = T.Compose([
                        T.ToTensor()
                        ])
        if weightPath.endswith(".weights"):
            # Load darknet weights
            self.model.load_darknet_weights(weightPath)
        else:
            # Load checkpoint weights
            self.model.load_state_dict(torch.load(weightPath))
        self.model.eval()

    def detect(self, img):
        imgPIL = Image.fromarray(img)
        inp = self.tensorType(self.trf(img).unsqueeze(0).cuda())
        with torch.no_grad():
            detections = self.model(inp)
            detections = non_max_suppression(detections, 
                                             self.confidenceThres, 
                                             self.IoUThres)
        if detections is not None and detections[0] is not None:
            #x1, y1, x2, y2, conf, confidence, class pred
            detections = rescale_boxes(detections[0], img.shape[0], self.orgSize)
        return detections

class objectVisualizer(object):
    def __init__(self, classPath, color=(0,0,255), thickness=1, textColor=(255,0,0), textThickness=1):
        self.color = color
        self.thickness = thickness
        self.classes = load_classes(classPath)
        self.textColor = textColor
        self.textThickness = textThickness

    def drawBoxes(self, img, detections):
        imgCpy = img.copy()
        if detections is not None:
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                x1y1 = (int(x1),int(y1))
                x2y2 = (int(x2),int(y2))
                imgCpy = cv2.rectangle(imgCpy,x1y1,x2y2,
                                        self.color, self.thickness)
                imgCpy = cv2.putText(imgCpy, f"{self.classes[int(cls_pred)]} {cls_conf}",
                                     x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL,1, self.textColor, self.textThickness)
        return imgCpy