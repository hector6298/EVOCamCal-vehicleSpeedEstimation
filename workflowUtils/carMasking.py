import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T

from PIL import Image
from matplotlib import cm

from PIL import Image
from torchvision import models
from .imgFeatureUtils import *
from .objectDetection import objectDetector

#class macros
BUS = 6
CAR = 7

class TraditionalForegroundExtractor(object):
    def __init__(self,
                kernelSize=(5,5)):
        self.subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,kernelSize)

    def _getCarsMaskAtFrame(self,frame):
        assert len(frame.shape) == 2, f"frame must be on grayscale, found {len(frame.shape)} dimensions"
        foregroundMask = self.subtractor.apply(frame)
        foregroundMask[foregroundMask != 0] = 255
        self.foregroundMask = cv2.morphologyEx(foregroundMask, cv2.MORPH_OPEN, self.kernel)
        return foregroundMask

    def maskCarsAtFrame(self,frame):
        self._getCarsMaskAtFrame(frame)
        maskedFrame = cv2.bitwise_and(frame,frame,mask=self.foregroundMask)
        return maskedFrame


class DLForegroundExtractor(object):
    def __init__(self, imgSize):
        self.dlab = models.segmentation.deeplabv3_resnet50(pretrained=1).eval()
        self.imgSize = imgSize
        self.trf = T.Compose([
                        T.ToTensor(), 
                        T.Normalize(mean = [0.485, 0.456, 0.406], 
                                    std = [0.229, 0.224, 0.225])])
    def _getCarsMaskAtFrame(self,img):
        inp = self.trf(img).unsqueeze(0)
        out = self.dlab(inp)['out']
        om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
        om[om != 0] = 255
        return om

    def segment(self, img, gray):
        imgPIL = Image.fromarray(img)
        om = self._getCarsMaskAtFrame(imgPIL)
        maskedFrame = cv2.bitwise_and(gray,gray,mask=np.uint8(om))
        return maskedFrame
    

class objectFocusTools(object):
    def __init__(self,imgSize : tuple, detector : objectDetector):
        self.imgSize = imgSize
        self.detector = detector
        self.extractor = TraditionalForegroundExtractor()

    def _getMaskfromYOLODetection(self, frame):
        detections = self.detector.detect(frame)
        masks = np.zeros((len(detections),*self.imgSize), np.uint8)
        i = 0
        for x1, y1, x2, y2, _, _, _ in detections:
            masks[i,int(y1):int(y2)+1,int(x1):int(x2)+1] = 1
            i += 1
        return masks

    def focusOnObjects(self, frame):
        masks = self._getMaskfromYOLODetection(frame)
        instancesAtFrame = [cv2.bitwise_and(frame,frame,mask=masks[i]) for i in range(len(masks))]
        return instancesAtFrame
    
    def getObjectLinesAtFrameYOLO(self, frame, grayFrame,angleBounds):
        linesCont = []
        masks = self._getMaskfromYOLODetection(frame)
        edgeMap = getEdgeMap(grayFrame,100,200)
        if masks is not None:
            instancesAtFrame = [cv2.bitwise_and(edgeMap,edgeMap,mask=masks[i]) for i in range(len(masks))]
            for i in range(len(instancesAtFrame)):
                lines = getLinesFromEdges(instancesAtFrame[i],angleBounds)
                if len(lines) > 0:
                    line = getLineMax(lines)
                    linesCont.append(line)
        #print(linesCont)
        return linesCont, edgeMap


################################
# NON OBJECT HELPERS
################################

def getMaskfromYOLODetection(imgSize,detections):
    masks = np.zeros((len(detections),*imgSize), np.uint8)
    i = 0
    for x1, y1, x2, y2, _, _, _ in detections:
        masks[i,int(y1):int(y2)+1,int(x1):int(x2)+1] = 1
        i += 1
    return masks

flatten = lambda t: [item for sublist in t for item in sublist]

def getLinesAtFrame(gray,detections, angleBounds):
    imgSize = gray.shape
    linesCont = []
    masks = getMaskfromYOLODetection(imgSize,detections)
    edgeMap = getEdgeMap(gray,100,200).astype(np.uint8)
 
    if masks is not None:
        instancesAtFrame = [cv2.bitwise_and(edgeMap,edgeMap,mask=masks[i]) for i in range(len(masks))]
        for i in range(len(instancesAtFrame)):
            lines = getLinesFromEdges(instancesAtFrame[i],angleBounds)
            if len(lines) > 0:
                linesCont.append(lines)
    linesCont = flatten(linesCont)
    #print(linesCont)

    return linesCont
