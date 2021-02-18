import numpy as np
import cv2
import matplotlib.pyplot as plt

from skimage import feature,color,transform,io

def getTheta(line: tuple):
    p0, p1 = line
    delta = np.array(p1) - np.array(p0)
    theta = np.arctan(float(delta[1])/delta[0])
    return theta

def euclideanDist(line : tuple):
    p0, p1 = line
    return np.sqrt((p1[0]-p0[0])**2 + (p1[1]-p0[1])**2)

def getLinesEuclideanDist(lines: list):
    distances = []
    for line in lines:
        distances.append(euclideanDist(line))
    return distances

def getLinesThetas(lines: list):
    thetas = []
    for line in lines:
        thetas.append(getTheta(line))
    return thetas

def getEdgeMap(grayImg, min, max):
    grayImgCpy = grayImg.copy()
    return cv2.Canny(grayImgCpy,min,max).astype('uint8')

def getLineMax(lines:list):
    maxDist = 0.0
    maxInd = 0
    for i in range(len(lines)):
        dist = euclideanDist(lines[i])
        if dist > maxDist:
            maxDist = dist
            maxInd = i
    return lines[maxInd]

def getLinesFromEdges(edgeMap, angleBounds: tuple = None, line_length=20, line_gap=2):
    lines = transform.probabilistic_hough_line(edgeMap, line_length=line_length, line_gap=line_gap)
    filteredLines = []
    for p0, p1 in lines:
        delta = np.array(p1) - np.array(p0)
        theta = np.arctan(float(delta[1])/delta[0])

        if angleBounds is not None and theta > (np.pi/180)*angleBounds[0]\
           and theta < (np.pi/180)*angleBounds[1]:
            filteredLines.append((p0,p1))
        elif angleBounds is None:
            filteredLines.append((p0,p1))
    return filteredLines
    

def drawLinesOnImg(img, lines: list, color=(255,0,255), thickness=2):
    imgCpy = img.copy()
    if len(lines) > 0:
        for line in lines:
            p0, p1 = line[0], line[1]
            imgCpy = cv2.line(imgCpy, tuple(p0),tuple(p1),color,thickness)
    
    return imgCpy


def convertImg2Grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

