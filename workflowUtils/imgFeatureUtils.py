import numpy as np
import cv2
import matplotlib.pyplot as plt
import numba as nb
import random
import colorsys

from skimage import feature,color,transform,io


def iou(bbox1, bbox2):
    """
    Calculates the intersection-over-union of two bounding boxes.
    Args:
        bbox1 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
        bbox2 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
    Returns:
        int: intersection-over-onion of bbox1, bbox2
    """

    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, x1_1, y1_1) = bbox1
    (x0_2, y0_2, x1_2, y1_2) = bbox2

    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union
    
def gen_cust_dist_func(kernel,parallel=True):

    kernel_nb=nb.njit(kernel,fastmath=True)

    def cust_dot_T(A,B):
        assert B.shape[1]==A.shape[1]
        out=np.empty((A.shape[0],B.shape[0]),dtype=np.float64)
        for i in nb.prange(A.shape[0]):
            for j in range(B.shape[0]):
                out[i,j]=kernel_nb(A[i,:],B[j,:])
        return out

    if parallel==True:
        return nb.njit(cust_dot_T,fastmath=True,parallel=True)
    else:
        return nb.njit(cust_dot_T,fastmath=True,parallel=False)

def compute_iou(boxA, boxB):

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

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


def draw_bbox(image, bboxes, show_label=True):
    num_classes = 80
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    fontScale = 0.5
    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    out_boxes, out_scores, out_classes, num_boxes = bboxes
    for i in range(len(out_classes)):
        if out_classes[i] not in [2,5]: continue
        coor = out_boxes[i]

        score = out_scores[i]
        class_ind = int(out_classes[i])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[1], coor[0]), (coor[3], coor[2])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '0: 000.00-km/h' 
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
            c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
            cv2.rectangle(image, c1, (np.float32(c3[0]), np.float32(c3[1])), bbox_color, -1) #filled

            # cv2.putText(image, bbox_mess, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                        #fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
    return image