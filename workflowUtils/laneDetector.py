import cv2
import numpy as np
from .carMasking import TraditionalForegroundExtractor
from .imgFeatureUtils import *

def getRoadMap(cap,
                frameSize:tuple, 
               iterations=2000,
               returnFrames=False,
               returnBackground=True):
    frames = []
    extractor = TraditionalForegroundExtractor()
    road_map = np.zeros(frameSize,dtype=np.int32)
    for i in range(iterations): 
        ret, frame = cap.read() 
        frame = cv2.resize(frame, frameSize, fx = 0, fy = 0, 
                            interpolation = cv2.INTER_CUBIC)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        mask = extractor._getCarsMaskAtFrame(gray)

        road_map += mask
        road_map[road_map > 1] = 1
        frames.append(gray)

    #road_map[road_map >= 255] = 255
    background = np.median(frames, axis=0).astype(np.uint8)
    road_map = road_map.astype('uint8')
    if returnBackground:
        return background, road_map
    if returnFrames and returnBackground:
        return frames, background, road_map
    return road_map

def removeAdjacent(frame):
    frame_cpy = frame.copy()
    for row in range(len(frame)):
        for col in range(len(frame[0])):
            val = 0
            for i in range(col+1, col+5):
                if i < len(frame[0]):
                    val += frame[row, i]
            if val >= 1:
                frame_cpy[row, col] = 0
    return frame_cpy

def getRoad(cap, frameSize, iterations=2000, returnBackground=True):
    backgroundImg, roadMap = getRoadMap(cap, frameSize, 
                    iterations, returnBackground=True)
    road = cv2.bitwise_and(backgroundImg,backgroundImg, mask=roadMap)
    if returnBackground:
        return backgroundImg, road
    return road

def getRoadEdges(cap,frameSize, iterations=2000, adjacent=False):
    backgroundImg, roadMap = getRoadMap(cap, frameSize, 
                    iterations, returnBackground=True)
    edgeImg = getEdgeMap(backgroundImg, 100, 200)
    road = cv2.bitwise_and(edgeImg,edgeImg, mask=roadMap)
    if adjacent:
        road = removeAdjacent(road)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
        road = cv2.dilate(road,kernel,iterations = 1)
    return road

#search marcos nieto filtering https://github.com/carlesu/Lane-Detection-Hough-Transform/blob/master/MATLAB/nieto_filtering.m
def nietoFiltering(img, tau):
    size = img.shape
    y = np.zeros_like(img)
    for i in range(size[0]):
        col = img[i]
        y_col = y[i]
        for j in range(tau, size[1]-tau):
            if col[j] != 0:
                aux = 2*col[j]
                aux = aux - (col[j-tau] + col[j+tau])
                aux = aux - abs(col[j-tau] + col[j+tau])
                if aux < 0:
                    aux = 0
                if aux > 255:
                    aux = 255
                col[j] = aux
        y[i] = col
    return y

def otsuBinarize(img):
    blur = cv2.GaussianBlur(img, (5,5), 0)
    hist = cv2.calcHist([blur], [0], None, [256], [0,256])
    hist_norm = hist.ravel()/hist.max()
    Q = hist_norm.cumsum()

    bins = np.arange(256)

    fn_min = np.inf
    thresh = -1

    for i in range(1,256):
        p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
        q1,q2 = Q[i],Q[255]-Q[i] # cum sum of classes
        b1,b2 = np.hsplit(bins,[i]) # weights

        # finding means and variances
        m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
        v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2

        # calculates the minimization function
        fn = v1*q1 + v2*q2
        if fn < fn_min:
            fn_min = fn
            thresh = i
    # find otsu's threshold value with OpenCV function
    ret, otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return ret, otsu


##################################################
# CONNECTED COMPONENTS UTILITIES #################
##################################################


def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue==0] = 0
    cv2.imshow("connected components", labeled_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Nmaxelements(list1, N): 
  final_list = [] 
  for i in range(0, N):  
      max1 = 0
      for j in range(len(list1)):      
          if list1[j][1] > max1: 
              max1 = list1[j][1]
              max_ind = list1[j][0]
      list1.remove((max_ind,max1)); 
      final_list.append(max_ind) 
  return final_list
  
def get_n_greatest_labels(label_map, limit):
  label_dict = dict()
  for row in range(len(label_map)):
    for col in range(len(label_map[0])):
      if label_map[row,col] not in label_dict:
        label_dict[label_map[row,col]] = 0
      label_dict[label_map[row,col]] += 1

  max_list = Nmaxelements(list(label_dict.items()),limit)
  max_list.pop(0)
  print(max_list)
  return max_list
  
def preserve_n_greatest_image(label_map,max_list):
  label_map_cpy = label_map.copy()
  for row in range(len(label_map)):
    for col in range(len(label_map[0])):
      if label_map[row,col] not in max_list:
        label_map_cpy[row,col] = 0
      else:
        label_map_cpy[row,col] = 255
  return label_map_cpy

def delete_n_greatest_image(label_map, max_list):
  label_map_cpy = label_map.copy()
  for row in range(len(label_map)):
    for col in range(len(label_map[0])):
      if label_map[row,col] in max_list:
        label_map_cpy[row,col] = 0
  return label_map_cpy  

def get_n_preserve_n_labels(label_map,limit):
  max_list = get_n_greatest_labels(label_map,limit)
  return preserve_n_greatest_image(label_map,max_list)

def get_n_delete_n_labels(label_map, limit):
    max_list = get_n_greatest_labels(label_map,limit)
    return delete_n_greatest_image(label_map, max_list)