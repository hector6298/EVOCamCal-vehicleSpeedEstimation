import cv2
import numpy as np
import umap
import math

from scipy.spatial import distance
from joblib import dump, load
from collections import OrderedDict
from haversine import haversine, Unit

from .projectiveGeometry import  parseHomography, homography2Dto3D
from .imgFeatureUtils import gen_cust_dist_func, bb_intersection_over_union


# TODO investigate SORT tracker, returns [x1,x2,y1,y2, id]. Create a velocity tracker, then modify yolo.draw_bbox to accept

class Tracker:
    def __init__(self, 
                 height, 
                 width, 
                 metric='euclidean',
                 reducer=False,
                 maxlost = 30, 
                 speed_update_rate = 25, 
                 emaAlpha = 0.4,
                 video=1, 
                 params_file=None):
        
        self.metric = metric
        self.speed_update_rate = speed_update_rate
        self.nextObjId = 0
        self.objs = OrderedDict()
        self.objsBB = OrderedDict()
        self.objsNewestRealLoc = OrderedDict()
        self.objsPrevRealLoc = OrderedDict()
        self.frameCount = OrderedDict()
        self.lost = OrderedDict()
        self.maxLost = maxlost
        self.height = height
        self.width = width
        self.projectionMat = parseHomography(params_file)
        self.emaAlpha = emaAlpha
        self.objsRealVel = OrderedDict()
        self.objsPrevRealVel = OrderedDict()
        self.objsEMARealVel = OrderedDict()
        self.objsVelDataCount = OrderedDict()
        self.objsVelHist = OrderedDict()
        self.frameCount = OrderedDict()
        self.bboxHist = OrderedDict()

        self.cust_iou = gen_cust_dist_func(bb_intersection_over_union, parallel=True)

        if reducer:
            self.reducer = load(f"models/reducer{video}.joblib")
        else:
            self.reducer = None

        self.valid_metrics = ['iou', 'euclidean']
        
    def emaVelocity(self, objectID:int):
        if self.objsVelDataCount[objectID] == 1:
            vel = self.objsRealVel[objectID]
        else:
            vel = self.emaAlpha*self.objsRealVel[objectID] + (1 - self.emaAlpha)*self.objsEMARealVel[objectID]
        return vel

    def addObject(self, new_object_location, bbox):
        self.objs[self.nextObjId] = new_object_location
        self.objsBB[self.nextObjId] = bbox
        self.objsNewestRealLoc[self.nextObjId] = None
        self.objsPrevRealLoc[self.nextObjId] = homography2Dto3D(new_object_location, self.projectionMat)
        self.objsRealVel[self.nextObjId] = 0.0
        self.objsPrevRealVel[self.nextObjId] = 0.0
        self.lost[self.nextObjId] = 0
        self.objsEMARealVel[self.nextObjId] = 0.0
        self.objsVelDataCount[self.nextObjId] = 0
        self.frameCount[self.nextObjId] = self.speed_update_rate 
        self.objsVelHist[self.nextObjId] = []
        self.bboxHist[self.nextObjId] = []
        self.nextObjId += 1

    def removeObject(self, objectID):
        del self.objs[objectID]
        del self.objsBB[objectID]
        del self.objsNewestRealLoc[objectID]
        del self.lost[objectID]
        del self.objsRealVel[objectID]
        del self.objsPrevRealLoc[objectID]
        del self.objsPrevRealVel[objectID]
        del self.objsEMARealVel[objectID]
        del self.objsVelDataCount[objectID]
        del self.frameCount[objectID]
        del self.objsVelHist[objectID]
        del self.bboxHist[objectID]

    def getLocation(self,bbox):
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) /2.0),int((y1 + y2 )/2.0))

    def write_velocities(self, image):
        fontScale = 0.5
        image_h, image_w, _ = image.shape
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        for objectID, bbox in  self.objsBB.items():
            c1, c2 = (bbox[0], bbox[1]), (bbox[2], bbox[3])
            #qif velocities[objectID] > 0:
            text = "{}: {:.2f}-km/h".format(objectID, self.objsEMARealVel[objectID])
            
            cv2.putText(image, text, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
            #cv2.circle(image, (centroid[0], centroid[1]), 4, (255,0,0), -1)
        return image


    def getObjsEmbeddings(self, frame, bboxes):
        histograms = []
        for bbox in bboxes:
            x1,y1,x2,y2 = int(bbox[1]), int(bbox[0]), int(bbox[3]), int(bbox[2])
            frame_piece = frame[y1:y2+1, x1:x2+1]
            det_hist_channels = []
            for channel in range(3):
                histr = np.squeeze(cv2.calcHist([frame_piece],[channel],None,[256],[0,256]))
                det_hist_channels.extend(histr)
            histograms.append(det_hist_channels)
        reduced_histograms = self.reducer.transform(histograms)
        return reduced_histograms


    def update(self, frame, bboxes, frameRate):
        if len(bboxes) == 0:
            lost_ids = list(self.lost.keys())
            for objectID in lost_ids:
                self.lost[objectID] += 1
                if self.lost[objectID] > self.maxLost: self.removeObject(objectID)
            return self.objs, self.objsRealVel
        
        new_object_locations = np.zeros((len(bboxes), 2), dtype="int")
        new_object_bbs = np.zeros((len(bboxes), 4), dtype='int')
        bbDists = []

        for i, bbox in enumerate(bboxes):
            bbox = [bbox[1], bbox[0], bbox[3], bbox[2]]
            bbDist = np.linalg.norm([bbox[1] - bbox[3], bbox[0] - bbox[2]])
            bbDists.append(bbDist)
            new_object_locations[i] = self.getLocation(bbox)
            new_object_bbs[i] = bbox

        if len(self.objs) == 0:
            for i in range(len(bboxes)):
                self.addObject(new_object_locations[i],  new_object_bbs[i])
        else:
            objectIDs = list(self.objs.keys())

            previous_object_locations = np.array(list(self.objs.values()))
            previous_object_bbs = np.array(list(self.objsBB.values()))

            row_idx, cols_idx = None, None

            if self.metric == 'euclidean':
                D = distance.cdist(previous_object_locations, new_object_locations)
                #print(D)

            elif self.metric == 'iou':
                D = self.cust_iou(previous_object_bbs, new_object_bbs)*(-1)+1
                #print(D)
            
            if self.reducer is not None:
                newestEmbeddings = self.getObjsEmbeddings(frame, new_object_bbs)
                previousEmbeddings = self.getObjsEmbeddings(frame, previous_object_bbs)
                Dembed = distance.cdist(previousEmbeddings, newestEmbeddings, metric='cosine')

                D = D*(1/2) + Dembed*(1/2)

            if self.reducer is None and self.metric not in self.valid_metrics:
                raise Exception("You must select a valid metric to assign objects. Abort...")

            row_idx = D.argmin(axis=1).argsort()
            cols_idx = D.argmin(axis=1)[row_idx]

            assignedRows, assignedCols = set(), set()
            for row, col in zip(row_idx, cols_idx):
                if row in assignedRows or col in assignedCols:
                    continue
                objectID = objectIDs[row]
                self.objs[objectID] = new_object_locations[col]
                self.objsBB[objectID] = new_object_bbs[col]
                self.objsNewestRealLoc[objectID] = homography2Dto3D(self.objs[objectID], self.projectionMat)

                self.frameCount[objectID] -= 1
                if(self.frameCount[objectID] == 0):

                    componentDist = haversine(self.objsNewestRealLoc[objectID], self.objsPrevRealLoc[objectID])
                    self.objsRealVel[objectID] = (componentDist/((self.lost[objectID]+self.speed_update_rate)/frameRate))*3600
                    self.objsPrevRealLoc[objectID] = self.objsNewestRealLoc[objectID]
                    self.objsVelDataCount[objectID] += 1
                    self.objsEMARealVel[objectID] = self.emaVelocity(objectID)
                    self.objsPrevRealVel[objectID] = self.objsEMARealVel[objectID]
                    self.frameCount[objectID] = self.speed_update_rate
                
                self.objsVelHist[objectID].append(self.objsEMARealVel[objectID])
                self.bboxHist[objectID].append(2)
                self.lost[objectID] = 0

                assignedRows.add(row)
                assignedCols.add(col)

            unassignedRows = set(range(0, D.shape[0])).difference(assignedRows)
            unassignedCols = set(range(0, D.shape[1])).difference(assignedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unassignedRows:
                    objectID = objectIDs[row]
                    self.lost[objectID] += 1
                    if self.lost[objectID] > self.maxLost:
                        self.removeObject(objectID)
            else:
                for col in unassignedCols:
                    self.addObject(new_object_locations[col], bboxes[col])
        return self.objs, self.objsEMARealVel

