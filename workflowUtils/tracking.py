import cv2
from scipy.spatial import distance
import numpy as np
from collections import OrderedDict
from haversine import haversine, Unit

from .projectiveGeometry import  parseHomography, homography2Dto3D
import math

# TODO investigate SORT tracker, returns [x1,x2,y1,y2, id]. Create a velocity tracker, then modify yolo.draw_bbox to accept

class Tracker:
    def __init__(self, height, width, maxlost = 30, speed_update_rate = 25, emaAlpha = 0.4, params_file=None):
        
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
        ##### exp moving average ##################
        self.emaAlpha = emaAlpha
        self.objsRealVel = OrderedDict()
        self.objsPrevRealVel = OrderedDict()
        self.objsEMARealVel = OrderedDict()
        self.objsVelDataCount = OrderedDict()
        self.frameCount = OrderedDict()

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

    def getLocation(self,bbox):
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) /2.0),int((y1 + y2 )/2.0))

    def write_velocities(self, image):
        fontScale = 0.5
        image_h, image_w, _ = image.shape
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        for objectID, bbox in  self.objsBB.items():
            c1, c2 = (bbox[1], bbox[0]), (bbox[3], bbox[2])
            #qif velocities[objectID] > 0:
            text = "{}: {:.2f}-km/h".format(objectID, self.objsEMARealVel[objectID])
            
            cv2.putText(image, text, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
            #cv2.circle(image, (centroid[0], centroid[1]), 4, (255,0,0), -1)
        return image

    def update(self, bboxes, frameRate):
        if len(bboxes) == 0:
            lost_ids = list(self.lost.keys())
            for objectID in lost_ids:
                self.lost[objectID] += 1
                if self.lost[objectID] > self.maxLost: self.removeObject(objectID)
            return self.objs, self.objsRealVel
        
        new_object_locations = np.zeros((len(bboxes), 2), dtype="int")
        for i, bbox in enumerate(bboxes):
            bbox = [bbox[1], bbox[0], bbox[3], bbox[2]]
            new_object_locations[i] = self.getLocation(bbox)
        
        if len(self.objs) == 0:
            for i in range(len(bboxes)):
                self.addObject(new_object_locations[i], bboxes[i])
        else:
            objectIDs = list(self.objs.keys())
            previous_object_locations = np.array(list(self.objs.values()))
            
            D = distance.cdist(previous_object_locations, new_object_locations)

            row_idx = D.argmin(axis=1).argsort()
            cols_idx = D.argmin(axis=1)[row_idx]

            assignedRows, assignedCols = set(), set()
            for row, col in zip(row_idx, cols_idx):
                if row in assignedRows or col in assignedCols:
                    continue
                objectID = objectIDs[row]
                self.objs[objectID] = new_object_locations[col]
                self.objsBB[objectID] = bboxes[col]
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

