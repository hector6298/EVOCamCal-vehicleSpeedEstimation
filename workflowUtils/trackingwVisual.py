import cv2
import numpy as np
import cupy as cp
import umap
import math

from scipy.spatial import distance
from joblib import dump, load
from collections import OrderedDict
from haversine import haversine, Unit
from cuml.metrics.pairwise_distances import pairwise_distances

from .projectiveGeometry import  parseHomography, homography2Dto3D
from .imgFeatureUtils import gen_cust_dist_func, bb_intersection_over_union


class Tracker:
    def __init__(self, 
                 height, 
                 width, 
                 distance_sigma,
                 metric='euclidean',
                 reducer=False,
                 maxlost = 30, 
                 speed_update_rate = 25, 
                 emaAlpha = 0.4,
                 video=1, 
                 params_file=None):

        self.distance_sigma = distance_sigma
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
        self.objsEmbeddings = OrderedDict()
        self.dist_threshold = OrderedDict()

        self.cust_iou = gen_cust_dist_func(bb_intersection_over_union, parallel=True)

        if reducer:
            self.reducer = load(f"modelsGPU/reducer{video}.joblib")
        else:
            self.reducer = None

        self.valid_metrics = ['iou', 'euclidean']
        
    def emaVelocity(self, objectID:int):
        if self.objsVelDataCount[objectID] == 1:
            vel = self.objsRealVel[objectID]
        else:
            vel = self.emaAlpha*self.objsRealVel[objectID] + (1 - self.emaAlpha)*self.objsEMARealVel[objectID]
        return vel

    def addObject(self, frame, new_object_location, bbox):
        self.objs[self.nextObjId] = [new_object_location]
        self.objsBB[self.nextObjId] = [bbox]
        self.objsEmbeddings[self.nextObjId] = [self.getObjEmbeddings(frame, bbox)]
        self.dist_threshold[self.nextObjId] = self.distance_sigma

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
        del self.objsEmbeddings[objectID]
        del self.dist_threshold[objectID]
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

    def getObjEmbeddings(self, frame, bbox):
        det_hist_channels = []
        x1,y1,x2,y2 = int(bbox[1]), int(bbox[0]), int(bbox[3]), int(bbox[2])
        frame_piece = frame[y1:y2+1, x1:x2+1]

        for channel in range(3):
            histr = np.squeeze(cv2.calcHist([frame_piece],[channel],None,[256],[0,256]))
            det_hist_channels.extend(histr)

        det_hist_channels = np.array(det_hist_channels)
        embeddings = self.reducer(det_hist_channels)
        return embeddings

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
            
        return image

    def _validate_distance(self, objectId, previous, current, prev_embds, curr_embds, metric='euclidean'):
        result_flag_dist = False
        result_flag_embds = False

        if metric == 'iou':
            if (1 - bb_intersection_over_union(previous, current)) <= self.dist_threshold[objectId]:
                result_flag_dist = True
        else:
            if pairwise_distances(previous, current, metric=metric) <= self.dist_threshold[objectId]:
                result_flag_dist = True
        
        if pairwise_distances(previous, current, metric='cosine') <= self.dist_threshold[objectId]:
            result_flag_embds = True
        
        return result_flag_dist * result_flag_embds

    def update(self, frame, bboxes, frameRate):
        len_bboxes = len(bboxes)

        # If there is no detections, then mark active objects tracked as lost one more frame
        if len_bboxes == 0:
            lost_ids = list(self.lost.keys())
            for objectID in lost_ids:
                self.lost[objectID] += 1
                if self.lost[objectID] > self.maxLost: self.removeObject(objectID)
            return self.objs, self.objsRealVel
        
        # Just convert bboxes to propper form, get the centroids and get the embeddings
        new_object_locations = np.zeros((len_bboxes, 2), dtype="int")
        new_object_bbs = np.zeros((len_bboxes, 4), dtype='int')
        new_object_embeddings = np.zeros((len_bboxes, 5), dtype='float32')
        bbDists = []

        for i, bbox in enumerate(bboxes):
            bbox = [bbox[1], bbox[0], bbox[3], bbox[2]]
            bbDist = np.linalg.norm([bbox[1] - bbox[3], bbox[0] - bbox[2]])
            bbDists.append(bbDist)
            new_object_locations[i] = self.getLocation(bbox)
            new_object_bbs[i] = bbox
            new_object_embeddings[i] = self.getObjEmbeddings(frame, bbox)


        # If there is not a single object being tracked
        if len(self.objs) == 0:
            for i in range(len_bboxes):
                self.addObject(frame, new_object_locations[i],  new_object_bbs[i])
        else:
            objectIDs = list(self.objs.keys())

            # Get the last location and bounding box for each object
            previous_object_locations = np.array(list(self.objs.values()))[:, -1]
            previous_object_bbs = np.array(list(self.objsBB.values()))[:, -1]
            previous_object_embeddings = np.array(list(self.objsEmbeddings.values()))[:,-1]
            row_idx, cols_idx = None, None

            if self.metric == 'euclidean':
                D = pairwise_distances(previous_object_locations, new_object_locations, metric='euclidean',  output_type='numpy')
                #print(D)

            elif self.metric == 'iou':
                D = self.cust_iou(previous_object_bbs, new_object_bbs)*(-1)+1
                #print(D)
            
           
            Dembed = pairwise_distances(previous_object_embeddings, new_object_embeddings, metric='cosine', output_type='numpy')

            D = D*(1/2) + Dembed*(1/2)

            if self.metric not in self.valid_metrics:
                raise Exception("You must select a valid metric to assign objects. Abort...")

            row_idx = D.argmin(axis=1).argsort()
            cols_idx = D.argmin(axis=1)[row_idx]

            assignedRows, assignedCols = set(), set()
            for row, col in zip(row_idx, cols_idx):
                if row in assignedRows or col in assignedCols:
                    continue
                objectID = objectIDs[row]
                if self._validate_distance(objectID, self.objs[objectID][-1], new_object_locations[col], 
                                           self.objsEmbeddings[objectID][-1], new_object_embeddings[col], 
                                           metric=self.metric):
                
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
                    self.addObject(frame, new_object_locations[col], bboxes[col])
        return self.objs, self.objsEMARealVel

