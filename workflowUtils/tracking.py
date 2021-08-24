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
from .imgFeatureUtils import gen_cust_dist_func, compute_iou, iou


class Tracker:

    def isInCaptureLine(self, location):
        # if y_car == mx_car+b ... if car is in capture line
        #print(location[1] - self.m*location[0] - self.b )
        if abs(location[1] - self.m*location[0] - self.b) < 2.0:
            return True
        else:
            return False
    
    def getLocation(self,bbox):
        x1, y1, x2, y2 = bbox
        return (float((x1 + x2) //2.0),float((y1 + y2 )//2.0))

    def emaVelocity(self):
        raise NotImplementedError("You must override this abstract method")

    def write_velocities(self, image): 
        raise NotImplementedError("You must override this abstract method")

    def getVelocityHistory(self):
        raise NotImplementedError("You must override this abstract method")
    
    def getBBDists(self):
        raise NotImplementedError("You must override this abstract method")
    
    def getVelocitiesInLine(self):
        raise NotImplementedError("You must override this abstract method")
    
    def update(self, detections, frame_num):
        raise NotImplementedError("You must override this abstract method")



class Tracker(Tracker):

    def __init__(self, xs, ys, sigma_h, sigma_iou, t_min, params_file, frameRate, metric='iou', emaAlpha=0.3, update_rate=25):
        
        super().__init__() 
        self.tracks_active = []
        self.tracks_finished = []
        self.sigma_iou = sigma_iou
        self.sigma_h = sigma_h
        self.t_min = t_min
        self.update_rate = update_rate
        self.projectionMat = parseHomography(params_file)
        self.frameRate = frameRate
        self.emaAlpha = emaAlpha
        self.velocitiesInLine = []
        self.xs = xs
        self.ys = ys
        self.m, self.b = np.polyfit(xs,ys,1)
        self.metric = metric

    def getVelocityHistory(self):
        speeds = []
        for track in self.tracks_finished:
            speeds.append(track['EMAspeeds'])
        return speeds
    
    def getBBDists(self):
        dists = []
        for track in self.tracks_finished:
            dists.append(track['bbDists'])
        return dists

    def getVelocitiesInLine(self):
        return self.velocitiesInLine

    def emaVelocity(self, track):
        if len(track['speeds']) == 1:
            vel = track['speeds'][0]
        else:
            vel = self.emaAlpha*track['speeds'][-1] + (1 - self.emaAlpha)*track['EMAspeeds'][-1]
        return vel
    
    def write_velocities(self, image):
        fontScale = 0.5
        image_h, image_w, _ = image.shape
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        for i, track in  enumerate(self.tracks_active):
            bbox = track['bboxes'][-1]
            
            if len(bbox) > 0 and len(track["EMAspeeds"]) > 0:
                c1, c2 = (bbox[0], bbox[1]), (bbox[2], bbox[3])
                text = "{}: {:.2f}-km/h".format(i, track["EMAspeeds"][-1])
                
                cv2.putText(image, text, (int(c1[0]), int(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
            
        return image

    def update(self, bboxes, frame_num):
        
        updated_tracks = []

        bboxes = list(map(lambda bbox: [bbox[1], bbox[0], bbox[3], bbox[2]], bboxes))
        for track in self.tracks_active:

            flag = False  

            if len(bboxes) > 0:
                
                if self.metric == "iou":
                    best_match = max(bboxes, key=lambda bbox: iou(track['bboxes'][-1], bbox)) #x[0] is the bounding box
                    if iou(track['bboxes'][-1], best_match) >= self.sigma_iou:
                        flag = True
                else:
                    best_match = min(bboxes, key=lambda bbox: np.linalg.norm(np.array(self.getLocation(track['bboxes'][-1])) - np.array(self.getLocation(bbox))))
                    flag = True

                if flag:
                    track['bboxes'].append(best_match)
                    track["bbDists"].append(np.linalg.norm([best_match[1] - best_match[3], best_match[0] - best_match[2]]))
                    location = self.getLocation(best_match)
                    track['newestRealLoc'] = homography2Dto3D(location, self.projectionMat)
                    track['frameCount'] += 1

                    if track['frameCount'] == self.update_rate:
                        componentDist = haversine(track['newestRealLoc'], track['prevRealLoc'])
                        track['speeds'].append((componentDist/((track['frameCount'])/self.frameRate))*3600)
                        track['prevRealLoc'] = track['newestRealLoc']
                        # revisar esta logica sea correcta de ema velocity
                        track['EMAspeeds'].append(self.emaVelocity(track))
                        track['frameCount'] = 0

                    if self.isInCaptureLine(location) and len(track['EMAspeeds']) > 0:
                        self.velocitiesInLine.append(track['EMAspeeds'][-1])

                    updated_tracks.append(track)

                    del bboxes[bboxes.index(best_match)]
                    #ind = np.where(bboxes == best_match)
                    #bboxes = bboxes[np.r_[ind]]

            #if track was not updated
            if len(updated_tracks) == 0 or track is not updated_tracks[-1]:
                #finish track when the conditions are met
                if len(track['bboxes']) >= self.t_min:
                    self.tracks_finished.append(track)

        # create new tracks
        new_tracks = [{'bboxes': [bbox], 
                       'start_frame': frame_num, 
                       'speeds': [], 
                       'EMAspeeds': [], 
                       'realLocs':[], 
                       'bbDists':[], 
                       'frameCount': 0, 
                       'prevRealLoc': homography2Dto3D(self.getLocation(bbox), self.projectionMat), 
                       'newestRealLoc': None
                       } for bbox in bboxes]
                       
        self.tracks_active = updated_tracks + new_tracks

"""

class euclideanTracker(Tracker):
    def __init__(self, 
                 xs,ys,
                 framerate,
                 distance_sigma,
                 metric='euclidean',
                 maxlost = 5, 
                 speed_update_rate = 25, 
                 emaAlpha = 0.4,
                 params_file=None):
        super().__init__()

        self.distance_sigma = distance_sigma
        self.metric = metric
        self.speed_update_rate = speed_update_rate
        self.nextObjId = 0
        self.objs = OrderedDict()
        self.objsBB = OrderedDict()
        self.objsNewestRealLoc = dict()
        self.objsPrevRealLoc = dict()
        self.frameCount = dict()
        self.lost = OrderedDict()
        self.maxLost = maxlost
        self.projectionMat = parseHomography(params_file)
        self.emaAlpha = emaAlpha
        self.objsRealVel = dict()
        self.objsPrevRealVel = dict()
        self.objsEMARealVel = dict()
        self.objsVelDataCount = dict()
        self.objsVelHist = dict()
        self.frameCount = dict()
        self.objsBBdist = dict()
        self.cust_iou = gen_cust_dist_func(compute_iou, parallel=True)
        self.valid_metrics = ['iou', 'euclidean']
        self.velocitiesInLine = []
        self.m, self.b = np.polyfit(xs,ys,1)
        self.framerate = framerate

    def emaVelocity(self, objectID:int):
        if self.objsVelDataCount[objectID] == 1:
            vel = self.objsRealVel[objectID]
        else:
            vel = self.emaAlpha*self.objsRealVel[objectID] + (1 - self.emaAlpha)*self.objsEMARealVel[objectID]
        return vel

    def addObject(self, new_object_location, bbox):
        # location and bounding box initialization
        self.objs[self.nextObjId] = [new_object_location]
        self.objsBB[self.nextObjId] = [bbox]
        self.objsBBdist[self.nextObjId] = []

        # Localization and real speed initialization
        self.objsNewestRealLoc[self.nextObjId] = None
        self.objsPrevRealLoc[self.nextObjId] = homography2Dto3D(new_object_location, self.projectionMat)
        self.objsRealVel[self.nextObjId] = 0.0
        self.objsPrevRealVel[self.nextObjId] = 0.0
        self.lost[self.nextObjId] = 0
        self.objsEMARealVel[self.nextObjId] = 0.0
        self.objsVelDataCount[self.nextObjId] = 0
        self.frameCount[self.nextObjId] = 0
        self.objsVelHist[self.nextObjId] = []

        # Pointer to a future object
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

    def write_velocities(self, image):
        fontScale = 0.5
        image_h, image_w, _ = image.shape
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        for objectID, bbox_track in  self.objsBB.items():
            bbox = bbox_track[-1]
            
            if len(bbox) > 0:
                c1, c2 = (bbox[0], bbox[1]), (bbox[2], bbox[3])
                #qif velocities[objectID] > 0:
                text = "{}: {:.2f}-km/h".format(objectID, self.objsEMARealVel[objectID])
                
                cv2.putText(image, text, (int(c1[0]), int(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
            
        return image


    def _validate_distance(self, objectId, previous, current, metric='euclidean'):
        result_flag_dist = False
        if metric == 'iou':
            D = (1 - compute_iou(previous, current))
            #print(D)
        else:
            D = pairwise_distances(np.array([previous]), 
                                   np.array([current]), 
                                   metric=metric, 
                                   output_type='numpy')
        #print(D)
        if  D <= self.distance_sigma:
            result_flag_dist = True

    
        return True 

    def getVelocityHistory(self):
        return self.objsVelHist
    
    def getBBDists(self):
        return self.objsBBdist
    
    def getVelocitiesInLine(self):
        return self.velocitiesInLine
    
    def update(self, bboxes, frame_num):

        len_bboxes = len(bboxes)

        # If there is no detections, then mark active objects tracked as lost one more frame
        if len_bboxes == 0:
            lost_ids = list(self.lost.keys())
            for objectID in lost_ids:
                self.lost[objectID] += 1
                if self.lost[objectID] > self.maxLost: self.removeObject(objectID)
            return self.objs, self.objsRealVel
        
        # Just convert bboxes to propper form, get the centroids and get the embeddings
        new_object_locations = []
        new_object_bbs = []
        bbDists = []

        for i, bbox in enumerate(bboxes):
            #print(bbox)
            bbox = [bbox[1], bbox[0], bbox[3], bbox[2]]
            bbDist = np.linalg.norm([bbox[1] - bbox[3], bbox[0] - bbox[2]])
            bbDists.append(bbDist)
            new_object_locations.append(self.getLocation(bbox))
            new_object_bbs.append(bbox)

        # If there is not a single object being tracked
        if len(self.objs) == 0:
            for i in range(len_bboxes):
                self.addObject(new_object_locations[i],  new_object_bbs[i])
        else:
            objectIDs = list(self.objs.keys())

            # Get the last location and bounding box for each object
            previous_object_locations = [obj_locs[-1] for obj_locs in self.objs.values()]
            previous_object_bbs = [obj_bbs[-1] for obj_bbs in self.objsBB.values()]
            row_idx, cols_idx = None, None

            if self.metric == 'euclidean':
                D = distance.cdist(np.array(previous_object_locations), np.array(new_object_locations), metric='euclidean')
                #print(D)

            elif self.metric == 'iou':
                D = distance.cdist(np.array(previous_object_bbs),  np.array(new_object_bbs), lambda u, v: compute_iou(u, v))*(-1)+1
                #D = self.cust_iou(np.array(previous_object_bbs), np.array(new_object_bbs))*(-1)+1
                #print(D)

            if self.metric not in self.valid_metrics:
                raise Exception("You must select a valid metric to assign objects. Abort...")

            row_idx = D.argmin(axis=1).argsort()
            cols_idx = D.argmin(axis=1)[row_idx]
            print(row_idx, cols_idx)
            assignedRows, assignedCols = set(), set()
            for row, col in zip(row_idx, cols_idx):
                if row in assignedRows or col in assignedCols:
                    continue
                objectID = objectIDs[row]
                
                if self.metric == 'euclidean':
                    previous = self.objs[objectID][-1]
                    current = new_object_locations[col]
                elif self.metric == 'iou':
                    previous = self.objsBB[objectID][-1]
                    current = new_object_bbs[col]

                if self._validate_distance(objectID, previous, current, 
                                           metric=self.metric):
                
                    self.objs[objectID].append(new_object_locations[col])
                    self.objsBB[objectID].append(new_object_bbs[col])
                    self.objsNewestRealLoc[objectID] = homography2Dto3D(self.objs[objectID][-1], self.projectionMat)

                    if(self.frameCount[objectID] >= self.speed_update_rate):

                        componentDist = haversine(self.objsNewestRealLoc[objectID], self.objsPrevRealLoc[objectID])
                        self.objsRealVel[objectID] = (componentDist/((self.lost[objectID]+self.frameCount[objectID])/self.frameRate))*3600
                        self.objsPrevRealLoc[objectID] = self.objsNewestRealLoc[objectID]
                        self.objsVelDataCount[objectID] += 1
                        self.objsEMARealVel[objectID] = self.emaVelocity(objectID)
                        self.objsVelHist[objectID].append(self.objsEMARealVel[objectID])
                        self.objsPrevRealVel[objectID] = self.objsEMARealVel[objectID]
                        self.frameCount[objectID] = 0
                    
                    #append velocities that fall into the virtual capture line
                    if self.isInCaptureLine(new_object_locations[col]):
                        self.velocitiesInLine.append(self.objsEMARealVel[objectID])

                    self.lost[objectID] = 0

                    #capture the diagonal length of all bounding boxes
                    self.objsBBdist[objectID].append(bbDists[col])

                    # if no object has been assigned this frame
                    assignedRows.add(row)
                    assignedCols.add(col)

                self.frameCount[objectID] += 1

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

#{'bbox': (bb[0], bb[1], bb[2], bb[3]), 'score': s, 'class': c}

"""