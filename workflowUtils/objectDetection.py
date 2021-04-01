import os
import datetime
import torch
import tensorflow as tf
import cv2
import numpy as np

from PIL import Image
from torch.autograd import Variable

from .yolov3.models import *
from .yolov3.utils import *
import torchvision.transforms as T

from .yolov4.core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import cv2
import random
import colorsys
import numpy as np
import tensorflow as tf

from .yolov4.core.config import cfg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def rescale_boxes(boxes, current_dim, original_shape):
    """ Rescales bounding boxes to the original shape """
    orig_h, orig_w = original_shape

    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = (boxes[:, 0] ) * orig_w
    boxes[:, 1] = (boxes[:, 1] ) * orig_w
    boxes[:, 2] = (boxes[:, 2] ) * orig_h
    boxes[:, 3] = (boxes[:, 3] ) * orig_h
    return boxes

class objectDetectorV4(object):
    def __init__(self, imgSize, orgSize, weightPath, confidenceThres=0.6, IoUThres=0.7):
        self.imgSize = imgSize
        self.confidenceThres = confidenceThres
        self.IoUThres = IoUThres
        self.orgSize = orgSize
        self.classFile = cfg.YOLO.CLASSES
        
        self.model = tf.saved_model.load(weightPath, tags=[tag_constants.SERVING])
        self.infer = self.model.signatures['serving_default']

        self.classes = self._read_class_names(self.classFile)
        self.num_classes = len(self.classes)

    def _read_class_names(self,class_file_name):
        names = {}
        with open(class_file_name, 'r') as data:
            for ID, name in enumerate(data):
                names[ID] = name.strip('\n')
        return names

    def draw_bbox(self, image, bboxes, show_label=True):
        
        image_h, image_w, _ = image.shape
        hsv_tuples = [(1.0 * x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

        random.seed(0)
        random.shuffle(colors)
        random.seed(None)

        centroids = []
        out_boxes, out_scores, out_classes, num_boxes = bboxes
        for i in range(num_boxes[0]):
            if int(out_classes[0][i]) < 0 or int(out_classes[0][i]) > self.num_classes: continue
            coor = out_boxes[0][i]

            fontScale = 0.5
            score = out_scores[0][i]
            class_ind = int(out_classes[0][i])
            bbox_color = colors[class_ind]
            bbox_thick = int(0.6 * (image_h + image_w) / 600)
            c1, c2 = (coor[1], coor[0]), (coor[3], coor[2])
            cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

            if show_label:
                bbox_mess = '%s: %.2f-km/h' % (self.classes[class_ind], score)
                t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
                c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
                cv2.rectangle(image, c1, (np.float32(c3[0]), np.float32(c3[1])), bbox_color, -1) #filled

               # cv2.putText(image, bbox_mess, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                            #fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
        return image

    def scale_detections(self, bboxes):
        out_boxes, out_scores, out_classes, num_boxes = bboxes
        image_h, image_w = self.orgSize
        for i in range(num_boxes[0]):
            if int(out_classes[0][i]) < 0 or int(out_classes[0][i]) > self.num_classes: continue
            out_boxes[0][i][0] = int(out_boxes[0][i][0] * image_h)
            out_boxes[0][i][2] = int(out_boxes[0][i][2] * image_h)
            out_boxes[0][i][1] = int(out_boxes[0][i][1] * image_w)
            out_boxes[0][i][3] = int(out_boxes[0][i][3] * image_w)
        bboxes = out_boxes, out_scores, out_classes, num_boxes
        return bboxes

    def detect(self, img):
        img = img / 255.
        img = img[np.newaxis, ...].astype(np.float32)
        batch_data = tf.constant(img)
        pred_bbox = self.infer(batch_data)

        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=self.IoUThres,
            score_threshold=self.confidenceThres
        )

        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        pred_bbox = self.scale_detections(pred_bbox)
       
        return pred_bbox


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
