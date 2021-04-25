import torch
import torchvision
import os
import os
import random
import numpy as np
import pandas as pd
import yaml
import warnings
warnings.filterwarnings('ignore')
import addict
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

os.environ['CUDA_VISIBLE_DEVICES'] = '5'


def get_object_detection_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    return model


def apply_nms(orig_prediction, iou_thresh=0.3):
    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
    
    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]
    
    return final_prediction

def torch_to_pil(img):
    return transforms.ToPILImage()(img).convert('RGB')



def test_time_transform(image, w=224, h=224):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    img_res = cv2.resize(img_rgb, (w, h), cv2.INTER_AREA)
    img_res /= 255.0
    return ToTensorV2()(image=img_res)['image']
    

def get_boxes_for_large_image(path: str, n_crops: int = 100) -> np.ndarray:
    src_img = cv2.imread(path)
    ## define a 100 of crops
    crops = {}
    for i in range(n_crops):
        size = np.random.randint(440, 512)
        y1 = np.random.randint(w//3, h - size)
        x1 = np.random.randint(h//3, w - size)
        x2 = x1 + size
        y2 = y1 + size
        d = {'image': src_img[y1: y2, x1: x2],
             'coords': (y1, y2, x1, x2)}
        crops[i] = d
    boxes = []
    scores = []
    for i, crop in crops.items():
        img = crop['image']
        img = test_time_transform(img).cuda()
        with torch.no_grad():
            prediction = model([img])[0]
        idx = torch.argmax(prediction['scores'].cpu())
        max_score = prediction['scores'][idx]
        box = prediction['boxes'][idx]
        # print(prediction['labels'][idx])
        boxes.append(box)
        scores.append(max_score)
    top3 = torch.argsort(torch.tensor(scores), descending=True)[: 3]
    for ind in top3:
        ind = ind.item()
        box = boxes[ind]
        crop = crops[ind]['image']
        xmin = box[0]
        ymin = box[1]
        xmax = box[2] + xmin
        ymax = box[3] + ymin
        cv2.rectangle(crop, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    return src_img




##entry-point
def main(**kwargs):
    path = kwargs['path']
    n_crops = kwargs['n_crops']
    path_out = kwargs['path_out']
    # How to: model initialization
    # model = get_object_detection_model(3)
    # model.load_state_dict(torch.load(path)['model'])
    # model.eval()
    # model.cuda()

    # to use this you need to create a model first
    res = get_boxes_for_large_image(path, n_crops)
    # res = get_boxes_for_large_image('../../hacc/add/scratch/IMG_3679.JPG', 500)
    cv2.imwrite(path_out, res)