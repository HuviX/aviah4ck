import os
import warnings
from typing import List

import cv2
import numpy as np
import torch
import torchvision
from albumentations.pytorch.transforms import ToTensorV2
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

warnings.filterwarnings('ignore')


# canny filter crop to remove background
def crop_canny(img):
    blurred = cv2.blur(img, (3, 3))
    canny = cv2.Canny(blurred, 50, 200)
    pts = np.argwhere(canny > 0)
    y1, x1 = pts.min(axis=0)
    y2, x2 = pts.max(axis=0)
    cropped = img[y1:y2, x1:x2]
    return cropped, (y1, y2), (x1, x2)


def get_object_detection_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def apply_nms(orig_prediction, iou_thresh=0.3):
    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(
        orig_prediction['boxes'], orig_prediction['scores'], iou_thresh
    )

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


def random_window_prediction(
    model, path: str, n_crops: int = 100, k: int = 1, canny_crop: bool = False
) -> np.ndarray:
    src_img = cv2.imread(path)
    if canny_crop:
        src_img = crop_canny(src_img)

    # define a 100 of crops
    crops = {}
    for i in range(n_crops):
        size = np.random.randint(440, 512)
        y1 = np.random.randint(w // 3, h - size)
        x1 = np.random.randint(h // 3, w - size)
        x2 = x1 + size
        y2 = y1 + size
        d = {'image': src_img[y1:y2, x1:x2], 'coords': (y1, y2, x1, x2)}
        crops[i] = d

    # Model outputs a dictionary with keys 'scores', 'boxes', 'labels'
    # scores_i corresponds to confidence of boxes_i.
    # Ex: {scores: [0.1, 0.43, 0.2, ...],
    #      bbox: [(x1, x2, w, h), ...],
    #      labels: [1, 2, 1, 1, ...]
    #    }

    # Get the maximum confidence for bbox across image(crop).
    # Store these values across all n_crops in order to
    # choose the most confident one. (or three as in this function)
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
        boxes.append(box)
        scores.append(max_score)
    # pick 3 most confident boxes for all n_crops
    top = torch.argsort(torch.tensor(scores), descending=True)[:k]

    for ind in top:
        ind = ind.item()
        box = boxes[ind]
        crop = crops[ind]['image']
        xmin = box[0]
        ymin = box[1]
        xmax = box[2] + xmin
        ymax = box[3] + ymin
        cv2.rectangle(crop, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    return src_img


def sliding_window_prediction(
    model, path: str, windows: List[int] = [224], canny_crop: bool = False, k: int = 1
):
    src_img = cv2.imread(path)
    im = src_img
    if canny_crop:
        im, cy, cx = crop_canny(src_img)

    imgheight = im.shape[0]
    imgwidth = im.shape[1]

    for window in windows:
        crops = dict()
        M = imgheight // (imgheight // window)
        N = imgwidth // (imgwidth // window)
        i = 0
        for y in range(0, imgheight, M):
            for x in range(0, imgwidth, N):
                y1 = y + M
                x1 = x + N
                tiles = im[y : y + M, x : x + N]
                d = {'image': im[y : y + M, x : x + N], 'coords': (y, y + M, x, x + N)}
                crops[i] = d
                i += 1

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
            boxes.append(box)
            scores.append(max_score)

        top = torch.argsort(torch.tensor(scores), descending=True)[:k]
        for ind in top:
            ind = ind.item()
            box = boxes[ind]
            crop = crops[ind]['image']
            xmin = box[0]
            ymin = box[1]
            xmax = box[2] + xmin
            ymax = box[3] + ymin
            cv2.rectangle(crop, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        src_img[cy[0] : cy[1], cx[0] : cx[1]] = im
    return src_img


def main(**kwargs):
    """
    ---- Usage ----
    from inference import main

    kwargs = {
        'path': '/path/to/img.png',
        'n_crops': 228,
        'path_out': 'res2.png',
        'top_k': 3,
        'type': 'window', # or 'random'
        'canny_crop': True,
        'model_path': 'state.pth'
    }

    main(**kwargs)
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    path = kwargs['path']
    n_crops = kwargs['n_crops']
    path_out = kwargs['path_out']

    # draw top k boxes
    top_k = kwargs['top_k']
    canny_crop = kwargs['canny_crop']
    model_path = kwargs['model_path']
    prediction_type = kwargs['type']

    # How to: model initialization
    model = get_object_detection_model(3)
    model.load_state_dict(torch.load(model_path)['model'])
    model.eval()
    model.cuda()

    if prediction_type == 'window':
        res = sliding_window_prediction(model, path, canny_crop=True, k=3)
    else:
        res = random_window_prediction(model, path, n_crops, top_k)
    cv2.imwrite(path_out, res)
    print('Done')


if __name__ == '__main__':
    kwargs = {
        'path': 'data/1/test/4.png',
        'n_crops': 228,
        'path_out': 'res.png',
        'top_k': 3,
        'type': 'window',  # or 'random'
        'canny_crop': True,
        'model_path': 'app/model/data/state9.pth'
    }

    main(**kwargs)
