import os
import warnings

import addict
import albumentations as A
import pandas as pd
import torch
import torchvision
import yaml
from albumentations.pytorch.transforms import ToTensorV2
from engine import evaluate, train_one_epoch
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from app.model import utils

warnings.filterwarnings('ignore')


def save_checkpoint(epoch, model, optimizer, checkpoint_path):
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    if checkpoint_path[-1] != '/':
        checkpoint_path += '/'
    torch.save(state, checkpoint_path + 'state' + str(epoch) + '.pth')


def get_object_detection_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def get_transform(train):
    if train:
        return A.Compose(
            [
                A.HorizontalFlip(0.5),
                # ToTensorV2 converts image to pytorch tensor without div by 255
                ToTensorV2(p=1.0),
            ],
            bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']},
        )
    else:
        return A.Compose(
            [ToTensorV2(p=1.0)],
            bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']},
        )


def main():
    # ------ Parse config ----------
    with open('config.yaml') as f:
        cfg = addict.Dict(yaml.safe_load(f))
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.device
    data = pd.read_csv(cfg.dataset_path)
    image_size = cfg.image_size
    pic_dir = cfg.pic_dir
    split_rate = cfg.split_rate
    num_classes = cfg.num_classes
    num_epochs = cfg.num_epochs
    checkpoint_path = cfg.checkpoint_path
    logdir = cfg.logdir
    device = torch.device('cuda:0')
    # -------- Train and validation datasets -------
    train_size = int(data.shape[0] * split_rate)
    validation_size = data.shape[0] - train_size
    data = data.sample(frac=1)
    train_col = ['train'] * train_size + ['val'] * validation_size
    data['train'] = train_col
    train_df = data[data['train'] == 'train'].reset_index(drop=True)
    train_dataset = utils.FruitImagesDataset(
        pic_dir, image_size, image_size, train_df, transforms=get_transform(True)
    )
    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        collate_fn=utils.collate_fn,
    )

    val_df = data[data['train'] == 'val'].reset_index(drop=True)
    val_df.reset_index(drop=True).to_csv('validation.csv', index=False)
    val_dataset = utils.FruitImagesDataset(
        pic_dir, image_size, image_size, val_df, transforms=get_transform(True)
    )

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=8, num_workers=4, collate_fn=utils.collate_fn
    )

    # ------ Model for object detection ------
    model = get_object_detection_model(num_classes)
    model.cuda()
    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    num_epochs = 10
    writer = SummaryWriter(logdir)

    for epoch in range(num_epochs):
        train_one_epoch(
            model,
            optimizer,
            data_loader,
            torch.device('cuda:0'),
            epoch,
            writer=writer,
            print_freq=20,
        )
        save_checkpoint(epoch, model, optimizer, checkpoint_path)
        lr_scheduler.step()
        evaluate(model, val_data_loader, device, writer)
    print('dats it')


if __name__ == '__main__':
    main()
