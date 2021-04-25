import pandas as pd
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from app.model.engine import evaluate, train_one_epoch
from app.model.utils import *


def save_checkpoint(epoch, model, optimizer, checkpoint_path):
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    if checkpoint_path[-1] != '/':
        checkpoint_path += '/'
    path = checkpoint_path + 'state' + str(epoch) + '.pth'
    torch.save(state, path)
    return path


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


def do_shell(checkpoint_dir, logdir):
    os.system(f'rm -r {logdir}')
    os.system(f'mkdir {logdir}')
    os.system(f'tensorboard --logdir={logdir} --bind_all')
    os.system(f'mkdir {checkpoint_dir}')


def return_two_loaders(path, batch_size):
    size = 224
    folders = ['train', 'test']
    train_df_params = {
        'files_dir': f'{path}/train/',
        'width': size,
        'height': size,
        'df': pd.read_csv(f'{path}/train/labels.csv'),
        'transforms': get_transform(True),
    }
    train_dataset = TrainImageDataset(**train_df_params)
    test_df_params = {
        'files_dir': f'{path}/test/',
        'width': size,
        'height': size,
        'df': pd.read_csv(f'{path}/test/labels.csv'),
        'transforms': get_transform(False),
    }
    test_dataset = TrainImageDataset(**test_df_params)

    print(f'Size of train: {len(train_dataset)}')
    print(f'Size of test: {len(test_dataset)}')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=4, collate_fn=collate_fn
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, num_workers=4, collate_fn=collate_fn
    )
    return train_loader, test_loader


class TrainImageDataset(torch.utils.data.Dataset):
    def __init__(self, files_dir, width, height, df, transforms=None):
        self.transforms = transforms
        self.files_dir = files_dir
        self.height = height
        self.width = width
        self.xmin = df.x.values
        self.ymin = df.y.values
        self.widths = df['width'].values
        self.heights = df['height'].values
        self.imgs = df.image.values
        self.df = df

        # classes: 0 index is reserved for background
        self.classes = ['back', 'defect']
        self.mapping = {'defect': 1}

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        image_path = os.path.join(self.files_dir, img_name)

        # reading the images and converting them to correct size and color
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_res = cv2.resize(img_rgb, (self.width, self.height), cv2.INTER_AREA)
        img_res /= 255.0

        boxes = []
        labels = []

        wt = img.shape[1]
        ht = img.shape[0]
        labels.append(1)
        xmin = self.xmin[idx]
        xmax = xmin + self.widths[idx]
        ymin = self.ymin[idx]
        ymax = ymin + self.heights[idx]
        xmin_corr = (xmin / wt) * self.width
        xmax_corr = (xmax / wt) * self.width
        ymin_corr = (ymin / ht) * self.height
        ymax_corr = (ymax / ht) * self.height

        boxes.append([xmin_corr, ymin_corr, xmax_corr, ymax_corr])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = torch.Tensor([self.widths[idx] * self.heights[idx]])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['area'] = area
        target['iscrowd'] = iscrowd
        image_id = torch.tensor([idx])
        target['image_id'] = image_id

        if self.transforms:
            sample = self.transforms(
                image=img_res, bboxes=target['boxes'], labels=labels
            )
            img_res = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])

        return img_res, target

    def __len__(self):
        return len(self.imgs)


def main(**kwargs):
    device = kwargs['device']
    path = kwargs['dataset_path']
    batch_size = kwargs['batch_size']
    pretrained = kwargs['pretrained']
    num_classes = kwargs['num_classes']
    checkpoint_path = kwargs['checkpoint_path']
    logdir = kwargs['logdir']

    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    train_loader, test_loader = return_two_loaders(path, batch_size)
    model = get_object_detection_model(num_classes)
    model.cuda()
    model.eval()
    device = torch.device('cuda:0')
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    num_epochs = 10
    writer = SummaryWriter(logdir)
    # do_shell(checkpoint_dir, logdir)

    for epoch in range(num_epochs):
        train_one_epoch(
            model, optimizer, train_loader, device, epoch, writer=writer, print_freq=20
        )
        # save_checkpoint(epoch, model, optimizer, checkpoint_path)
        lr_scheduler.step()
        evaluate(model, test_loader, device, writer)
    return save_checkpoint(epoch, model, optimizer, checkpoint_path)


# Usage
# from train_entry import main
# kwargs = {
#     'device': 5,
#     'dataset_path': 'data/',
#     'batch_size': 4,
#     'pretrained': True,
#     'num_classes': 3,
#     'checkpoint_path': 'train_entry_check',
#     'logdir': 'train_entry_log'
# }
# main(**kwargs)
