"""
Experiments with pytorch
"""


import torch.optim as optim
import pandas as pd
from torch import np  # Torch wrapper for Numpy

import os
from PIL import Image

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
from torch.autograd import Variable
from torch.nn import MultiLabelSoftMarginLoss

from torchvision.models import resnet18, resnet50
import torch.nn.functional as F
import utils
import tqdm

from sklearn.metrics import fbeta_score
import numpy as np
import shutil
import argparse
from torch.optim import Adam
import data_loader
import augmentations


def f2_score(y_true, y_pred):
    # fbeta_score throws a confusing error if inputs are not numpy arrays
    y_true, y_pred, = np.array(y_true), np.array(y_pred)
    # We need to use average='samples' here, any other average method will generate bogus results
    return fbeta_score(y_true, y_pred, beta=2, average='samples')


def validation(model, criterion, valid_loader):
    model.eval()
    losses = []
    f2_scores = []
    for inputs, targets in valid_loader:
        inputs = utils.variable(inputs, volatile=True)
        targets = utils.variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        losses.append(loss.data[0])
        f2_scores.append(f2_score(y_true=targets.data.cpu().numpy(), y_pred=F.sigmoid(outputs).data.cpu().numpy() > 0.2))
    valid_loss = np.mean(losses)  # type: float
    valid_f2 = np.mean(f2_scores)  # type: float
    print('Valid loss: {:.4f}, F2: {:.4f}'.format(valid_loss, valid_f2))
    return {'valid_loss': valid_loss, 'valid_f2': valid_f2}


def get_model(num_classes):
    model = resnet50(pretrained=True).cuda()
    model.fc = nn.Linear(model.fc.in_features, num_classes).cuda()
    return model


def add_args(parser):
    arg = parser.add_argument
    arg('root', help='checkpoint root')
    arg('--batch-size', type=int, default=4)
    arg('--n-epochs', type=int, default=100)
    arg('--lr', type=float, default=0.0001)
    arg('--workers', type=int, default=2)
    arg('--fold', type=int, default=0)
    arg('--n-folds', type=int, default=10)
    arg('--clean', action='store_true')
    arg('--epoch-size', type=int)
    arg('--device-ids', type=str, help='For example 0,1 to run on two GPUs')


if __name__ == '__main__':
    random_state = 2016

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--mode', choices=['train', 'valid', 'predict_valid', 'predict_test'], default='train')
    add_args(parser)
    args = parser.parse_args()

    batch_size = args.batch_size

    train_transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        augmentations.RandomVerticalFlip(0.5),
        augmentations.Random90Rotation(),
        # augmentations.Rotate(),
        # augmentations.GaussianBlur(),
        augmentations.Add(-10, 10, per_channel=True),
        augmentations.ContrastNormalization(0.9, 1.1, per_channel=True),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_loader, valid_loader = data_loader.get_loaders(batch_size,
                                                         train_transform=train_transform,
                                                         fold=0)

    num_classes = 17

    model = get_model(num_classes)

    if utils.cuda_is_available:
        if args.device_ids:
            device_ids = list(map(int, args.device_ids.split(',')))
        else:
            device_ids = None

        model = nn.DataParallel(model, device_ids=device_ids).cuda()

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = MultiLabelSoftMarginLoss()

    n_epochs = 2

    utils.train(
        init_optimizer=lambda lr: Adam(model.parameters(), lr=lr),
        args=args,
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        valid_loader=valid_loader,
        validation=validation,
        # save_predictions=save_predictions,
        patience=2,
    )
