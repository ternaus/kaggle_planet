"""
Experiments with pytorch
"""
from __future__ import division


import torch.optim as optim
from sklearn.preprocessing import MultiLabelBinarizer
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


from sklearn.metrics import fbeta_score
import numpy as np


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


class KaggleAmazonDataset(Dataset):
    """Dataset wrapping images and target labels for Kaggle - Planet Amazon from Space competition.

    Arguments:
        A CSV file path
        Path to image folder
        Extension of images
        PIL transforms
    """

    def __init__(self, tmp_df, img_path, img_ext, transform=None):
        assert tmp_df['image_name'].apply(lambda x: os.path.isfile(os.path.join(img_path, x + img_ext))).all(), \
            "Some images referenced in the CSV file were not found"

        self.img_path = img_path
        self.img_ext = img_ext
        self.transform = transform
        self.mlb = MultiLabelBinarizer()

        self.X_train = tmp_df['image_name'].values
        self.y_train = self.mlb.fit_transform(tmp_df['tags'].str.split()).astype(np.float32)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.X_train[index] + self.img_ext))
        img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        label = torch.from_numpy(self.y_train[index])
        return img, label

    def __len__(self):
        return self.X_train.shape[0]


if __name__ == '__main__':
    data_path = '../data'

    train_path = os.path.join(data_path, 'train-jpg')

    img_ext = '.jpg'
    train_labels_path = os.path.join(data_path, 'train_v2.csv')

    transformations = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_df = pd.read_csv(train_labels_path)

    dset_train = KaggleAmazonDataset(train_df, train_path, img_ext, transformations)

    train_loader = DataLoader(dset_train,
                              batch_size=4,
                              shuffle=True,
                              num_workers=1,  # 1 for CUDA
                              pin_memory=True  # CUDA only
                              )

    num_classes = 17
    net = resnet18(pretrained=True).cuda()
    net.fc = nn.Linear(net.fc.in_features, num_classes).cuda()

    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    criterion = MultiLabelSoftMarginLoss()

    def train(epoch):
        net.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(async=True), target.cuda(async=True)  # On GPU
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = net(data)

            # Tricky business: models have linear outputs and we need to deal with this in the loss

            loss = criterion(output, target)

            loss.backward()
            optimizer.step()
            if batch_idx % 250 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.data[0]))


    for epoch in range(0, 2):
        train(epoch)
