"""
Experiments with pytorch
"""
from __future__ import division


import torch.optim as optim
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
from torch import np  # Torch wrapper for Numpy
from sklearn.model_selection import StratifiedShuffleSplit

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


cuda_is_available = torch.cuda.is_available()


def cuda(x):
    return x.cuda() if cuda_is_available else x


def variable(x, volatile=False):
    if isinstance(x, (list, tuple)):
        return [variable(y, volatile=volatile) for y in x]
    return cuda(Variable(x, volatile=volatile))


def get_train_test_split(labels):
    labels['image_name'] = labels['image_name'] + '.jpg'

    labels['unified'] = np.nan
    labels.loc[labels['clear'] == 1, 'unified'] = 0
    labels.loc[labels['cloudy'] == 1, 'unified'] = 1
    labels.loc[labels['haze'] == 1, 'unified'] = 2
    labels.loc[labels['partly_cloudy'] == 1, 'unified'] = 3

    labels = labels[labels['unified'].notnull()]

    sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=random_state)

    train_index, val_index = sss.split(labels['unified'].values.astype(int),
                                       labels['unified'].values.astype(int)).next()

    return train_index, val_index


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


def train(args, model, criterion, train_loader, valid_loader, validation, init_optimizer, save_predictions=None, n_epochs=None, patience=2):
    lr = args.lr
    n_epochs = n_epochs or args.n_epochs
    optimizer = init_optimizer(lr)

    root = data_path
    model_path = os.path.join(root, 'model.pt')
    best_model_path = os.path.join(root, 'best-model.pt')

    if os.path.exists(model_path):
        state = torch.load(str(model_path))
        epoch = state['epoch']
        step = state['step']
        best_valid_loss = state['best_valid_loss']
        model.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
    else:
        epoch = 1
        step = 0
        best_valid_loss = float('inf')

    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
        'best_valid_loss': best_valid_loss
    }, str(model_path))

    report_each = 10
    save_prediction_each = report_each * 10
    valid_losses = []
    lr_reset_epoch = epoch
    for epoch in range(epoch, n_epochs + 1):
        model.train()
        tq = tqdm.tqdm(total=(args.epoch_size or len(train_loader) * args.batch_size))

        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        losses = []

        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = variable(inputs), variable(targets)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()

            batch_size = inputs.size(0)
            (batch_size * loss).backward()

            optimizer.step()
            step += 1
            tq.update(batch_size)
            losses.append(loss.data[0])
            mean_loss = np.mean(losses[-report_each:])
            tq.set_postfix(loss='{:.4f}'.format(mean_loss))
            if i and i % report_each == 0:
                if save_predictions and i % save_prediction_each == 0:
                    p_i = (i // save_prediction_each) % 5
                    save_predictions(root, p_i, inputs, targets, outputs)
        tq.close()
        save(epoch + 1)
        valid_metrics = validation(model, criterion, valid_loader)
        valid_loss = valid_metrics['valid_loss']
        valid_losses.append(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            shutil.copy(str(model_path), str(best_model_path))
        elif patience and epoch - lr_reset_epoch > patience and min(valid_losses[-patience:]) > best_valid_loss:
            # "patience" epochs without improvement
            lr /= 5
            lr_reset_epoch = epoch
            optimizer = init_optimizer(lr)


def add_args(parser):
    arg = parser.add_argument
    arg('root', help='checkpoint root')
    arg('--batch-size', type=int, default=4)
    arg('--n-epochs', type=int, default=100)
    arg('--lr', type=float, default=0.0001)
    arg('--workers', type=int, default=2)
    arg('--fold', type=int, default=1)
    arg('--n-folds', type=int, default=5)
    arg('--clean', action='store_true')
    arg('--epoch-size', type=int)


if __name__ == '__main__':
    random_state = 2016

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--mode', choices=['train', 'valid', 'predict_valid', 'predict_test'],
        default='train')
    utils.add_args(parser)
    args = parser.parse_args()

    data_path = '../data'

    labels = pd.read_csv(os.path.join(data_path, 'train_labels.csv'))

    train_path = os.path.join(data_path, 'train-jpg')

    img_ext = '.jpg'
    train_labels_path = os.path.join(data_path, 'train_v2.csv')

    transformations_train = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transformations_val = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_index, val_index = get_train_test_split(labels)

    labels_df = pd.read_csv(train_labels_path)
    train_labels_df = labels_df.loc[train_index]
    val_labels_df = labels_df.loc[val_index]

    dset_train = KaggleAmazonDataset(train_labels_df, train_path, img_ext, transformations_train)

    dset_val = KaggleAmazonDataset(val_labels_df, train_path, img_ext, transformations_val)

    train_loader = DataLoader(dset_train,
                              batch_size=16,
                              shuffle=True,
                              num_workers=1,  # 1 for CUDA
                              pin_memory=True  # CUDA only
                              )

    val_loader = DataLoader(dset_val,
                            batch_size=16,
                            shuffle=False,
                            num_workers=1,  # 1 for CUDA pin_memory=True  # CUDA only
                            pin_memory=True
                            )

    num_classes = 17
    model = resnet50(pretrained=True).cuda()
    model.fc = nn.Linear(model.fc.in_features, num_classes).cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = MultiLabelSoftMarginLoss()

    n_epochs = 2

    train(
        init_optimizer=lambda lr: Adam(model.parameters(), lr=lr),
        args=args,
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        valid_loader=val_loader,
        validation=validation,
        # save_predictions=save_predictions,
        patience=2,
    )
