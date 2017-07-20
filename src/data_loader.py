"""

Data Loader

@author Evgeny Nizhibitsky

"""
import numpy as np
import pandas as pd
from PIL import Image

import torch.utils.data as data
import torchvision.transforms as transforms
import tifffile as tiff
import augmentations
# import skimage.io
import os
import cv2
from pathlib import Path


percentile_eps = 0.5
function_resoution = 32

jpg_folder = '../data/train-jpg'
tif_folder = '../data/train-tif-v2'


def match_percentiles(im_tif, im_jpg):
    """
    from https://www.kaggle.com/bguberfain/tif-to-jpg-by-matching-percentiles
    :param im_tif:
    :param im_jpg:
    :return:
    """
    # Lineary distribute the percentiles
    percentiles = np.linspace(percentile_eps, 100 - percentile_eps, function_resoution)

    # Calculate the percentiles for TIF and JPG, one per channel
    x_per_channel = [np.percentile(im_tif[..., c].ravel(), percentiles) for c in range(3)]
    y_per_channel = [np.percentile(im_jpg[..., c].ravel(), percentiles) for c in range(3)]

    # This is the main part: we use np.interp to convert intermediate values between
    # percentiles from TIF to JPG
    convert_channel = lambda im, c: np.interp(im[..., c], x_per_channel[c], y_per_channel[c])

    # Convert all channels, join and cast to uint8 at range [0, 255]
    # tif2jpg = lambda im: np.dstack([convert_channel(im, c) for c in range(3)]).clip(0, 255).astype(np.uint8)
    tif2jpg = lambda im: np.dstack([convert_channel(im, c) for c in range(3)])

    # The function could stop here, but we are going to plot a few charts about its results
    im_tif_adjusted = tif2jpg(im_tif[..., :3])

    return im_tif_adjusted[:, :, ::-1]
    # return Image.fromarray(im_tif_adjusted[:, :, ::-1])


class CSVDataset(data.Dataset):
    def __init__(self, df, transform=None):
        # df = pd.read_csv(path)
        self.path = df.iloc[:, 0].values.astype(str)
        self.target = df.iloc[:, 1:].values.astype(np.float32)
        self.transform = transform

    def __len__(self):
        return self.target.shape[0]

    @staticmethod
    def _load_pil(path):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def __getitem__(self, idx):
        X = self._load_pil(self.path[idx])
        if self.transform:
            X = self.transform(X)
        y = self.target[idx]
        return X, y


# class CSVDatasetTiff(data.Dataset):
#     def __init__(self, path, transform=None):
#         df = pd.read_csv(path)
#         self.path = df.iloc[:, 0].values.astype(str)
#         self.target = df.iloc[:, 1:].values.astype(np.float32)
#         self.transform = transform
#
#     def __len__(self):
#         return self.target.shape[0]
#
#     @staticmethod
#     def _load_pil(path):
#         with open(path, 'rb') as f:
#             with Image.open(f) as img:
#                 return img.convert('RGB')
#
#     def __getitem__(self, idx):
#         X = self._load_pil(self.path[idx])
#         if self.transform:
#             X = self.transform(X)
#         y = self.target[idx]
#         return X, y


# class CSVDatasetTiff(data.Dataset):
#     def __init__(self, path, transform=None):
#         df = pd.read_csv(path)
#         self.path = df.iloc[:, 0].values.astype(str)
#         self.target = df.iloc[:, 1:].values.astype(np.float32)
#         self.transform = transform
#
#     def __len__(self):
#         return self.target.shape[0]
#
#     @staticmethod
#     def _load_pil(path):
#         file_name = path.split('/')[-1].split('.')[0]
#
#         im_tif = tiff.imread(os.path.join(tif_folder, file_name + '.tif'))
#         im_tif[:, :, 2] = im_tif[:, :, 3]  # Replace R channel with NIR
#
#         im_jpg = cv2.imread(os.path.join(jpg_folder, file_name + '.jpg'))
#
#         tuned_tif = match_percentiles(im_tif, im_jpg)
#         return tuned_tif
#
#     def __getitem__(self, idx):
#         X = self._load_pil(self.path[idx])
#         if self.transform:
#             X = self.transform(X)
#         y = self.target[idx]
#         return X, y


def get_loaders(batch_size,
                fold,
                train_transform=None,
                valid_transform=None):
    train_df = pd.read_csv(f'../data/fold{fold}/train.csv')

    train_dataset = CSVDataset(train_df, transform=train_transform)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=8,
                                   pin_memory=True)

    if not valid_transform:
        valid_transform = transforms.Compose([
          transforms.Scale(256),
          transforms.CenterCrop(224),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    valid_df = pd.read_csv(f'../data/fold{fold}/train.csv')

    valid_dataset = CSVDataset(valid_df, transform=valid_transform)
    valid_loader = data.DataLoader(valid_dataset,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   num_workers=8,
                                   pin_memory=True)

    return train_loader, valid_loader


def get_loaders_tiff(batch_size,
                     fold,
                     train_transform=None,
                     valid_transform=None):

    train_df = pd.read_csv(f'../data/fold{fold}/train.csv')

    train_df['path'] = train_df['path'].str.replace('train-jpg', 'train-tif-v2_new')

    train_dataset = CSVDataset(train_df, transform=train_transform)

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=8,
                                   pin_memory=True)

    if not valid_transform:
        valid_transform = transforms.Compose([
          # transforms.Scale(256),
          transforms.CenterCrop(224),
            # augmentations.CenterCrop(224),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    valid_df = pd.read_csv(f'../data/fold{fold}/val.csv')
    valid_df['path'] = valid_df['path'].str.replace('train-jpg', 'train-tif-v2_new')

    valid_dataset = CSVDataset(valid_df, transform=valid_transform)

    valid_loader = data.DataLoader(valid_dataset,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   num_workers=8,
                                   pin_memory=True)

    return train_loader, valid_loader


def load_image(path: Path):
    return Image.open(str(path)).convert('RGB')
