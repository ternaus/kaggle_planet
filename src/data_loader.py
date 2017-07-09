"""

Data Loader

@author Evgeny Nizhibitsky

"""
import numpy as np
import pandas as pd
from PIL import Image

import torch.utils.data as data
import torchvision.transforms as transforms


class CSVDataset(data.Dataset):
  def __init__(self, path, transform=None):
    df = pd.read_csv(path)
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


def get_loaders(batch_size, fold=0):
    train_transform = transforms.Compose([
      transforms.RandomSizedCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_dataset = CSVDataset(f'../data/fold{fold}/train.csv', transform=train_transform)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=4,
                                   pin_memory=True)

    valid_transform = transforms.Compose([
      transforms.Scale(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    valid_dataset = CSVDataset(f'../data/fold{fold}/val.csv', transform=valid_transform)
    valid_loader = data.DataLoader(valid_dataset,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   num_workers=4,
                                   pin_memory=True)

    return train_loader, valid_loader
