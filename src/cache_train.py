"""
data does not fit into memory => let's cache it into hdf5
"""
from __future__ import division
import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
import h5py

data_path = '../data'
train_path = os.path.join(data_path, 'train-jpg')

random_state = 2016

labels = pd.read_csv(os.path.join(data_path, 'train_labels.csv'))

labels['image_name'] = labels['image_name'] + '.jpg'

labels['unified'] = np.nan
labels.loc[labels['clear'] == 1, 'unified'] = 0
labels.loc[labels['cloudy'] == 1, 'unified'] = 1
labels.loc[labels['haze'] == 1, 'unified'] = 2
labels.loc[labels['partly_cloudy'] == 1, 'unified'] = 3


weather_features = ['clear', 'cloudy', 'haze', 'partly_cloudy']

labels = labels[labels['unified'].notnull()]

sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=random_state)

train_index, val_index = sss.split(labels['unified'].values.astype(int), labels['unified'].values.astype(int)).next()

train_labels = labels.iloc[train_index]
val_labels = labels.iloc[val_index]

num_train = train_labels.shape[0]
num_val = val_labels.shape[0]

# f = h5py.File(os.path.join(data_path, 'train_jpg.h5'), 'w', compression='blosc:lz4', compression_opts=9)
f = h5py.File(os.path.join(data_path, 'train_jpg.h5'), 'w')

imgs = f.create_dataset('X', (num_train, 256, 256, 3), dtype=np.uint8)

for i, file_name in enumerate(tqdm(train_labels['image_name'])):
    img = cv2.imread(os.path.join(train_path, file_name))

    imgs[i] = img.astype(np.uint8)

f['y'] = train_labels.drop(['image_name', 'unified'], 1).values

f.close()
# f = h5py.File(os.path.join(data_path, 'val_jpg.h5'), 'w', compression='blosc:lz4', compression_opts=9)
f = h5py.File(os.path.join(data_path, 'val_jpg.h5'), 'w')

imgs = f.create_dataset('X', (num_val, 256, 256, 3), dtype=np.uint8)

for i, file_name in enumerate(tqdm(val_labels['image_name'])):
    img = cv2.imread(os.path.join(train_path, file_name)).astype(np.uint8)
    imgs[i] = img.astype(np.uint8)

f['y'] = val_labels.drop(['image_name', 'unified'], 1).values

print np.array(f['y']).shape

f.close()
