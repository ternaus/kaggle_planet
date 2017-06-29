"""
data does not fit into memory => let's cache it into hdf5
"""
from __future__ import division

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
import shutil
import os

data_path = os.path.abspath('../data')
train_path = os.path.join(data_path, 'train-jpg')


for i in ['train', 'val']:
    try:
        os.mkdir(os.path.join(data_path, i + '_weather'))
    except:
        pass

    for class_name in ['0', '1', '2', '3']:
        try:
            os.mkdir(os.path.join(data_path, i + '_weather', class_name))
        except:
            pass

random_state = 2016

labels = pd.read_csv(os.path.join(data_path, 'train_labels.csv'))

labels['unified'] = np.nan
labels.loc[labels['clear'] == 1, 'unified'] = '0'
labels.loc[labels['cloudy'] == 1, 'unified'] = '1'
labels.loc[labels['haze'] == 1, 'unified'] = '2'
labels.loc[labels['partly_cloudy'] == 1, 'unified'] = '3'

labels['image_name'] = labels['image_name'] + '.jpg'

labels = labels[labels['unified'].notnull()]

sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=random_state)

train_index, val_index = sss.split(labels['unified'].values.astype(int), labels['unified'].values.astype(int)).next()

train_labels = labels.iloc[train_index]
val_labels = labels.iloc[val_index]

num_train = train_labels.shape[0]
num_val = train_labels.shape[0]

for file_name in tqdm(train_labels['image_name']):
    class_name = train_labels.loc[train_labels['image_name'] == file_name, 'unified'].values[0]
    shutil.copy(os.path.join(train_path, file_name), os.path.join(data_path, 'train_weather', class_name, file_name))


for file_name in tqdm(val_labels['image_name']):
    class_name = val_labels.loc[val_labels['image_name'] == file_name, 'unified'].values[0]
    shutil.copy(os.path.join(train_path, file_name), os.path.join(data_path, 'val_weather', class_name, file_name))
