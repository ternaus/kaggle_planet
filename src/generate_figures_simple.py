"""
Let's try to visualize what we predict...
"""

from tqdm import tqdm
import os
import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
import cv2
import tifffile as tiff
from pylab import *


NUM_CLASSES = 17
LABELS = ['agriculture', 'artisinal_mine', 'bare_ground', 'blooming',
          'blow_down', 'clear', 'cloudy', 'conventional_mine',
          'cultivation', 'habitation', 'haze', 'partly_cloudy',
          'primary', 'road', 'selective_logging', 'slash_burn', 'water']


def flatten_train(df):
    print(df.head())
    print(df.shape)

    idx2label = dict(enumerate(LABELS))
    label2idx = {v: k for k, v in idx2label.items()}

    targets = np.zeros((df.shape[0], NUM_CLASSES), np.uint8)
    for i, tags in tqdm(list(enumerate(df.tags)), miniters=1000):
        for t in tags.split(' '):
            targets[i][label2idx[t]] = 1

    del df['tags']
    for i, l in enumerate(LABELS):
        df[l] = targets[:, i]
    print(df.head())
    print(df.shape)
    return df


if __name__ == '__main__':
    data_path = '../data'

    result_folder = os.path.join(data_path, 'predictions', 'visualize')

    try:
        os.mkdir(result_folder)
    except:
        pass

    num_folds = 10
    num_classes = 17

    matches = pd.read_csv('../data/image_mosaic.csv')

    y_pred = pd.read_csv(os.path.join('../submissions', 'avg1-20170716-025336.csv')).sort_values('image_name')

    y_pred = y_pred[y_pred['image_name'].str.contains('file')]

    y_pred = y_pred.merge(matches[['image_name', 'region']], on='image_name')

    y_pred = flatten_train(y_pred)

    y_pred = y_pred.set_index('image_name')

    for i, file_name in tqdm(enumerate(y_pred.index)):
        path = os.path.join(data_path, 'test-jpg', file_name + '.jpg')
        jpg = cv2.imread(path)
        jpg = cv2.cvtColor(jpg, cv2.COLOR_BGR2RGB)

        y_pos = np.arange(NUM_CLASSES)

        fig, ax = plt.subplots(1, 2, figsize=(30, 20))

        ax[1].set_title('RGB, {image_name}, {region}'.format(image_name=file_name, region=y_pred.loc[file_name, 'region']), fontsize=30)
        ax[1].imshow(jpg)

        preds = y_pred.loc[file_name].values[1:]

        ax[0].barh(y_pos, preds, align='center', alpha=0.5, color='red')

        ax[0].set_yticks(y_pos)
        ax[0].set_yticklabels(y_pred.columns[1:], fontsize=25)

        plt.tight_layout()

        savefig(os.path.join(result_folder, file_name))
        close()
