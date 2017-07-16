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


if __name__ == '__main__':
    data_path = '../data'
    model_name = 'resnet50'

    result_folder = os.path.join(data_path, 'predictions', model_name, 'visualize')

    try:
        os.mkdir(result_folder)
    except:
        pass

    num_folds = 10
    num_classes = 17

    r_val_prediction_aug = []

    r_val_labels = []

    for fold in tqdm(range(num_folds)):
        f = h5py.File(os.path.join(data_path, 'predictions', model_name, 'val_pred_{fold}.hdf5'.format(fold=fold)))

        val_prediction_aug = np.array(f['val_prediction_aug'])
        r_val_prediction_aug += [val_prediction_aug]

        r_val_labels += [pd.read_csv('../data/fold{fold}/val.csv'.format(fold=fold)).sort_values(by='path')]

        f.close()

    val_true = pd.concat(r_val_labels)
    y_true = val_true.drop('path', 1)

    val_true = val_true.set_index('path')

    val_pred_aug = np.vstack(r_val_prediction_aug)

    print('log_loss_aug = ', log_loss(y_true.values.ravel(), val_pred_aug.ravel(), eps=1e-7))

    for i, path in tqdm(enumerate(val_true.index)):
        file_name = path.split('/')[-1]

        jpg = cv2.imread(path)
        jpg = cv2.cvtColor(jpg, cv2.COLOR_BGR2RGB)

        tiff_path = path.replace('train-jpg', 'train-tif-v2').replace('.jpg', '.tif')
        tf = tiff.imread(tiff_path)

        objects = val_true.columns
        y_pos = np.arange(len(objects))
        performance = val_true.loc[path].values

        fig, ax = plt.subplots(2, 5, figsize=(30, 20))

        G = tf[:, :, 1].astype(np.float64)
        NIR = tf[:, :, 3].astype(np.float64)

        NDVI = 1.0 * (G - NIR) / (G + NIR)

        ax[0, 1].set_title('R', fontsize=30)
        ax[0, 1].imshow(jpg[:, :, 0])
        ax[0, 2].set_title('G', fontsize=30)
        ax[0, 2].imshow(jpg[:, :, 1])
        ax[0, 3].set_title('B', fontsize=30)
        ax[0, 3].imshow(jpg[:, :, 2])
        ax[0, 4].set_title('RGB', fontsize=30)
        ax[0, 4].imshow(jpg)

        ax[1, 1].imshow(tf[:, :, 2])
        ax[1, 1].set_title('R', fontsize=30)
        ax[1, 2].imshow(tf[:, :, 1])
        ax[1, 2].set_title('G', fontsize=30)
        ax[1, 3].imshow(tf[:, :, 0])
        ax[1, 3].set_title('B', fontsize=30)

        ax[1, 4].set_title('NIR', fontsize=30)
        ax[1, 4].imshow(tf[:, :, 3])

        ax[1, 0].set_title('NDVI', fontsize=30)
        ax[1, 0].imshow(NDVI)

        ax[0, 0].barh(y_pos, performance, align='center', alpha=0.5, color='blue')
        ax[0, 0].barh(y_pos, val_pred_aug[i], align='center', alpha=0.5, color='red')

        ax[0, 0].set_yticks(y_pos)
        ax[0, 0].set_yticklabels(objects, fontsize=30)

        plt.tight_layout()

        savefig(os.path.join(result_folder, file_name))
        close()
