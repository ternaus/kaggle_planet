"""
training on tiff works too long => let's transform tif to a workable format
"""

import os
from tqdm import tqdm
import cv2
import tifffile as tiff
import data_loader
import numpy as np


train_jpg_folder = '../data/train-jpg'
train_tif_folder = '../data/train-tif-v2'

test_jpg_folder = '../data/test-jpg'
test_tif_folder = '../data/test-tif-v2'

try:
    os.mkdir(train_tif_folder + '_new')
except:
    pass

try:
    os.mkdir(test_tif_folder + '_new')
except:
    pass


train_file_names = os.listdir(train_jpg_folder)

for file_name in tqdm(train_file_names):
    tif_file_name = file_name.replace('jpg', 'tif')

    im_jpg = cv2.imread(os.path.join(train_jpg_folder, file_name))
    im_tif = tiff.imread(os.path.join(train_tif_folder, tif_file_name))

    im_tif[:, :, 2] = im_tif[:, :, 3]  # Replace R channel with NIR
    tuned_tif = data_loader.match_percentiles(im_tif, im_jpg).clip(0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(train_tif_folder + '_new', file_name), tuned_tif)


test_file_names = os.listdir(test_jpg_folder)

for file_name in tqdm(test_file_names):
    tif_file_name = file_name.replace('jpg', 'tif')
    im_jpg = cv2.imread(os.path.join(test_jpg_folder, file_name))
    im_tif = tiff.imread(os.path.join(test_tif_folder, tif_file_name))
    im_tif[:, :, 2] = im_tif[:, :, 3]  # Replace R channel with NIR
    tuned_tif = data_loader.match_percentiles(im_tif, im_jpg).clip(0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(test_tif_folder + '_new', file_name), tuned_tif)
