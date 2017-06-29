"""
data does not fit into memory => let's cache it into hdf5
"""
from __future__ import division
import os
import cv2
import numpy as np
from tqdm import tqdm
import h5py

data_path = '../data'
test_path = os.path.join(data_path, 'test-jpg')

random_state = 2016

num_test = len(os.listdir(test_path))

f = h5py.File(os.path.join(data_path, 'test_jpg.h5'), 'w', compression='blosc:lz4', compression_opts=9)

imgs = f.create_dataset('X', (num_test, 256, 256, 3), dtype=np.uint8)

for i, file_name in enumerate(tqdm(os.listdir(test_path))):
    img = cv2.imread(os.path.join(test_path, file_name))

    imgs[i] = img

f.close()
