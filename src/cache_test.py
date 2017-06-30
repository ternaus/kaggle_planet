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

imgs = f.create_dataset('X', (num_test, 256, 256, 3), dtype=np.float16)

image_names = []

for i, file_name in enumerate(tqdm(os.listdir(test_path))):
    img = cv2.imread(os.path.join(test_path, file_name)).astype(np.float16)

    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68

    imgs[i] = img.astype(np.float16)

    image_names += [file_name]

f['file_name'] = image_names

f.close()
