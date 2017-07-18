"""
Frustrating business with the fact that some tifs in Private are wrongly named
"""

import pandas as pd
import shutil
from tqdm import tqdm

import os


target_dir = '../data/test-tif-v2'
temp_dir = os.path.join(target_dir, 'temp')

os.mkdir(temp_dir)

mapping = pd.read_csv('../data/test_v2_file_mapping.csv')

for i in tqdm(mapping.index):
    old_file_name = mapping.loc[i, 'old']
    new_file_name = mapping.loc[i, 'new']

    shutil.move(os.path.join(target_dir, old_file_name), os.path.join(temp_dir, new_file_name))

for i in tqdm(mapping.index):
    new_file_name = mapping.loc[i, 'new']

    shutil.move(os.path.join(temp_dir, new_file_name), os.path.join(target_dir, new_file_name))
