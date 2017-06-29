"""
Let's fine tune resnet for weather conditions
"""
from __future__ import division

import pandas as pd
import os
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense
from keras.models import Model
import h5py
import datetime
from keras.callbacks import EarlyStopping, ModelCheckpoint


def get_model():
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    x = base_model.output
    predictions = Dense(17, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def fbeta(y_true, y_pred, threshold_shift=0):
    beta = 2

    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    return (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())


def fbeta_loss(y_true, y_pred):
    beta_squared = 4

    tp = K.sum(y_true * y_pred) + K.epsilon()
    fp = K.sum(y_pred) - tp
    fn = K.sum(y_true) - tp

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    result = (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())

    return result


def total_loss(y_true, y_pred):
    return K.binary_crossentropy(y_pred, y_true) - K.log(fbeta_loss(y_true, y_pred))


if __name__ == '__main__':
    random_state = 2016
    data_path = '../data'

    model = get_model()
    print('[{}] Compiling model...'.format(str(datetime.datetime.now())))
    model.compile(optimizer='adam', loss=total_loss, metrics=['accuracy', fbeta, 'binary_crossentropy'])

    f_train = h5py.File(os.path.join(data_path, 'train_jpg.h5'))
    f_val = h5py.File(os.path.join(data_path, 'val_jpg.h5'))

    X_train = f_train['X']
    y_train = f_train['y']

    X_val = f_val['X']
    y_val = f_val['y']

    batch_size = 32

    print X_train.shape, y_train.shape
    print X_val.shape, y_val.shape

    now = datetime.datetime.now()
    suffix = str(now.strftime("%Y-%m-%d-%H-%M"))

    callbacks = [
        ModelCheckpoint('cache/resnet_full_' + suffix + '.hdf5', monitor='val_loss',
                        save_best_only=True, verbose=1),
        EarlyStopping(patience=20, monitor='val_loss'),
    ]

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=500, callbacks=callbacks, shuffle="batch",
              batch_size=batch_size)

    f_train.close()
    f_val.close()