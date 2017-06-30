"""
Let's fine tune resnet for weather conditions
"""
from __future__ import division

import pandas as pd
import os
from keras import backend as K
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense
from keras.models import Model
import h5py
import datetime
from keras.callbacks import EarlyStopping, ModelCheckpoint, History
from keras.layers import merge
import tensorflow as tf
from keras.layers.core import Lambda
import numpy as np
import random
from keras.optimizers import Adam


def make_parallel(model, gpu_count):
    __author__ = "kuza55"

    # https://github.com/kuza55/keras-extras/blob/master/utils/multi_gpu.py

    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat([shape[:1] // parts, shape[1:]], axis=0)
        stride = tf.concat([shape[:1] // parts, shape[1:] * 0], axis=0)
        start = stride * idx
        return tf.slice(data, start, size)

    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    # Place a copy of the model on each GPU, each getting a slice of the batch
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:

                inputs = []
                # Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx': i, 'parts': gpu_count})(x)
                    inputs.append(slice_n)

                outputs = model(inputs)

                if not isinstance(outputs, list):
                    outputs = [outputs]

                # Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        if len(outputs_all) == 1:
            merged.append(merge(outputs_all[0], mode='concat', concat_axis=0, name='output'))
        else:
            for outputs in outputs_all:
                merged.append(merge(outputs, mode='concat', concat_axis=0))

        return Model(input=model.inputs, output=merged)


def get_model():
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    x = base_model.output
    predictions = Dense(17, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def save_model(model, cross):
    json_string = model.to_json()
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    json_name = 'architecture_' + cross + '.json'
    weight_name = 'model_weights_' + cross + '.h5'
    open(os.path.join('cache', json_name), 'w').write(json_string)
    model.save_weights(os.path.join('cache', weight_name), overwrite=True)


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


def save_history(history, suffix):
    if not os.path.isdir('history'):
        os.mkdir('history')
    filename = 'history/history_' + suffix + '.csv'
    pd.DataFrame(history.history).to_csv(filename, index=False)


def form_batch(X, y, batch_size):
    X_batch = np.zeros((batch_size, 256, 256, 3))
    y_batch = np.zeros((batch_size, num_classes))

    for i in range(batch_size):
        random_image = random.randint(0, X.shape[0] - 1)

        y_batch[i] = np.array(y[random_image])
        X_batch[i] = np.array(X[random_image])
    return X_batch, y_batch


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def batch_generator(X, y, batch_size, horizontal_flip=False, vertical_flip=False, swap_axis=False):
    while True:
        X_batch, y_batch = form_batch(X, y, batch_size)

        for i in range(X_batch.shape[0]):
            xb = X_batch[i]

            if horizontal_flip:
                if np.random.random() < 0.5:
                    xb = flip_axis(xb, 1)

            if vertical_flip:
                if np.random.random() < 0.5:
                    xb = flip_axis(xb, 0)

            if swap_axis:
                if np.random.random() < 0.5:
                    xb = xb.swapaxes(0, 1)

            X_batch[i] = xb

        yield X_batch, y_batch


if __name__ == '__main__':
    random_state = 2016
    data_path = '../data'

    num_classes = 17

    model = make_parallel(get_model(), 2)

    print('[{}] Compiling model...'.format(str(datetime.datetime.now())))

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

    history = History()

    callbacks = [
        ModelCheckpoint('cache/resnet_full_' + suffix + '.hdf5', monitor='val_loss',
                        save_best_only=True, verbose=1),
        EarlyStopping(patience=5, monitor='val_loss'),
        history
    ]

    save_model(model, "{batch_size}_{suffix}".format(batch_size=batch_size, suffix=suffix))

    model.compile(optimizer=Adam(lr=1e-3), loss=total_loss, metrics=[fbeta, 'binary_crossentropy'])

    model.fit_generator(batch_generator(X_train,
                                        y_train,
                                        batch_size,
                                        vertical_flip=True,
                                        horizontal_flip=True,
                                        swap_axis=True),
                        steps_per_epoch=1000,
                        callbacks=callbacks,
                        validation_data=(X_val, y_val),
                        epochs=500)

    model.compile(optimizer=Adam(lr=1e-4), loss=total_loss, metrics=[fbeta, 'binary_crossentropy'])

    model.load_weights('cache/resnet_full_' + suffix + '.hdf5')

    model.fit_generator(batch_generator(X_train,
                                        y_train,
                                        batch_size,
                                        vertical_flip=True,
                                        horizontal_flip=True,
                                        swap_axis=True),
                        steps_per_epoch=1000,
                        callbacks=callbacks,
                        validation_data=(X_val, y_val),
                        epochs=500)

    save_history(history, suffix)

    f_train.close()
    f_val.close()
