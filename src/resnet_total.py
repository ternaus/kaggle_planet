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
from keras.optimizers import Adam, SGD
from imgaug import augmenters as iaa
import imgaug as ia


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

        return Model(inputs=model.inputs, outputs=merged)


def get_model():
    # base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
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
    # return K.binary_crossentropy(y_pred, y_true) - K.log(fbeta_loss(y_true, y_pred))
    return - K.log(fbeta_loss(y_true, y_pred))


def save_history(history, suffix):
    if not os.path.isdir('history'):
        os.mkdir('history')
    filename = 'history/history_' + suffix + '.csv'
    pd.DataFrame(history.history).to_csv(filename, index=False)


def form_batch(X, y, batch_size):
    X_batch = np.zeros((batch_size, 224, 224, 3))
    y_batch = np.zeros((batch_size, num_classes))

    for i in range(batch_size):
        random_image = random.randint(0, X.shape[0] - 1)
        random_h = random.randint(0, 256-224 - 1)
        random_w = random.randint(0, 256-224 - 1)

        y_batch[i] = np.array(y[random_image])
        X_batch[i] = np.array(X[random_image, random_h: random_h + 224, random_w + 224])
    return X_batch, y_batch


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def batch_generator(X, y, batch_size, augment=False):
    while True:
        X_batch, y_batch = form_batch(X, y, batch_size)

        if augment:
            X_batch = seq.augment_images(X_batch.astype(np.uint8))

        X_batch = X_batch.astype(np.float32)

        X_batch[:, :, :, 0] -= 103.939
        X_batch[:, :, :, 1] -= 116.779
        X_batch[:, :, :, 2] -= 123.68

        yield X_batch, y_batch


if __name__ == '__main__':
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            iaa.Flipud(0.5),  # vertically flip 50% of all images
            sometimes(iaa.Crop(percent=(0, 0.1))),  # crop images by 0-10% of their height/width
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
                rotate=(-45, 45),  # rotate by -45 to +45 degrees
                shear=(-16, 16),  # shear by -16 to +16 degrees
                order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 5),
                       [
                           sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                           # convert images into their superpixel representation
                           iaa.OneOf([
                               iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                               iaa.AverageBlur(k=(2, 7)),
                               # blur image using local means with kernel sizes between 2 and 7
                               # iaa.MedianBlur(k=(3, 11)),
                               # blur image using local medians with kernel sizes between 2 and 7
                           ]),
                           iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                           iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
                           # search either for all edges or for directed edges
                           sometimes(iaa.OneOf([
                               iaa.EdgeDetect(alpha=(0, 0.7)),
                               iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
                           ])),
                           iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                           # add gaussian noise to images
                           iaa.OneOf([
                               iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                               iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                           ]),
                           iaa.Invert(0.05, per_channel=True),  # invert color channels
                           iaa.Add((-10, 10), per_channel=0.5),
                           # change brightness of images (by -10 to 10 of original value)
                           iaa.Multiply((0.5, 1.5), per_channel=0.5),
                           # change brightness of images (50-150% of original value)
                           iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                           iaa.Grayscale(alpha=(0.0, 1.0)),
                           sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
                           # move pixels locally around (with random strengths)
                           sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
                           # sometimes move parts of the image around
                       ],
                       random_order=True
                       )
        ],
        random_order=True
    )

    random_state = 2016
    data_path = '../data'

    num_classes = 17

    # model = make_parallel(get_model(), 2)
    model = get_model()

    print('[{}] Compiling model...'.format(str(datetime.datetime.now())))

    f_train = h5py.File(os.path.join(data_path, 'train_jpg.hdf5'))
    f_val = h5py.File(os.path.join(data_path, 'val_jpg.hdf5'))

    X_train = f_train['X']
    y_train = f_train['y']

    X_val = f_val['X']
    y_val = f_val['y']

    batch_size = 64

    print X_train.shape, y_train.shape
    print X_val.shape, y_val.shape

    now = datetime.datetime.now()
    suffix = str(now.strftime("%Y-%m-%d-%H-%M"))

    history = History()

    callbacks = [
        ModelCheckpoint('cache/resnet_full_' + suffix + '.hdf5', monitor='val_loss',
                        save_best_only=True, verbose=1),
        EarlyStopping(patience=10, monitor='val_loss'),
        history
    ]

    save_model(model, "{batch_size}_{suffix}".format(batch_size=batch_size, suffix=suffix))

    # model.load_weights('cache/resnet_full_2017-07-02-22-33.hdf5')
    #
    for layer in model.layers[:-3]:
        layer.trainable = False

    model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=[fbeta])
    # model.compile(optimizer=Adam(lr=1e-3), loss=total_loss, metrics=[fbeta, 'binary_crossentropy'])

    model.fit_generator(batch_generator(X_train,
                                        y_train,
                                        batch_size, augment=True),
                        steps_per_epoch=1000,
                        callbacks=callbacks,
                        validation_data=(X_val, y_val),
                        epochs=500)

    model.load_weights('cache/resnet_full_' + suffix + '.hdf5')

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=[fbeta])
    #
    # model.load_weights('cache/resnet_full_' + suffix + '.hdf5')

    model.fit_generator(batch_generator(X_train,
                                        y_train,
                                        batch_size, augment=True),
                        steps_per_epoch=1000,
                        callbacks=callbacks,
                        validation_data=(X_val, y_val),
                        epochs=500)

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=[fbeta])
    #
    # model.load_weights('cache/resnet_full_' + suffix + '.hdf5')

    model.fit_generator(batch_generator(X_train,
                                        y_train,
                                        batch_size),
                        steps_per_epoch=1000,
                        callbacks=callbacks,
                        validation_data=(X_val, y_val),
                        epochs=500)

    save_history(history, suffix)

    f_train.close()
    f_val.close()
