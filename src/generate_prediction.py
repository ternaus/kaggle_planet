"""
Script generates predictions from model.
"""


import utils
import augmentations
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.models import resnet50, resnet101
from pathlib import Path
import torch.nn.functional as F
from sklearn.metrics import fbeta_score
from torch import nn
from torchvision import transforms
import torch
from sklearn.metrics import log_loss
import h5py
import os
from pt_model import get_model
import shutil


def f2_score(y_true, y_pred):
    # fbeta_score throws a confusing error if inputs are not numpy arrays
    y_true, y_pred, = np.array(y_true), np.array(y_pred)
    return fbeta_score(y_true, y_pred, beta=2, average='samples')


class PredictionDatasetPure:
    def __init__(self, paths, n_test_aug):
        self.paths = paths
        self.n_test_aug = n_test_aug

    def __len__(self):
        return len(self.paths) * self.n_test_aug

    def __getitem__(self, idx):
        path = self.paths[idx % len(self.paths)]
        image = utils.load_image(path)
        return valid_transform_pure(image), path.stem


class PredictionDatasetAug:
    def __init__(self, paths, n_test_aug):
        self.paths = paths
        self.n_test_aug = n_test_aug

    def __len__(self):
        return len(self.paths) * self.n_test_aug

    def __getitem__(self, idx):
        path = self.paths[idx % len(self.paths)]
        image = utils.load_image(path)
        return valid_transform_aug(image), path.stem


def predict(model, paths, batch_size: int, n_test_aug: int, aug=False):
    if aug:
        loader = DataLoader(
            dataset=PredictionDatasetAug(paths, n_test_aug),
            shuffle=False,
            batch_size=batch_size,
            num_workers=8,
            pin_memory=True
        )
    else:
        loader = DataLoader(
            dataset=PredictionDatasetPure(paths, n_test_aug),
            shuffle=False,
            batch_size=batch_size,
            num_workers=8,
            pin_memory=True
        )

    model.eval()
    all_outputs = []
    all_stems = []
    for inputs, stems in tqdm(loader, desc='Predict'):
        inputs = utils.variable(inputs, volatile=True)
        outputs = F.sigmoid(model(inputs))
        all_outputs.append(outputs.data.cpu().numpy())
        all_stems.extend(stems)

    return np.vstack(all_outputs), all_stems


def threashold_pred(y_pred, dict_th):
    temp = y_pred.copy()

    for c1, value in dict_th.items():
        temp[:, c1] = (temp[:, c1] > value).astype(np.float32)
    return temp.astype(int)


def find_threasholds(y_true, y_pred):
    num_classes = 17
    threasholds = dict(zip(range(num_classes), [0.2] * num_classes))

    for c in range(num_classes):
        temp = y_pred.copy()

        temp = threashold_pred(temp, threasholds)

        scores = []

        t_range = np.arange(0.1, 0.5, 0.01)

        temp1 = temp.copy()

        for t in t_range:
            temp1[:, c] = (y_pred[:, c] > t)
            scores += [f2_score(y_true.values, temp1)]

        max_score_index = np.argmax(scores)
        max_threashold = t_range[max_score_index]

        threasholds[c] = max_threashold

    assert f2_score(y_true.values, threashold_pred(y_pred, threasholds)) > f2_score(y_true.values, (y_pred > 0.2))
    return threasholds


def apply_threasholds(y_pred, threasholds):
    temp = y_pred.copy()
    for key, value in threasholds.items():
        temp[:, key] = temp[:, key] > value.astype(np.float32)
    return temp


def group_aug(val_p):
    """
    Average augmented predictions
    :param val_p_aug:
    :return:
    """
    df = pd.DataFrame(val_p[0])
    df['id'] = val_p[1]
    g = df.groupby('id').mean()
    g = g.reset_index()
    g = g.sort_values(by='id')
    return g.drop('id', 1).values, g['id'].values


if __name__ == '__main__':
    batch_size = 192
    num_classes = 17
    num_aug = 5

    data_path = '../data'
    model_name = 'resnet101'

    try:
        os.mkdir(os.path.join(data_path, 'predictions'))
    except:
        pass

    try:
        os.mkdir(os.path.join(data_path, 'predictions', model_name))
    except:
        pass

    sample = pd.read_csv(os.path.join(data_path, 'sample_submission_v2.csv'))
    test_paths = [os.path.join('../data/test-jpg', x + '.jpg') for x in sample['image_name']]

    valid_transform_pure = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    valid_transform_aug = transforms.Compose([
        transforms.Scale(256),
        transforms.RandomCrop(224),
        augmentations.D4(),
        # transforms.RandomHorizontalFlip(),
        # augmentations.RandomVerticalFlip(0.5),
        # augmentations.Random90Rotation(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    for fold in range(0, 10):
        fold_dir = os.path.join(data_path, 'predictions', 'fold{fold}_{model_name}'.format(fold=fold, model_name=model_name))
        try:
            os.mkdir(fold_dir)
        except:
            pass

        val_labels = pd.read_csv('../data/fold{fold}/val.csv'.format(fold=fold))
        val_labels['id'] = val_labels['path'].str.split('/').str.get(-1)

        y_true = val_labels.sort_values(by='id').drop(['path', 'id'], 1)

        new_columns = y_true.columns

        model = get_model(num_classes, model_name)
        model = nn.DataParallel(model, device_ids=[0, 1]).cuda()

        state = torch.load('../src/models/{model_name}/best-model_{fold}.pt'.format(fold=fold, model_name=model_name))
        shutil.copy('../src/models/{model_name}/best-model_{fold}.pt'.format(fold=fold, model_name=model_name),
                    os.path.join(fold_dir, 'best-model_{fold}.pt'.format(fold=fold, model_name=model_name)))

        epoch = state['epoch']
        step = state['step']
        best_valid_loss = state['best_valid_loss']
        model.load_state_dict(state['model'])

        val_p = predict(model, val_labels['path'].apply(Path), batch_size, 1, aug=False)

        val_predictions = val_p[0]
        val_image_names = val_p[1]

        val_predictions, val_image_names = group_aug(val_p)

        val_p_aug = predict(model, val_labels['path'].apply(Path), batch_size, num_aug, aug=True)

        val_predictions_aug, val_image_names_aug = group_aug(val_p_aug)

        # Find val_loss
        val_loss = log_loss(y_true.values.ravel(), val_predictions.ravel(), eps=1e-7)
        print('val_loss = ', val_loss)

        val_loss_aug = log_loss(y_true.values.ravel(), val_predictions_aug.ravel(), eps=1e-7)
        print('val_loss_aug = ', val_loss_aug)

        assert val_loss > val_loss_aug

        # Find raw fbeta loss
        raw_f2 = f2_score(y_true.values, val_predictions > 0.2)
        print('raw f2 = ', raw_f2)

        raw_f2_aug = f2_score(y_true.values, val_predictions_aug > 0.2)
        print('raw f2 aug = ', raw_f2_aug)

        # Find threasholds
        threasholds = find_threasholds(y_true, val_predictions)
        val_predictions_threasholded = apply_threasholds(val_predictions, threasholds)
        tuned_f2 = f2_score(y_true.values, val_predictions_threasholded)
        print('tuned f2 = ', tuned_f2)

        # Find threasholds aug
        threasholds_aug = find_threasholds(y_true, val_predictions_aug)
        val_predictions_threasholded_aug = apply_threasholds(val_predictions_aug, threasholds_aug)
        tuned_f2_aug = f2_score(y_true.values, val_predictions_threasholded_aug)
        print('tuned f2 aug = ', tuned_f2_aug)

        assert tuned_f2 < tuned_f2_aug

        test_p = predict(model, list(map(Path, test_paths)), batch_size, 1, aug=False)
        test_predictions, test_image_names = group_aug(test_p)

        test_p_aug = predict(model, list(map(Path, test_paths)), batch_size, num_aug, aug=True)
        test_predictions_aug, test_image_names_aug = group_aug(test_p_aug)

        df = pd.DataFrame(test_predictions, columns=new_columns)
        df.index = test_image_names
        df.to_hdf(os.path.join(fold_dir, 'test_center.h5'), key='prob')

        df = pd.DataFrame(test_predictions_aug, columns=new_columns)
        df.index = test_image_names_aug
        df.to_hdf(os.path.join(fold_dir, 'test_{num_aug}.h5'.format(num_aug=num_aug)), key='prob')

        df = pd.DataFrame(val_predictions, columns=new_columns)
        df.index = val_image_names
        df.to_hdf(os.path.join(fold_dir, 'val_center.h5'), key='prob')

        df = pd.DataFrame(val_predictions_aug, columns=new_columns)
        df.index = val_image_names_aug
        df.to_hdf(os.path.join(fold_dir, 'val_{num_aug}.h5'.format(num_aug=num_aug)), key='prob')

        # Save to h5py
        f = h5py.File(os.path.join(data_path, 'predictions', model_name, 'val_pred_{fold}.hdf5'.format(fold=fold)), 'w')

        f['val_prediction'] = val_predictions
        f['val_prediction_aug'] = val_predictions_aug

        f['val_true'] = y_true.values
        f['val_ids'] = list(map(lambda x: int(x.split('_')[-1].split('.')[0]), val_image_names))
        f['val_ids_aug'] = list(map(lambda x: int(x.split('_')[-1].split('.')[0]), val_image_names_aug))

        f['val_loss'] = val_loss
        f['val_loss_aug'] = val_loss_aug

        f['raw_f2'] = raw_f2
        f['tuned_f2'] = tuned_f2

        f['raw_f2_aug'] = raw_f2_aug
        f['tuned_f2_aug'] = tuned_f2_aug

        threasholds_keys, threasholds_values = zip(*threasholds.items())
        f['threasholds_keys'] = threasholds_keys
        f['threasholds_values'] = threasholds_values

        threasholds_keys_aug, threasholds_values_aug = zip(*threasholds_aug.items())
        f['threasholds_keys_aug'] = threasholds_keys_aug
        f['threasholds_values_aug'] = threasholds_values_aug

        f['test_preds'] = test_predictions

        f['test_preds_aug'] = test_predictions_aug

        f.close()
