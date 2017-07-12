"""
Script generates predictions from model.
"""


import utils
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


def f2_score(y_true, y_pred):
    # fbeta_score throws a confusing error if inputs are not numpy arrays
    y_true, y_pred, = np.array(y_true), np.array(y_pred)
    return fbeta_score(y_true, y_pred, beta=2)


class PredictionDataset:
    def __init__(self, paths, n_test_aug):
        self.paths = paths
        self.n_test_aug = n_test_aug

    def __len__(self):
        return len(self.paths) * self.n_test_aug

    def __getitem__(self, idx):
        path = self.paths[idx % len(self.paths)]
        image = utils.load_image(path)
        return valid_transform(image), path.stem


def predict(model, paths, batch_size: int, n_test_aug: int):
    loader = DataLoader(
        dataset=PredictionDataset(paths, n_test_aug),
        shuffle=False,
        batch_size=batch_size,
        num_workers=4,
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


def get_model(num_classes, model_name):
    if model_name == 'resnet50':
        model = resnet50(pretrained=True).cuda()
        model.fc = nn.Linear(model.fc.in_features, num_classes).cuda()
    elif model_name == 'resnet101':
        model = resnet101(pretrained=True).cuda()
        model.fc = nn.Linear(model.fc.in_features, num_classes).cuda()
    return model


def threashold_pred(y_pred, dict_th):
    temp = y_pred.copy()

    for c1, value in dict_th.items():
        temp[:, c1] = (temp[:, c1] > value).astype(np.float32)
    return temp.astype(int)


def find_threasholds(y_true, y_pred):
    threasholds = dict(zip(range(num_classes), [0.2] * num_classes))

    for c in range(num_classes):
        temp = y_pred.copy()

        temp = threashold_pred(temp, threasholds)

        scores = []

        t_range = np.arange(0, 1, 0.01)

        temp1 = temp.copy()

        for t in t_range:
            temp1[:, c] = (y_pred[:, c] > t)
            scores += [f2_score(y_true.values.ravel(), temp1.ravel())]

        max_score_index = np.argmax(scores)
        max_threashold = t_range[max_score_index]

        threasholds[c] = max_threashold

    assert f2_score(y_true.values.ravel(), threashold_pred(y_pred, threasholds).ravel()) > f2_score(y_true.values.ravel(), (y_pred > 0.2).ravel())
    return threasholds


def apply_threasholds(y_pred, threasholds):
    temp = y_pred.copy()
    for key, value in threasholds.items():
        temp[:, key] = temp[:, key] > value.astype(np.float32)
    return temp


if __name__ == '__main__':

    batch_size = 32
    num_classes = 17
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

    valid_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    for fold in range(0, 10):
        val_labels = pd.read_csv('../data/fold{fold}/val.csv'.format(fold=fold))

        y_true = val_labels.drop('path', 1)

        new_columns = [x for x in val_labels.columns if x != 'path']

        model = get_model(num_classes, model_name)
        model = nn.DataParallel(model, device_ids=[0]).cuda()

        state = torch.load('../src/models/{model_name}/best-model_{fold}.pt'.format(fold=fold, model_name=model_name))

        epoch = state['epoch']
        step = state['step']
        best_valid_loss = state['best_valid_loss']
        model.load_state_dict(state['model'])

        val_p = predict(model, val_labels['path'].apply(Path), batch_size, 1)

        val_predictions = val_p[0]
        val_image_names = val_p[1]

        # Find val_loss
        val_loss = log_loss(y_true.values.ravel(), val_predictions.ravel(), eps=1e-7)
        print('val_loss = ', val_loss)
        # Find raw fbeta loss
        raw_f2 = f2_score(y_true.values.ravel(), val_predictions.ravel() > 0.2)
        print('raw f2 = ', raw_f2)
        # Find threasholds
        threasholds = find_threasholds(y_true, val_predictions)
        val_predictions_threasholded = apply_threasholds(val_predictions, threasholds)
        tuned_f2 = f2_score(y_true.values.ravel(), val_predictions_threasholded.ravel())
        print('tuned f2 = ', tuned_f2)

        test_p = predict(model, list(map(Path, test_paths)), batch_size, 1)
        test_predictions = test_p[0]
        test_image_names = test_p[1]

        # Save to h5py
        f = h5py.File(os.path.join(data_path, 'predictions', model_name, 'val_pred_{fold}.hdf5'.format(fold=fold)), 'w')

        f['val_prediction'] = val_predictions
        f['val_true'] = y_true.values
        f['val_ids'] = list(map(lambda x: int(x.split('_')[-1].split('.')[0]), val_image_names))
        f['val_loss'] = val_loss
        f['raw_f2'] = raw_f2
        f['tuned_f2'] = tuned_f2

        threasholds_keys, threasholds_values = zip(*threasholds.items())
        f['threasholds_keys'] = threasholds_keys
        f['threasholds_values'] = threasholds_values

        f['test_preds'] = test_predictions
        f['test_ids'] = list(map(lambda x: int(x.split('_')[-1].split('.')[0]), test_image_names))

        f.close()
