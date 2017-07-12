import h5py
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import log_loss
from generate_prediction import f2_score, find_threasholds, apply_threasholds


def get_pred_str(x):
    result = []
    for index, value in enumerate(x):
        if value != 0:
            result += [new_columns[index]]
    return ' '.join(result)


if __name__ == '__main__':
    data_path = '../data'
    model_name = 'resnet50'
    num_folds = 10

    r_val_loss = []
    r_val_prediction = []
    r_val_ids = []
    r_raw_f2 = []
    r_tuned_f2 = []
    r_threashold_keys = []
    r_threashold_values = []
    r_test_preds = []
    r_test_ids = []
    r_val_labels = []

    for fold in tqdm(range(10)):
        f = h5py.File(os.path.join(data_path, 'predictions', model_name, 'val_pred_{fold}.hdf5'.format(fold=fold)))
        val_loss = np.array(f['val_loss'])
        val_prediction = np.array(f['val_prediction'])
        val_ids = np.array(f['val_ids'])
        raw_f2 = np.array(f['raw_f2'])
        tuned_f2 = np.array(f['tuned_f2'])
        th_keys = np.array(f['threasholds_keys'])
        th_values = np.array(f['threasholds_values'])
        test_preds = np.array(f['test_preds'])
        test_ids = np.array(f['test_ids'])

        r_val_loss += [val_loss]
        r_val_prediction += [val_prediction]
        r_val_ids += [val_ids]
        r_raw_f2 += [raw_f2]
        r_tuned_f2 += [tuned_f2]
        r_threashold_keys += [th_keys]
        r_threashold_values += [th_values]
        r_test_preds += [test_preds]
        r_test_ids += [test_ids]
        r_val_labels += [pd.read_csv('../data/fold{fold}/val.csv'.format(fold=fold))]

        f.close()

    val_pred = np.vstack(r_val_prediction)
    val_true = pd.concat(r_val_labels)

    y_true = val_true.drop('path', 1)
    print('log_loss = ', log_loss(y_true.values.ravel(), val_pred.ravel(), eps=1e-7))
    print('f2 score = ', f2_score(y_true.ravel(), val_pred.ravel() > 0.2))
    threasholds = find_threasholds(y_true, val_pred)

    print('threasholds = ', threasholds)

    val_pred_t = apply_threasholds(val_pred, threasholds)
    print('tuned f2score = ', f2_score(y_true.ravel(), val_pred_t.ravel()))

    new_test = np.zeros((num_folds, r_test_preds[0].shape[0], r_test_preds[0].shape[1]))

    for fold in range(num_folds):
        new_test[fold] = r_test_preds[fold]

    test_mean = np.mean(new_test, axis=0)
    test_tr = apply_threasholds(test_mean, threasholds)
    new_columns = [x for x in val_true.columns if x != 'path']

    pred = list(map(get_pred_str, test_tr))

    sample = pd.read_csv('../data/sample_submission_v2.csv')
    submission = sample.copy()
    submission['tags'] = pred
    submission.to_csv(os.path.join(data_path, model_name + '_total_notta.cvs'), index=False)
