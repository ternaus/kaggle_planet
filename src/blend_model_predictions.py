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
    num_classes = 17

    r_val_loss = []
    r_val_prediction = []
    r_val_loss_aug = []
    r_val_prediction_aug = []

    r_val_ids = []
    r_raw_f2 = []
    r_tuned_f2 = []
    r_threashold_keys = []
    r_threashold_values = []
    r_test_preds = []
    r_test_preds_aug = []

    r_test_ids = []
    r_val_labels = []

    for fold in tqdm(range(10)):
        f = h5py.File(os.path.join(data_path, 'predictions', model_name, 'val_pred_{fold}.hdf5'.format(fold=fold)))
        val_loss = np.array(f['val_loss'])
        val_prediction = np.array(f['val_prediction'])
        val_loss_aug = np.array(f['val_loss_aug'])
        val_prediction_aug = np.array(f['val_prediction_aug'])

        val_ids = np.array(f['val_ids'])
        raw_f2 = np.array(f['raw_f2'])
        tuned_f2 = np.array(f['tuned_f2'])
        th_keys = np.array(f['threasholds_keys'])
        th_values = np.array(f['threasholds_values'])
        test_preds = np.array(f['test_preds'])

        test_preds_aug = np.array(f['test_preds_aug'])
        # test_ids = np.array(f['test_ids'])

        r_val_loss += [val_loss]
        r_val_prediction += [val_prediction]

        r_val_loss_aug += [val_loss_aug]
        r_val_prediction_aug += [val_prediction_aug]

        r_val_ids += [val_ids]
        r_raw_f2 += [raw_f2]
        r_tuned_f2 += [tuned_f2]
        r_threashold_keys += [th_keys]
        r_threashold_values += [th_values]
        r_test_preds += [pd.read_hdf(os.path.join(data_path, 'predictions', model_name, 'test_predictions_{fold}.hdf5'.format(fold=fold)))]
        r_test_preds_aug += [pd.read_hdf(
            os.path.join(data_path, 'predictions', model_name, 'test_predictions_aug_{fold}.hdf5'.format(fold=fold)))]

        r_val_labels += [pd.read_csv('../data/fold{fold}/val.csv'.format(fold=fold)).sort_values(by='path')]

        f.close()

    val_pred = np.vstack(r_val_prediction)
    val_true = pd.concat(r_val_labels)

    val_pred_aug = np.vstack(r_val_prediction_aug)

    y_true = val_true.drop('path', 1)

    print('log_loss = ', log_loss(y_true.values.ravel(), val_pred.ravel(), eps=1e-7))
    print('f2 score = ', f2_score(y_true.values.ravel(), val_pred.ravel() > 0.2))
    threasholds = find_threasholds(y_true, val_pred)
    print('threasholds = ', threasholds)

    print('log_loss_aug = ', log_loss(y_true.values.ravel(), val_pred_aug.ravel(), eps=1e-7))
    print('f2 score aug = ', f2_score(y_true.values.ravel(), val_pred_aug.ravel() > 0.2))
    threasholds = find_threasholds(y_true, val_pred)
    print('threasholds aug = ', threasholds)

    val_pred_t = apply_threasholds(val_pred, threasholds)
    print('tuned f2score = ', f2_score(y_true.values.ravel(), val_pred_t.ravel()))

    val_pred_aug_t = apply_threasholds(val_pred_aug, threasholds)
    print('tuned f2score aug = ', f2_score(y_true.values.ravel(), val_pred_aug_t.ravel()))

    new_test = pd.concat(r_test_preds)
    new_test_aug = pd.concat(r_test_preds_aug)

    test_mean = new_test.groupby('image_name').mean().reset_index().sort_values(by='image_name')
    test_ids = test_mean['image_name']

    test_aug_mean = new_test_aug.groupby('image_name').mean().reset_index().sort_values(by='image_name')
    test_aug_ids = test_aug_mean['image_name']

    test_tr = apply_threasholds(test_mean.drop('image_name', 1).values, threasholds)
    test_aug_tr = apply_threasholds(test_aug_mean.drop('image_name', 1).values, threasholds)

    new_columns = [x for x in test_mean.columns if x != 'image_name']

    pred = list(map(get_pred_str, test_tr))

    pred_aug = list(map(get_pred_str, test_aug_tr))

    submission = pd.DataFrame({'tags': pred, 'image_name': test_ids})
    submission.to_csv(os.path.join(data_path, model_name + '_total.cvs'), index=False)

    submission_aug = pd.DataFrame({'tags': pred_aug, 'image_name': test_aug_ids})
    submission_aug.to_csv(os.path.join(data_path, model_name + '_aug_3.cvs'), index=False)
