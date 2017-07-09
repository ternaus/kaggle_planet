#!/usr/bin/env python3
"""
Split train into folds

@author Evgeny Nizhibitsky
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

NUM_FOLDS = 10
NUM_CLASSES = 17
LABELS = ['agriculture', 'artisinal_mine', 'bare_ground', 'blooming',
          'blow_down', 'clear', 'cloudy', 'conventional_mine',
          'cultivation', 'habitation', 'haze', 'partly_cloudy',
          'primary', 'road', 'selective_logging', 'slash_burn', 'water']


def flatten_train():
    df = pd.read_csv('data/train_v2.csv')
    print(df.head())
    print(df.shape)

    idx2label = dict(enumerate(LABELS))
    label2idx = {v: k for k, v in idx2label.items()}

    targets = np.zeros((df.shape[0], NUM_CLASSES), np.uint8)
    for i, tags in tqdm(list(enumerate(df.tags)), miniters=1000):
        for t in tags.split(' '):
            targets[i][label2idx[t]] = 1

    del df['tags']
    for i, l in enumerate(LABELS):
        df[l] = targets[:, i]
    print(df.head())
    print(df.shape)

    df.to_csv('data/train_flat.csv', index=False)


def make_folds():
    df = pd.read_csv('data/train_flat.csv')
    del df['image_name']
    targets = df.as_matrix().astype(np.int32)
    print(targets.shape)

    idx_primary = LABELS.index('primary')
    idx_clear = LABELS.index('clear')

    # whilst this classes' counts are large, distribute 0's equally at first...
    targets[:, idx_primary] = 1 - targets[:, idx_primary]
    targets[:, idx_clear] = 1 - targets[:, idx_clear]
    tags_total = np.sum(targets, axis=0)

    # plt.plot(sorted(tags_total))
    # plt.savefig('tags_total.png', bbox_inches='tight', pad_inches=0)

    folds = [[] for _ in range(NUM_FOLDS)]
    fold_tags_cnt = np.zeros((NUM_FOLDS, NUM_CLASSES), np.int32)
    fold_tags_cnt_true = np.zeros((NUM_FOLDS, NUM_CLASSES), np.int32)
    distributed = set()
    idx2label = dict(enumerate(LABELS))

    for tag_idx in np.argsort(tags_total).tolist():
        samples = list(set(np.array(np.nonzero(targets[:, tag_idx]))[0]) - distributed)
        np.random.seed(42 + tag_idx)
        np.random.shuffle(samples)

        print(f'{tags_total[tag_idx]:5d}: {idx2label[tag_idx]} ({len(samples)})')
        print(fold_tags_cnt[:, tag_idx])

        for sample in samples:
            min_args = np.argwhere(fold_tags_cnt[:, tag_idx] == np.min(fold_tags_cnt[:, tag_idx]))
            fold_idx = np.random.choice(min_args.flatten())
            folds[fold_idx].append(sample)
            target = targets[sample].copy()

            fold_tags_cnt[fold_idx, :] += target
            target[idx_primary] = 1 - target[idx_primary]
            target[idx_clear] = 1 - target[idx_clear]
            fold_tags_cnt_true[fold_idx, :] += target

        print(fold_tags_cnt[:, tag_idx])
        assert len(distributed & set(samples)) == 0
        distributed |= set(samples)

    # ...and then finish distribution of 1's when all other classes are done
    targets = df.as_matrix().astype(np.int32)
    tags_total = np.sum(targets, axis=0)

    fold_tags_cnt = fold_tags_cnt_true

    for tag_idx in [LABELS.index('primary'), LABELS.index('clear')]:
        samples = list(set(np.array(np.nonzero(targets[:, tag_idx]))[0]) - distributed)
        np.random.seed(42 + tag_idx)
        np.random.shuffle(samples)

        print(f'{tags_total[tag_idx]:5d}: {idx2label[tag_idx]} ({len(samples)})')
        print(fold_tags_cnt[:, tag_idx])

        for sample in samples:
            min_args = np.argwhere(fold_tags_cnt[:, tag_idx] == np.min(fold_tags_cnt[:, tag_idx]))
            fold_idx = np.random.choice(min_args.flatten())
            folds[fold_idx].append(sample)
            fold_tags_cnt[fold_idx, :] += targets[sample]

        print(fold_tags_cnt[:, tag_idx])
        assert len(distributed & set(samples)) == 0
        distributed |= set(samples)

    for i in range(1, NUM_FOLDS):
        assert np.all(np.equal(fold_tags_cnt[0, :] // 100, fold_tags_cnt[i, :] // 100))
    print(fold_tags_cnt % 100)
    print(fold_tags_cnt.sum(1))
    print([len(fold) for fold in folds])

    # print(folds[0][:3])
    # print(folds[-1][-3:])

    assert fold_tags_cnt.sum() == 116205
    assert len(distributed) == df.shape[0]
    assert np.concatenate([np.array(f) for f in folds], axis=0).shape[0] == df.shape[0]
    assert np.all(np.equal(folds[0][:3], [3152, 31495, 18964]))
    assert np.all(np.equal(folds[-1][-3:], [22790, 36269, 7618]))

    np.save(f'data/{NUM_FOLDS}_folds', np.array([np.array(f) for f in folds]))


def plot_folds():
    df = pd.read_csv('data/train_flat.csv')
    del df['image_name']
    targets = df.as_matrix()

    folds = np.load(f'data/{NUM_FOLDS}_folds.npy')
    assert np.concatenate([np.array(f) for f in folds], axis=0).shape[0] == df.shape[0]
    assert np.all(np.equal(folds[0][:3], [3152, 31495, 18964]))
    assert np.all(np.equal(folds[-1][-3:], [22790, 36269, 7618]))

    plt.figure()
    for fold_idx in range(NUM_FOLDS):
        fold = folds[fold_idx]
        fold_counts = np.sum(targets[fold], axis=0)
        print(fold_counts)

        ax = plt.subplot(2, 5, 1 + fold_idx)
        ax.set_yscale('log')
        plt.bar(range(NUM_CLASSES), fold_counts)
        # print(fold[:10])
    plt.savefig('folds.png', bbox_inches='tight', pad_inches=0)


def make_splits():
    prefix = os.path.abspath(f'data')
    df = pd.read_csv('data/train_flat.csv')
    df.rename(columns={'image_name': 'path'}, inplace=True)
    folds = np.load(f'data/{NUM_FOLDS}_folds.npy')
    for i, fold in enumerate(folds):
        os.makedirs(f'data/fold{i}', exist_ok=True)
        train = df.iloc[list(set(range(df.shape[0])) - set(fold))].copy()
        train['path'] = train['path'].apply(lambda s: os.path.join(prefix, 'train-jpg', s) + '.jpg')
        train.to_csv(f'data/fold{i}/train.csv', index=False)
        val = df.iloc[fold].copy()
        val['path'] = val['path'].apply(lambda s: os.path.join(prefix, 'train-jpg', s) + '.jpg')
        val.to_csv(f'data/fold{i}/val.csv', index=False)

if __name__ == '__main__':
    flatten_train()
    make_folds()
    plot_folds()
    make_splits()
