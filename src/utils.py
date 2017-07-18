"""
Modification of the
https://github.com/lopuhin/mapillary-vistas-2017/blob/master/utils.py

by Konstantin Lopuhin
"""

from concurrent.futures import ThreadPoolExecutor
import json
from datetime import datetime
import glob
from itertools import islice
import functools
from pathlib import Path
from pprint import pprint
import random
import shutil

import matplotlib.pyplot as plt

import json_lines
import numpy as np
from PIL import Image
from sklearn.model_selection import KFold
import statprof
import torch
from torch import nn
from torch.autograd import Variable
from torchvision.transforms import ToTensor, Normalize, Compose
import tqdm
import tifffile as tiff
import cv2
import os
import data_loader


DATA_ROOT = Path(__file__).absolute().parent / 'data'


test_tif_folder = '../data/test-tif-v2'
test_jpg_folder = '../data/test-jpg'

train_tif_folder = '../data/train-tif-v2'
train_jpg_folder = '../data/train-jpg'


cuda_is_available = torch.cuda.is_available()


def variable(x, volatile=False):
    if isinstance(x, (list, tuple)):
        return [variable(y, volatile=volatile) for y in x]
    return cuda(Variable(x.cuda(async=True), volatile=volatile))


def cuda(x):
    return x.cuda() if cuda_is_available else x


img_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_image(path: Path) -> Image.Image:
    return Image.open(str(path)).convert('RGB')


# def load_image_tif(path: Path) -> np.array:
#     file_name = str(path).split('/')[-1].split('.')[0]
#
#     if 'train' in file_name:
#         im_tif = tiff.imread(os.path.join(train_tif_folder, file_name + '.tif'))
#         im_jpg = cv2.imread(os.path.join(train_jpg_folder, file_name + '.jpg'))
#     else:
#         im_tif = tiff.imread(os.path.join(test_tif_folder, file_name + '.tif'))
#         im_jpg = cv2.imread(os.path.join(test_jpg_folder, file_name + '.jpg'))
#
#     im_tif[:, :, 2] = im_tif[:, :, 3]  # Replace R channel with NIR
#
#     tuned_tif = data_loader.match_percentiles(im_tif, im_jpg)
#     return tuned_tif
#

def train_valid_split(args, img_paths):
    img_paths = np.array(sorted(img_paths))
    cv_split = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    img_folds = list(cv_split.split(img_paths))
    train_ids, valid_ids = img_folds[args.fold - 1]
    return img_paths[train_ids], img_paths[valid_ids]


def profile(fn):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        statprof.start()
        try:
            return fn(*args, **kwargs)
        finally:
            statprof.stop()
            statprof.display()
    return wrapped


def write_event(log, step: int, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()


# def train(args, model: nn.Module, criterion, *, train_loader, valid_loader,
#           validation, init_optimizer, save_predictions=None, n_epochs=None,
#           patience=2, max_lr_changes=2):
#     lr = args.lr
#     n_epochs = n_epochs or args.n_epochs
#     optimizer = init_optimizer(lr)
#
#     root = Path(args.root)
#     model_path = root / 'model.pt'
#     best_model_path = root / 'best-model.pt'
#     if model_path.exists():
#         state = torch.load(str(model_path))
#         epoch = state['epoch']
#         step = state['step']
#         best_valid_loss = state['best_valid_loss']
#         model.load_state_dict(state['model'])
#         print('Restored model, epoch {}, step {:,}'.format(epoch, step))
#     else:
#         epoch = 1
#         step = 0
#         best_valid_loss = float('inf')
#     lr_changes = 0
#
#     save = lambda ep: torch.save({
#         'model': model.state_dict(),
#         'epoch': ep,
#         'step': step,
#         'best_valid_loss': best_valid_loss
#     }, str(model_path))
#
#     report_each = 10
#     save_prediction_each = report_each * 10
#     log = root.joinpath('train.log').open('at', encoding='utf8')
#     valid_losses = []
#     lr_reset_epoch = epoch
#     for epoch in range(epoch, n_epochs + 1):
#         model.train()
#         random.seed()
#         tq = tqdm.tqdm(total=(args.epoch_size or
#                               len(train_loader) * args.batch_size))
#         tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
#         losses = []
#         tl = train_loader
#         if args.epoch_size:
#             tl = islice(tl, args.epoch_size // args.batch_size)
#         try:
#             mean_loss = 0
#             for i, (inputs, targets) in enumerate(tl):
#                 inputs, targets = variable(inputs), variable(targets)
#                 outputs = model(inputs)
#                 loss = criterion(outputs, targets)
#                 optimizer.zero_grad()
#                 batch_size = inputs.size(0)
#                 (batch_size * loss).backward()
#                 optimizer.step()
#                 step += 1
#                 tq.update(batch_size)
#                 losses.append(loss.data[0])
#                 mean_loss = np.mean(losses[-report_each:])
#                 tq.set_postfix(loss='{:.3f}'.format(mean_loss))
#                 if i and i % report_each == 0:
#                     write_event(log, step, loss=mean_loss)
#                     if save_predictions and i % save_prediction_each == 0:
#                         p_i = (i // save_prediction_each) % 5
#                         save_predictions(root, p_i, inputs, targets, outputs)
#             write_event(log, step, loss=mean_loss)
#             tq.close()
#             save(epoch + 1)
#             valid_metrics = validation(model, criterion, valid_loader)
#             write_event(log, step, **valid_metrics)
#             valid_loss = valid_metrics['valid_loss']
#             valid_losses.append(valid_loss)
#             if valid_loss < best_valid_loss:
#                 best_valid_loss = valid_loss
#                 shutil.copy(str(model_path), str(best_model_path))
#             elif (patience and epoch - lr_reset_epoch > patience and
#                   min(valid_losses[-patience:]) > best_valid_loss):
#                 # "patience" epochs without improvement
#                 lr_changes +=1
#                 if lr_changes > max_lr_changes:
#                     break
#                 lr /= 5
#                 lr_reset_epoch = epoch
#                 optimizer = init_optimizer(lr)
#         except KeyboardInterrupt:
#             tq.close()
#             print('Ctrl+C, saving snapshot')
#             save(epoch)
#             print('done.')
#             return


def train(args,
          model: nn.Module,
          criterion, *, train_loader,
          valid_loader,
          validation,
          init_optimizer,
          save_predictions=None,
          n_epochs=None,
          patience=2):
    lr = args.lr
    n_epochs = n_epochs or args.n_epochs
    optimizer = init_optimizer(lr)

    root = Path(args.root)
    model_path = root / 'model_{}.pt'.format(args.fold)
    best_model_path = root / 'best-model_{}.pt'.format(args.fold)
    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch']
        step = state['step']
        best_valid_loss = state['best_valid_loss']
        model.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
    else:
        epoch = 1
        step = 0
        best_valid_loss = float('inf')

    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
        'best_valid_loss': best_valid_loss
    }, str(model_path))

    report_each = 10
    save_prediction_each = report_each * 20
    log = root.joinpath('train_{}.log'.format(args.fold)).open('at', encoding='utf8')
    valid_losses = []
    lr_reset_epoch = epoch
    for epoch in range(epoch, n_epochs + 1):
        model.train()
        random.seed()
        tq = tqdm.tqdm(total=(args.epoch_size or
                              len(train_loader) * args.batch_size))
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        losses = []
        tl = train_loader
        if args.epoch_size:
            tl = islice(tl, args.epoch_size // args.batch_size)
        try:
            mean_loss = 0
            for i, (inputs, targets) in enumerate(tl):
                inputs, targets = variable(inputs), variable(targets)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                batch_size = inputs.size(0)
                (batch_size * loss).backward()
                optimizer.step()
                step += 1
                tq.update(batch_size)
                losses.append(loss.data[0])
                mean_loss = np.mean(losses[-report_each:])
                tq.set_postfix(loss='{:.3f}'.format(mean_loss))
                if i and i % report_each == 0:
                    write_event(log, step, loss=mean_loss)
                    if save_predictions and i % save_prediction_each == 0:
                        p_i = (i // save_prediction_each) % 5
                        save_predictions(root, p_i, inputs, targets, outputs)
            write_event(log, step, loss=mean_loss)
            tq.close()
            save(epoch + 1)
            valid_metrics = validation(model, criterion, valid_loader)
            write_event(log, step, **valid_metrics)
            valid_loss = valid_metrics['valid_loss']
            valid_losses.append(valid_loss)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                shutil.copy(str(model_path), str(best_model_path))
            elif (patience and epoch - lr_reset_epoch > patience and
                  min(valid_losses[-patience:]) > best_valid_loss):
                # "patience" epochs without improvement
                lr /= 5
                lr_reset_epoch = epoch
                optimizer = init_optimizer(lr)
        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save(epoch)
            print('done.')
            return


def load_best_model(model: nn.Module, root: Path) -> None:
    state = torch.load(str(root / 'best-model.pt'))
    model.load_state_dict(state['model'])
    print('Loaded model from epoch {epoch}, step {step:,}'.format(**state))


def batches(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


def imap_fixed_output_buffer(fn, it, threads: int):
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = []
        max_futures = threads + 1
        for x in it:
            while len(futures) >= max_futures:
                future, futures = futures[0], futures[1:]
                yield future.result()
            futures.append(executor.submit(fn, x))
        for future in futures:
            yield future.result()

