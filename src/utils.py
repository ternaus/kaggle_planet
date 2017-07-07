"""
@author Konstantin Lopuhin
https://github.com/lopuhin/mapillary-vistas-2017/blob/master/utils.py

"""
from concurrent.futures import ThreadPoolExecutor
import json
from datetime import datetime

import numpy as np
from PIL import Image
from sklearn.model_selection import KFold

import torch
from torch.autograd import Variable
from torchvision.transforms import ToTensor, Normalize, Compose


cuda_is_available = torch.cuda.is_available()


def variable(x, volatile=False):
    if isinstance(x, (list, tuple)):
        return [variable(y, volatile=volatile) for y in x]
    return cuda(Variable(x, volatile=volatile))


def cuda(x):
    return x.cuda() if cuda_is_available else x


img_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_image(path):
    return Image.open(str(path)).convert('RGB')


def train_valid_split(args, img_paths):
    img_paths = np.array(sorted(img_paths))
    cv_split = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    img_folds = list(cv_split.split(img_paths))
    train_ids, valid_ids = img_folds[args.fold - 1]
    return img_paths[train_ids], img_paths[valid_ids]


def write_event(log, step, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()


def add_args(parser):
    arg = parser.add_argument
    arg('--batch-size', type=int, default=4)
    arg('--n-epochs', type=int, default=100)
    arg('--lr', type=float, default=0.0001)
    arg('--workers', type=int, default=2)
    arg('--fold', type=int, default=1)
    arg('--clean', action='store_true')
    arg('--epoch-size', type=int)


def load_best_model(model, root):
    state = torch.load(str(root / 'best-model.pt'))
    model.load_state_dict(state['model'])
    print('Loaded model from epoch {epoch}, step {step:,}'.format(**state))


def batches(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


def imap_fixed_output_buffer(fn, it, threads):
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
