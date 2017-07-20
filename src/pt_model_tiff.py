"""
Experiments with pytorch
"""

from torchvision import transforms
from torch import nn

from torch.nn import MultiLabelSoftMarginLoss

from torchvision.models import resnet50, resnet101, resnet152
from torchvision.models import densenet121, densenet161, densenet201

import torch.nn.functional as F
import utils

from sklearn.metrics import fbeta_score
import numpy as np
import argparse
from torch.optim import Adam
import data_loader
import augmentations
import models


def f2_score(y_true, y_pred):
    # fbeta_score throws a confusing error if inputs are not numpy arrays
    y_true, y_pred, = np.array(y_true), np.array(y_pred)
    # We need to use average='samples' here, any other average method will generate bogus results
    return fbeta_score(y_true, y_pred, beta=2, average='samples')


def validation(model, criterion, valid_loader):
    model.eval()
    losses = []
    f2_scores = []
    for inputs, targets in valid_loader:
        inputs = utils.variable(inputs, volatile=True)
        targets = utils.variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        losses.append(loss.data[0])
        f2_scores.append(f2_score(y_true=targets.data.cpu().numpy(), y_pred=F.sigmoid(outputs).data.cpu().numpy() > 0.2))
    valid_loss = np.mean(losses)  # type: float
    valid_f2 = np.mean(f2_scores)  # type: float
    print('Valid loss: {:.4f}, F2: {:.4f}'.format(valid_loss, valid_f2))
    return {'valid_loss': valid_loss, 'valid_f2': valid_f2}


def get_model(num_classes, model_type='resnet50'):
    if model_type == 'resnet50':
        model = resnet50(pretrained=True).cuda()
        model.fc = nn.Linear(model.fc.in_features, num_classes).cuda()
    elif model_type == 'resnet101':
        model = resnet101(pretrained=True).cuda()
        model.fc = nn.Linear(model.fc.in_features, num_classes).cuda()
    elif model_type == 'resnet152':
        model = resnet152(pretrained=True).cuda()
        model.fc = nn.Linear(model.fc.in_features, num_classes).cuda()
    elif model_type == 'densenet121':
        model = densenet121(pretrained=True).cuda()
        model.classifier = nn.Linear(model.classifier.in_features, num_classes).cuda()
    elif model_type == 'densenet161':
        model = densenet161(pretrained=True).cuda()
        model.classifier = nn.Linear(model.classifier.in_features, num_classes).cuda()
    elif model_type == 'densenet201':
        model = densenet201(pretrained=True).cuda()
        model.classifier = nn.Linear(model.classifier.in_features, num_classes).cuda()
    return model


def add_args(parser):
    arg = parser.add_argument
    arg('root', help='checkpoint root')
    arg('--batch-size', type=int, default=4)
    arg('--n-epochs', type=int, default=30)
    arg('--lr', type=float, default=0.0001)
    arg('--workers', type=int, default=8)
    arg('--fold', type=int)
    arg('--n-folds', type=int, default=10)
    arg('--clean', action='store_true')
    arg('--epoch-size', type=int)
    arg('--model', type=str)
    arg('--device-ids', type=str, help='For example 0,1 to run on two GPUs')


if __name__ == '__main__':
    random_state = 2016

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--mode', choices=['train', 'valid', 'predict_valid', 'predict_test'], default='train')
    add_args(parser)
    args = parser.parse_args()

    model_name = args.model

    batch_size = args.batch_size

    train_transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        augmentations.D4(),
        # augmentations.Rotate(),
        # augmentations.GaussianBlur(),
        augmentations.Add(-5, 5, per_channel=True),
        augmentations.ContrastNormalization(0.8, 1.2, per_channel=True),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_loader, valid_loader = data_loader.get_loaders_tiff(batch_size,
                                                              train_transform=train_transform,
                                                              fold=args.fold)

    num_classes = 17

    model = get_model(num_classes, model_name)

    fresh_params = model.classifier.parameters()

    # model = getattr(models, args.model)(num_classes=num_classes)
    # model = utils.cuda(model)

    if utils.cuda_is_available:
        if args.device_ids:
            device_ids = list(map(int, args.device_ids.split(',')))
        else:
            device_ids = None

        model = nn.DataParallel(model, device_ids=device_ids).cuda()

    criterion = MultiLabelSoftMarginLoss()

    #
    # train_loader, valid_loader = map(make_loader,
    #                                  [train_paths, valid_paths])
    # if root.exists() and args.clean:
    #     shutil.rmtree(str(root))
    # root.mkdir(exist_ok=True)
    # root.joinpath('params.json').write_text(
    #     json.dumps(vars(args), indent=True, sort_keys=True))

    train_kwargs = dict(
        args=args,
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        valid_loader=valid_loader,
        validation=validation,
        patience=4,
    )

    # if getattr(model, 'finetune', None):
    utils.train(
        init_optimizer=lambda lr: Adam(fresh_params, lr=lr),
        n_epochs=1,
        **train_kwargs)

    utils.train(
        init_optimizer=lambda lr: Adam(model.parameters(), lr=lr),
        **train_kwargs)
    # else:
    #     utils.train(
    #         init_optimizer=lambda lr: Adam(model.parameters(), lr=lr),
    #         **train_kwargs)

    # utils.train(
    #     init_optimizer=lambda lr: Adam(model.parameters(), lr=lr),
    #     args=args,
    #     model=model,
    #     criterion=criterion,
    #     train_loader=train_loader,
    #     valid_loader=valid_loader,
    #     validation=validation,
    #     # save_predictions=save_predictions,
    #     patience=3,
    # )
