"""
Extra augmentations for pytorch
"""

from imgaug import augmenters as iaa
import numpy as np
from PIL import Image
import random
import math
import cv2
import numbers


class RandomVerticalFlip(object):
    """Randomly vertically flips the given PIL.Image with given probability
    """
    def __init__(self, probability):
        self.probability = probability

    def __call__(self, img):
        img_array = np.array(img)
        flipper = iaa.Fliplr(self.probability)
        img_n = flipper.augment_image(img_array)

        return Image.fromarray(img_n, mode='RGB')


class Random90Rotation(object):
    """Randomly rotates image on [0, 90, 180, 270] degrees
    """

    def __call__(self, img):
        angles = [0, 90, 180, 270]
        random_angle = np.random.choice(angles)

        if random_angle == 0:
            return img

        img_array = np.array(img)

        augmentor = iaa.Affine(rotate=random_angle)

        img_n = augmentor.augment_image(img_array)

        return Image.fromarray(img_n, mode='RGB')


class D4(object):
    """Random transformation from D4 group
    """
    def __call__(self, img):
        if random.random() < 0.5:
            img = np.transpose(img, [1, 0, 2])

        if random.random() < 0.5:
            img = np.flipud(img)

        if random.random() < 0.5:
            img = np.fliplr(img)

        return np.copy(img)


class GaussianBlur(object):
    """Blurs image with blur in a range
    """
    def __init__(self, from_blur=0, to_blur=4):
        self.from_blur = from_blur
        self.to_blur = to_blur

    def __call__(self, img):
        img_array = np.array(img)

        blurer = iaa.GaussianBlur(sigma=(self.from_blur, self.to_blur))

        img_n = blurer.augment_image(img_array)

        return Image.fromarray(img_n, mode='RGB')


class Add(object):
    """Adds values within a range

    per_channel : bool, optional(default=False)
            Whether to use the same value for all channels (False)
            or to sample a new value for each channel (True).
            If this value is a float p, then for p percent of all images
            per_channel will be treated as True, otherwise as False.
    """
    def __init__(self, from_add=-10, to_add=+10, per_channel=0.5):
        self.from_add = from_add
        self.to_add = to_add
        self.per_channel = per_channel

    def __call__(self, img):
        img_array = np.array(img)

        adder = iaa.Add((self.from_add, self.to_add), per_channel=self.per_channel)

        img_n = adder.augment_image(img_array)

        return Image.fromarray(img_n, mode='RGB')


class Rotate(object):
    """Rotates image
    """
    def __init__(self, from_angle=-20, to_angle=+20, mode='reflect'):
        self.from_angle = from_angle
        self.to_angle = to_angle
        self.mode = mode

    def __call__(self, img):
        img_array = np.array(img)

        rotator = iaa.Affine(rotate=(self.from_angle, self.to_angle), mode=self.mode)
        img_n = rotator.augment_image(img_array)

        return Image.fromarray(img_n, mode='RGB')


class ContrastNormalization(object):
    """Changes contrast
    """
    def __init__(self, contrast_from=0.9, contrast_to=1.1, per_channel=0.5):
        self.contrast_from = contrast_from
        self.contrast_to = contrast_to
        self.per_channel = per_channel

    def __call__(self, img):
        img_array = np.array(img)

        contrastor = iaa.ContrastNormalization((self.contrast_from, self.contrast_to), self.per_channel)
        img_n = contrastor.augment_image(img_array)

        return Image.fromarray(img_n, mode='RGB')


class CenterCrop(object):
    """Crops the given np.array at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        h, w = img.shape[0], img.shape[1]
        # w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img[y1:y1 + th, x1:x1 + th, :].astype(np.int64)


class RandomCrop(object):
    """Crops the given np.array randomly to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        h, w = img.shape[0], img.shape[1]
        # w, h = img.size
        th, tw = self.size
        x1 = np.random.randint(0, w - tw - 1)
        y1 = np.random.randint(0, h - th - 1)
        return img[y1:y1 + th, x1:x1 + th, :].astype(np.int64)


class RandomSizedCrop(object):
    """Random crop the given np.array to a random size of (0.08 to 1.0) of the original size
    and and a random aspect ratio of 3/4 to 4/3 of the original aspect ratio
    This is popularly used to train the Inception networks
    size: size of the smaller edge
    interpolation: Default:  CV_INTER_LINEAR
    """

    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        for attempt in range(10):
            area = img.shape[0] * img.shape[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.shape[1] and h <= img.shape[0]:
                y1 = random.randint(0, img.shape[0] - h)
                x1 = random.randint(0, img.shape[1] - w)

                img = img[y1:y1+h, x1:x1+w, :]

                assert img.shape[0] == h
                assert img.shape[1] == w

                return cv2.resize(img, (self.size, self.size), interpolation=self.interpolation).astype(np.int64)

        # Fallback
        scale = cv2.resize(img, (self.size, self.size), interpolation=self.interpolation)

        crop = CenterCrop(self.size)
        result = crop(scale)

        return result.astype(np.int64)
