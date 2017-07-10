"""
Extra augmentations for pytorch
"""

from imgaug import augmenters as iaa
import numpy as np
from PIL import Image


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