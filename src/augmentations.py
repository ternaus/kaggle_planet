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
